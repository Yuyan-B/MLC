# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's TRL library.
# https://github.com/huggingface/trl/blob/v0.8.0/trl/trainer/dpo_trainer.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from collections import defaultdict
from contextlib import nullcontext
from types import MethodType
from typing import TYPE_CHECKING, Literal, Optional, Union

import torch
import torch.nn.functional as F
from transformers import Trainer
from trl import DPOTrainer
from trl.trainer import disable_dropout_in_model
from typing_extensions import override

from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler, get_batch_logps, nested_detach
import os
import re
import copy
import json

if TYPE_CHECKING:
    from transformers import PreTrainedModel, ProcessorMixin

    from ...hparams import FinetuningArguments


class CustomDPOTrainer(DPOTrainer):
    def __init__(
        self,
        model: Union["PreTrainedModel", torch.nn.Module],
        ref_model: Optional[Union["PreTrainedModel", torch.nn.Module]],
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        disable_dropout: bool = True,
        **kwargs,
    ):
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")

        if disable_dropout:
            disable_dropout_in_model(model)
            if ref_model is not None:
                disable_dropout_in_model(ref_model)

        self.finetuning_args = finetuning_args
        self.f_divergence_type = "reverse_kl"
        self.reference_free = False
        self.use_dpo_data_collator = True  # hack to avoid warning
        self.generate_during_eval = False  # disable at evaluation
        self.label_pad_token_id = IGNORE_INDEX
        self.padding_value = 0
        self.is_encoder_decoder = model.config.is_encoder_decoder
        self.precompute_ref_log_probs = False
        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False
        self._peft_has_been_casted_to_bf16 = False

        self.ref_model = ref_model
        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        # dpo hyperparams
        self.beta = finetuning_args.pref_beta
        self.loss_type = finetuning_args.pref_loss
        self.ftx_gamma = finetuning_args.pref_ftx
        self.label_smoothing = finetuning_args.dpo_label_smoothing
        self.simpo_gamma = finetuning_args.simpo_gamma
        self.ld_alpha = finetuning_args.ld_alpha

        hidden_size = model.config.hidden_size
        representation_dim=finetuning_args.extractor_dim
        if finetuning_args.aux_enable:
            model.add_module("extractor", torch.nn.Linear(hidden_size, representation_dim))
            self.extractor = model.extractor  

        Trainer.__init__(self, model=model, **kwargs)
        self.model_accepts_loss_kwargs = False  # overwrite trainer's default behavior
        if not hasattr(self, "accelerator"):
            raise AttributeError("Please update `transformers`.")

        warnings.simplefilter("ignore")  # remove gc warnings on ref model

        if ref_model is not None:
            if self.is_deepspeed_enabled:
                if not (
                    getattr(ref_model, "is_loaded_in_8bit", False) or getattr(ref_model, "is_loaded_in_4bit", False)
                ):  # quantized models are already set on the correct device
                    self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
                self.ref_model.eval()

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        self.aux_enable = finetuning_args.aux_enable
        # if self.aux_enable:
        
        self.aux_every = finetuning_args.aux_every
        self.hs_layer_index = finetuning_args.hs_layer_index
        self.lambda_multilinconsistency = finetuning_args.lambda_multilinconsistency
        self.consistency_tau = finetuning_args.consistency_tau
            
    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False) -> None:

        resolved_dir = output_dir or getattr(self.args, "output_dir", None)
        if resolved_dir is None:
            raise ValueError("save_model: output_dir not provided, and TrainingArguments.output_dir is also None.")

        # if aux_enable is not enabled: directly use the parent class logic
        if not getattr(self, "aux_enable", False):
            try:
                return super().save_model(output_dir=resolved_dir, _internal_call=_internal_call)
            except TypeError:
                return super().save_model(output_dir=resolved_dir)

        # unwrap to handle ZeRO/FSDP
        wrapped_model = getattr(self, "model_wrapped", None) or self.model
        model_unwrapped = self.accelerator.unwrap_model(self.model)

        extractor_state = None
        extractor__backup = None
        had_extractor = hasattr(model_unwrapped, "extractor") and (model_unwrapped.extractor is not None)

        if had_extractor:
            # === A) only aggregate extractor (compatible with ZeRO-3 / non-ZeRO) ===
            try:
                import deepspeed
                is_deepspeed = hasattr(wrapped_model, "zero_optimization") or \
                            wrapped_model.__class__.__name__.lower().startswith("deepspeed")
            except Exception:
                deepspeed = None
                is_deepspeed = False

            if is_deepspeed and deepspeed is not None:
                # in ZeRO-3, aggregate the parameters of extractor to rank0
                params = list(model_unwrapped.extractor.parameters())
                with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                    if self.is_world_process_zero():
                        # note: at this point, the tensor has been gathered, so it can be safely moved to CPU
                        extractor_state = {k: v.detach().cpu() for k, v in model_unwrapped.extractor.state_dict().items()}
            else:
                # non-DS (or non-ZeRO-3) scenario: directly take the state_dict of the submodule
                extractor_state = {k: v.detach().cpu() for k, v in model_unwrapped.extractor.state_dict().items()}

            # temporarily remove the submodule, avoid being saved together with the main model
            extractor_backup = model_unwrapped.extractor
            delattr(model_unwrapped, "extractor")


        # save the main model (without extractor)
        try:
            try:
                super().save_model(output_dir=resolved_dir, _internal_call=_internal_call)
            except TypeError:
                super().save_model(output_dir=resolved_dir)
        finally:
            if extractor_backup is not None:
                setattr(model_unwrapped, "extractor", extractor_backup)

        # only save the extractor in the main process
        if (extractor_state is not None) and self.is_world_process_zero():
            extractor_dir = os.path.join(resolved_dir, "extractor")
            os.makedirs(extractor_dir, exist_ok=True)
            extractor_path = os.path.join(extractor_dir, "extractor.pt")
            torch.save(extractor_state, extractor_path)

            # save a minimum configuration, for subsequent loading/re-hanging
            cfg = {
                "hs_layer_index": int(getattr(self, "hs_layer_index", -1)),
                "extractor_dim": int(getattr(self.finetuning_args, "extractor_dim", -1)),
            }
            with open(os.path.join(extractor_dir, "config.json"), "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)



    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    @override
    def get_batch_samples(self, *args, **kwargs):
        r"""Replace the method of DPO Trainer with the one of the standard Trainer."""
        return Trainer.get_batch_samples(self, *args, **kwargs)

    def odds_ratio_loss(self, chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor") -> "torch.Tensor":
        r"""Compute ORPO's odds ratio (OR) loss for batched log probabilities of the policy model."""
        log_odds = (chosen_logps - rejected_logps) - (
            torch.log1p(-torch.exp(chosen_logps)) - torch.log1p(-torch.exp(rejected_logps))
        )
        sft_loss = -chosen_logps
        odds_ratio_loss = -F.logsigmoid(log_odds)
        orpo_loss = sft_loss + self.beta * odds_ratio_loss
        return orpo_loss

    def simpo_loss(self, chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor") -> "torch.Tensor":
        r"""Compute SimPO loss for batched log probabilities of the policy model."""
        pi_logratios = chosen_logps - rejected_logps
        gamma_logratios = self.simpo_gamma / self.beta
        logits = pi_logratios - gamma_logratios
        simpo_loss = -F.logsigmoid(self.beta * logits)
        return simpo_loss

    def compute_preference_loss(
        self,
        policy_chosen_logps: "torch.Tensor",
        policy_rejected_logps: "torch.Tensor",
        reference_chosen_logps: Optional["torch.Tensor"],
        reference_rejected_logps: Optional["torch.Tensor"],
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        r"""Compute loss for preference learning."""
        if not self.finetuning_args.use_ref_model:
            if self.loss_type == "orpo":
                losses = self.odds_ratio_loss(policy_chosen_logps, policy_rejected_logps)
            elif self.loss_type == "simpo":
                losses = self.simpo_loss(policy_chosen_logps, policy_rejected_logps)
            else:
                raise NotImplementedError(f"Unknown loss type: {self.loss_type}.")

            chosen_rewards = self.beta * policy_chosen_logps.to(self.accelerator.device).detach()
            rejected_rewards = self.beta * policy_rejected_logps.to(self.accelerator.device).detach()
        else:
            losses, chosen_rewards, rejected_rewards = self.dpo_loss(
                policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps
            )

        return losses, chosen_rewards, rejected_rewards

    def _compute_aux_loss(self, multilin_hidden_states,multilin_attention_mask,len_lans):
        
        BxL, T, H = multilin_hidden_states.shape
        L=len_lans
        B=BxL//L
        assert BxL == B * L, f"shape mismatch: got {BxL} vs B*L={B*L}"

        x = multilin_hidden_states
        m = multilin_attention_mask

        x = x.reshape(B, L, T, H)
        if m is not None:
            m = m.contiguous().reshape(B, L, T)

        if m is not None:
            # the last valid position idx = sum(mask) - 1
            last_idx = (m.sum(dim=-1) - 1).clamp(min=0).to(torch.long)                  # (B, L)
        else:
            # if there is no mask, it degenerates to taking the last one
            last_idx = torch.full((B, L), T - 1, dtype=torch.long, device=x.device)
        
        # if self.accelerator.is_main_process:
        #     print("=============last_idx=============")
        #     print(last_idx)


        idx = last_idx.unsqueeze(-1).unsqueeze(-1).expand(B, L, 1, H)    # (B, L, 1, H), long dtype
        last_h = x.gather(dim=2, index=idx).squeeze(2)  
        


        z_dense = self.extractor(last_h)  # (B, L, D)
        
 
        z=z_dense        
            

        zc = z/(z.norm(dim=-1, keepdim=True)+1e-12)

        with torch.cuda.amp.autocast(enabled=False):
            Z=zc.to(torch.float32)

            eps = 1e-12

            U_V, svals, W_V = torch.linalg.svd(Z, full_matrices=False)

            if self.accelerator.is_main_process:
                print("=============ori svd=============")
                print(svals.detach().float().cpu())

            #
            r = svals.size(-1)


            stop = svals[..., :1].detach()
            S_scaled = svals / (stop + 1e-12)           # s1_scaled ≈ 1, tail ≪ 1

            tau = float(getattr(self, "consistency_tau", 0.3))   # suggest to tune 0.2~1.0
            logits_rank = (S_scaled / tau).reshape(-1, r).float()
            targets_rank = torch.zeros(logits_rank.size(0), dtype=torch.long, device=logits_rank.device)
            

        loss_cons = torch.nn.functional.cross_entropy(logits_rank,targets_rank)
        if self.accelerator.is_main_process:
            print("=============loss_cons=============")
            print(loss_cons.detach().float().cpu())

        w_wlc  = float(getattr(self, "lambda_multilinconsistency", 1.0))

        aux_loss = w_wlc * loss_cons 

        # metrics for monitoring
        metrics = {
            "mlc_losses":             aux_loss.detach()
        }
        return aux_loss, metrics

    @override
    def concatenated_forward(
        self, model: "PreTrainedModel", batch: dict[str, "torch.Tensor"], is_ref_model: bool = False
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor","torch.Tensor","torch.Tensor","int"]:
        r"""Compute the sum log probabilities of the labels under given logits if loss_type is not IPO, ORPO or SimPO.

        Otherwise the average log probabilities.
        """
        if "num_pairs" in batch:
            num_pairs_t = batch.get("num_pairs")
            # num_pairs_t may be on GPU, here convert to python int
            num_pairs = int(num_pairs_t.item() if hasattr(num_pairs_t, "item") else int(num_pairs_t))
        else:
            # fallback: old data stream or not set, fall back to the original logic
            num_pairs = batch["input_ids"].size(0) // 2
        pref_bs = 2 * num_pairs

        # 2) when using reference model, disconnect the gradient
        if self.finetuning_args.use_ref_model:
            batch = nested_detach(batch, clone=True)  # avoid error

        # 3) forward; here will do forward for the whole batch, but only slice the first 2*num_pairs later
        forward_kwargs = {k: v for k, v in batch.items() if k != "num_pairs"}
        
        outputs = model(**forward_kwargs, return_dict=True, use_cache=False,output_hidden_states=True)


        all_logits: torch.Tensor = outputs.logits.to(torch.float32)

        # 4) only compute logps for the preference sample interval (to avoid the risk of division by zero for aux)
        pref_logits = all_logits[:pref_bs]
        pref_labels = batch["labels"][:pref_bs]

        all_logps, valid_length = get_batch_logps(
            logits=pref_logits,
            labels=pref_labels,
            ld_alpha=(self.ld_alpha if not is_ref_model else None),
        )

        if self.loss_type in ["ipo", "orpo", "simpo"]:
            all_logps = all_logps / valid_length

        # 5) split chosen / rejected
        chosen_logps   = all_logps[:num_pairs]
        rejected_logps = all_logps[num_pairs:pref_bs]

        chosen_logits   = pref_logits[:num_pairs]
        rejected_logits = pref_logits[num_pairs:pref_bs]

        chosen_length, _ = valid_length.split(num_pairs, dim=0)


        if num_pairs == batch["input_ids"].size(0) // 2:
            len_lans = 0  
            multilin_hidden_states = None
            multilin_attention_mask = None
        else:
            len_lans=len(batch["input_ids"])//num_pairs-2

            hs = outputs.hidden_states[self.hs_layer_index][pref_bs:]
            multilin_hidden_states = hs           # (B_aux, T, H)
            multilin_attention_mask=batch["attention_mask"][pref_bs:]

        if self.loss_type in ["ipo", "orpo", "simpo"]:
            return chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_logps,multilin_hidden_states,multilin_attention_mask,len_lans
        else:
            return chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_logps / chosen_length,multilin_hidden_states,multilin_attention_mask,len_lans
   
    @override
    def compute_reference_log_probs(
        self, model: "PreTrainedModel", batch: dict[str, "torch.Tensor"]
    ) -> tuple[Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Compute log probabilities of the reference model."""
        if not self.finetuning_args.use_ref_model:
            return None, None

        if self.ref_model is None:
            ref_model = model
            ref_context = self.accelerator.unwrap_model(model).disable_adapter()
        else:
            ref_model = self.ref_model
            ref_context = nullcontext()

        with torch.no_grad(), ref_context:
            reference_chosen_logps, reference_rejected_logps, *_ = self.concatenated_forward(
                ref_model, batch, is_ref_model=True
            )

        return reference_chosen_logps, reference_rejected_logps

    @override
    def get_batch_loss_metrics(
        self,
        model: "PreTrainedModel",
        batch: dict[str, "torch.Tensor"],
        train_eval: Literal["train", "eval"] = "train",
    ) -> tuple["torch.Tensor", dict[str, "torch.Tensor"]]:
        r"""Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_chosen_logps_avg,
            multilin_hidden_states,
            multilin_attention_mask,
            len_lans
        ) = self.concatenated_forward(model, batch)

        
        reference_chosen_logps, reference_rejected_logps = self.compute_reference_log_probs(model, batch)


        losses, chosen_rewards, rejected_rewards = self.compute_preference_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
        sft_loss = -policy_chosen_logps_avg
        if self.ftx_gamma > 1e-6:
            losses += self.ftx_gamma * sft_loss

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().item()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().item()
        metrics[f"{prefix}rewards/accuracies"] = (chosen_rewards > rejected_rewards).float().mean().item()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().item()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.mean().item()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.mean().item()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.mean().item()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.mean().item()
        if self.loss_type == "orpo":
            metrics[f"{prefix}sft_loss"] = sft_loss.mean().item()
            metrics[f"{prefix}odds_ratio_loss"] = ((losses - sft_loss) / self.beta).mean().item()

        total_loss, metrics = losses.mean(), metrics  # original
        metrics.update({f"{prefix}dpo_losses": losses.mean().item()})
        if self.aux_enable and len_lans and (self.state.global_step % getattr(self, "aux_every", 1) == 0):
            aux_loss, m = self._compute_aux_loss(multilin_hidden_states, multilin_attention_mask, len_lans)
            if aux_loss is not None:
                total_loss = total_loss + aux_loss
                metrics.update({f"{prefix}{k}": v.item() if torch.is_tensor(v) else v for k, v in m.items()})

        return total_loss, metrics

        
    @override
    def compute_loss(
        self, model: "PreTrainedModel", inputs: dict[str, "torch.Tensor"], return_outputs: bool = False, **kwargs
    ) -> Union["torch.Tensor", tuple["torch.Tensor", list["torch.Tensor"]]]:
        r"""Subclass and override to accept extra kwargs."""
        return super().compute_loss(model, inputs, return_outputs)

    @override
    def log(self, logs: dict[str, float], *args, **kwargs) -> None:
        r"""Log `logs` on the various objects watching training, including stored metrics."""
        # logs either has "loss" or "eval_loss"
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        key_list, metric_list = [], []
        for key, metrics in self._stored_metrics[train_eval].items():
            key_list.append(key)
            metric_list.append(torch.tensor(metrics, dtype=torch.float).to(self.accelerator.device).mean().item())

        del self._stored_metrics[train_eval]
        if len(metric_list) < 10:  # pad to for all reduce
            for i in range(10 - len(metric_list)):
                key_list.append(f"dummy_{i}")
                metric_list.append(0.0)

        metric_list = torch.tensor(metric_list, dtype=torch.float).to(self.accelerator.device)
        metric_list = self.accelerator.reduce(metric_list, "mean").tolist()
        for key, metric in zip(key_list, metric_list):  # add remaining items
            if not key.startswith("dummy_"):
                logs[key] = metric

        return Trainer.log(self, logs, *args, **kwargs)
