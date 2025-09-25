# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
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

import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler
import torch.nn.functional as F


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments


logger = logging.get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        gen_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: PreTrainedTokenizer = kwargs.get("tokenizer")

        super().__init__(**kwargs)
        hidden_size = self.model.config.hidden_size
        representation_dim=finetuning_args.extractor_dim
        if finetuning_args.aux_enable:
            self.model.add_module("extractor", torch.nn.Linear(hidden_size, representation_dim))
            try:
                self.model.extractor.to(self.model.device)
            except AttributeError:
                self.model.extractor.to(next(self.model.parameters()).device)
            self.extractor = self.model.extractor  
            self._reinitialize_new_modules()

        
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        self.aux_enable = finetuning_args.aux_enable
        self.aux_every = finetuning_args.aux_every
        self.hs_layer_index = finetuning_args.hs_layer_index
        self.lambda_multilinconsistency = finetuning_args.lambda_multilinconsistency
        self.consistency_tau = finetuning_args.consistency_tau


    def _reinitialize_new_modules(self):

        import torch.nn as nn
        
        if hasattr(self.model, 'extractor'):
            # use the standard initialization method
            for module in self.model.extractor.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

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
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    # @override
    # def compute_loss(self, model, inputs, *args, **kwargs):
    #     return super().compute_loss(model, inputs, *args, **kwargs)
    @override
    def compute_loss(self, model, inputs, *args, **kwargs):


        aux_enable = bool(getattr(self, "aux_enable", False))
        aux_every  = max(1, int(getattr(self, "aux_every", 1)))
        do_aux_now = aux_enable and ("num_pairs" in inputs) and (self.state.global_step % aux_every == 0)

        lang_keys = [k for k in inputs.keys() if k.startswith("prompt_input_ids_")]
        langs = sorted([k.split("prompt_input_ids_")[1] for k in lang_keys])  # fixed order
        device = getattr(self.model, "device", next(self.model.parameters()).device)

        aux_input_ids_list = []
        aux_attn_mask_list = []

        if do_aux_now:
            for lan in langs:
                ids_k = f"prompt_input_ids_{lan}"
                msk_k = f"prompt_attention_mask_{lan}"
                if ids_k not in inputs or msk_k not in inputs:
                    raise KeyError(f"[AUX] Missing {ids_k} or {msk_k} in inputs.")

                ids = inputs[ids_k]
                msk = inputs[msk_k]

                if ids.dtype != torch.long:
                    ids = ids.to(dtype=torch.long)
                if msk.dtype not in (torch.long, torch.bool):
                    msk = msk.to(dtype=torch.long)

                if ids.size(1) == 0:
                    bos = getattr(self.processing_class, "bos_token_id", None)
                    pad = getattr(self.processing_class, "pad_token_id", None)
                    fill_id = bos if bos is not None else (pad if pad is not None else 0)
                    B = ids.size(0)
                    ids = torch.full((B, 1), fill_id, dtype=torch.long, device=ids.device)
                    msk = torch.ones((B, 1), dtype=torch.long, device=msk.device)

                aux_input_ids_list.append(ids.to(device, non_blocking=True))
                aux_attn_mask_list.append(msk.to(device, non_blocking=True))


            def right_pad_to(t: torch.Tensor, L: int, value: int):
                cur = t.size(1)
                if cur == L: return t
                if cur < L:  return F.pad(t, (0, L - cur), value=value)
                return t[:, :L]

            target_len = max(
                max(x.size(1) for x in aux_input_ids_list),
                max(x.size(1) for x in aux_attn_mask_list),
            )
            pad_id = getattr(self.processing_class, "pad_token_id", None) \
                    or getattr(self.tokenizer, "pad_token_id", None) \
                    or getattr(self.tokenizer, "eos_token_id", None) or 0

            aux_input_ids_list  = [ right_pad_to(t, target_len, pad_id) for t in aux_input_ids_list ]
            aux_attn_mask_list  = [ right_pad_to(t, target_len, 0)      for t in aux_attn_mask_list ]

            aux_input_ids   = torch.cat(aux_input_ids_list,   dim=0)  # (B_aux, T_aux)
            aux_attn_mask   = torch.cat(aux_attn_mask_list,   dim=0)  # (B_aux, T_aux)
            B_aux           = aux_input_ids.size(0)
        else:
            B_aux = 0


        sft_input_ids     = inputs["input_ids"].to(device)
        sft_attn_mask     = inputs.get("attention_mask", torch.ones_like(sft_input_ids)).to(device)
        sft_labels        = inputs.get("labels", None)
        B_sft             = sft_input_ids.size(0)


        if B_aux > 0:

            T_sft = sft_input_ids.size(1)
            T_aux = aux_input_ids.size(1)

            max_len = max(T_sft, T_aux)
            pad_id  = getattr(self.processing_class, "pad_token_id", None) \
                    or getattr(self.tokenizer, "pad_token_id", None) \
                    or getattr(self.tokenizer, "eos_token_id", None) or 0

            def pad_to_len(ids, attn, L, pad_id):
                if ids.size(1) < L:
                    ids  = F.pad(ids,  (0, L - ids.size(1)),  value=pad_id)
                    attn = F.pad(attn, (0, L - attn.size(1)), value=0)
                elif ids.size(1) > L:
                    ids  = ids[:, :L]
                    attn = attn[:, :L]
                return ids, attn

            sft_input_ids, sft_attn_mask = pad_to_len(sft_input_ids, sft_attn_mask, max_len, pad_id)
            aux_input_ids, aux_attn_mask = pad_to_len(aux_input_ids, aux_attn_mask, max_len, pad_id)

            merged_input_ids   = torch.cat([sft_input_ids, aux_input_ids], dim=0)     # (B_sft+B_aux, L)
            merged_attn_mask   = torch.cat([sft_attn_mask, aux_attn_mask], dim=0)     # (B_sft+B_aux, L)

            def pad_labels_to_len(labels, L, pad_value=-100):
                if labels.size(1) < L:
                    labels = F.pad(labels, (0, L - labels.size(1)), value=pad_value)
                elif labels.size(1) > L:
                    labels = labels[:, :L]
                return labels

            if sft_labels is not None:
                sft_labels = sft_labels.to(device)
                sft_labels = pad_labels_to_len(sft_labels, max_len, pad_value=-100)
                ignore_labels = torch.full(
                    (aux_input_ids.size(0), max_len),  # B_aux × max_len
                    -100, dtype=sft_labels.dtype, device=device
                )
                merged_labels = torch.cat([sft_labels, ignore_labels], dim=0)

            
            else:
                merged_labels = None
        else:
            merged_input_ids = sft_input_ids
            merged_attn_mask = sft_attn_mask
            merged_labels    = sft_labels


        outputs = model(
            input_ids=merged_input_ids,
            attention_mask=merged_attn_mask,
            labels=merged_labels,                 
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )


        sft_loss = outputs.loss if hasattr(outputs, "loss") else None
        if self.accelerator.is_main_process:
            print("=============sft_loss=============")
            print(sft_loss.detach().float().cpu())


        if do_aux_now:

            hs_all = outputs.hidden_states[self.hs_layer_index]          # (B_sft+B_aux, L, H)
            hs_aux = hs_all[B_sft:B_sft+B_aux]                  
            attn_aux = merged_attn_mask[B_sft:B_sft+B_aux]

            aux_loss, aux_metrics = self._compute_aux_loss(
                multilin_hidden_states=hs_aux,
                multilin_attention_mask=attn_aux,
                len_lans=len(langs)
            )
            total_loss = sft_loss + aux_loss

            # log
            if self.args.logging_steps and (self.state.global_step % self.args.logging_steps == 0):
                self.log({
                    "sft_losses": float(sft_loss.item()),
                    "mlc_losses": float(getattr(aux_metrics["mlc_loss"], "item", lambda: aux_metrics["mlc_loss"])()),
                })
            return total_loss

        else:
            
            if self.args.logging_steps and (self.state.global_step % self.args.logging_steps == 0):
                self.log({"sft_losses": float(sft_loss.item())})
            return sft_loss

    @override
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False) -> None:

        resolved_dir = output_dir or getattr(self.args, "output_dir", None)
        if resolved_dir is None:
            raise ValueError("save_model: output_dir not provided, and TrainingArguments.output_dir is also empty.")

        # if aux_enable is not enabled: directly use the parent class logic
        if not getattr(self, "aux_enable", False):
            try:
                return super().save_model(output_dir=resolved_dir, _internal_call=_internal_call)
            except TypeError:
                return super().save_model(output_dir=resolved_dir)

        # unwrap to handle ZeRO/FSDP
        model_unwrapped = self.accelerator.unwrap_model(self.model)

        # pre-aggregate extractor weights (if exists)
        extractor_state = None
        had_sae = hasattr(model_unwrapped, "extractor") and (model_unwrapped.extractor is not None)
        if had_sae:
            # use accelerate to aggregate sub-module parameters into a complete state_dict
            extractor_state = self.accelerator.get_state_dict(model_unwrapped.extractor)

            # temporarily remove from the main model, to avoid being saved together
            extractor_backup = model_unwrapped.extractor
            delattr(model_unwrapped, "extractor")
        else:
            extractor_backup = None

        # save the main model (without extractor)
        try:
            try:
                super().save_model(output_dir=resolved_dir, _internal_call=_internal_call)
            except TypeError:
                super().save_model(output_dir=resolved_dir)
        finally:
            # restore the attribute
            if extractor_backup is not None:
                setattr(model_unwrapped, "extractor", extractor_backup)

        # only save the extractor in the main process
        if had_sae and (extractor_state is not None) and self.is_world_process_zero():
            extractor_dir = os.path.join(resolved_dir, "extractor")
            os.makedirs(extractor_dir, exist_ok=True)
            extractor_path = os.path.join(extractor_dir, "extractor.pt")
            torch.save(extractor_state, extractor_path)

            # save a minimum configuration, for subsequent loading/re-hanging
            cfg = {
                "hs_layer_index": getattr(self, "hs_layer_index", -1),
                "extractor_dim": getattr(self.finetuning_args, "extractor_dim", -1),
                "lambda_multilinconsistency": getattr(self, "lambda_multilinconsistency", 0.0)
            }
            with open(os.path.join(extractor_dir, "config.json"), "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)

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
            last_idx = (m.sum(dim=-1) - 1).clamp(min=0).to(torch.long)                  # (B, L)
        else:
            last_idx = torch.full((B, L), T - 1, dtype=torch.long, device=x.device)
        

        idx = last_idx.unsqueeze(-1).unsqueeze(-1).expand(B, L, 1, H)    # (B, L, 1, H), long dtype
        last_h = x.gather(dim=2, index=idx).squeeze(2)  
        

        # z_dense,x_hat = self.extractor(last_h)  # (B, L, D)
        z_dense = self.extractor(last_h)

 
        z=z_dense
     

        zc = z/(z.norm(dim=-1, keepdim=True)+1e-12)


        with torch.cuda.amp.autocast(enabled=False):
            Z=zc.to(torch.float32)

            eps = 1e-12

            U_V, svals, W_V = torch.linalg.svd(Z, full_matrices=False)

            if self.accelerator.is_main_process:
                print("=============ori svd=============")
                print(svals.detach().float().cpu())


            r = svals.size(-1)
            stop = svals[..., :1].detach()
            S_scaled = svals[..., :r] / (stop + 1e-12)           
            tau = float(getattr(self, "consistency_tau", 0.3))   
            logits_rank = (S_scaled / tau).reshape(-1, r).float()
            targets_rank = torch.zeros(logits_rank.size(0), dtype=torch.long, device=logits_rank.device)
            

        loss_cons = torch.nn.functional.cross_entropy(logits_rank,targets_rank)
        if self.accelerator.is_main_process:
            print("=============loss_cons=============")
            print(loss_cons.detach().float().cpu())

        w_mlc   = float(getattr(self, "lambda_multilinconsistency", 1.0))


        aux_loss = w_mlc * loss_cons

        # metrics for monitoring
        metrics = {
            "mlc_loss": aux_loss.detach()
        }
        return aux_loss, metrics

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        

        sft_loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()


        loss=sft_loss

        return loss, generated_tokens, labels

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")
