# Copyright 2025 the LlamaFactory team.
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

import os
from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Union

from ..extras import logging
from .data_utils import Role


if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from transformers import Seq2SeqTrainingArguments

    from ..hparams import DataArguments
    from .mm_plugin import AudioInput, ImageInput, VideoInput
    from .parser import DatasetAttr

    MediaType = Union[ImageInput, VideoInput, AudioInput]


logger = logging.get_logger(__name__)


@dataclass
class DatasetConverter:
    dataset_attr: "DatasetAttr"
    data_args: "DataArguments"

    def _find_medias(self, medias: Union["MediaType", list["MediaType"], None]) -> Optional[list["MediaType"]]:
        r"""Optionally concatenate media path to media dir when loading from local disk."""
        if medias is None:
            return None
        elif not isinstance(medias, list):
            medias = [medias]
        elif len(medias) == 0:
            return None
        else:
            medias = medias[:]

        if self.dataset_attr.load_from in ["script", "file"]:
            if isinstance(medias[0], str):
                for i in range(len(medias)):
                    media_path = os.path.join(self.data_args.media_dir, medias[i])
                    if os.path.isfile(media_path):
                        medias[i] = media_path
                    else:
                        logger.warning_rank0_once(
                            f"Media {medias[i]} does not exist in `media_dir`. Use original path."
                        )
            elif isinstance(medias[0], list):  # for processed video frames
                # medias is a list of lists, e.g., [[frame1.jpg, frame2.jpg], [frame3.jpg, frame4.jpg]]
                for i in range(len(medias)):
                    for j in range(len(medias[i])):
                        media_path = os.path.join(self.data_args.media_dir, medias[i][j])
                        if os.path.isfile(media_path):
                            medias[i][j] = media_path
                        else:
                            logger.warning_rank0_once(
                                f"Media {medias[i][j]} does not exist in `media_dir`. Use original path."
                            )

        return medias

    @abstractmethod
    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        r"""Convert a single example in the dataset to the standard format."""
        ...


@dataclass
class AlpacaDatasetConverter(DatasetConverter):
    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        prompt = []
        if self.dataset_attr.history and isinstance(example[self.dataset_attr.history], list):
            for old_prompt, old_response in example[self.dataset_attr.history]:
                prompt.append({"role": Role.USER.value, "content": old_prompt})
                prompt.append({"role": Role.ASSISTANT.value, "content": old_response})

        query = []
        if self.dataset_attr.prompt and example[self.dataset_attr.prompt]:
            query.append(example[self.dataset_attr.prompt])

        if self.dataset_attr.query and example[self.dataset_attr.query]:
            query.append(example[self.dataset_attr.query])

        prompt.append({"role": Role.USER.value, "content": "\n".join(query)})  # "prompt\nquery"

        if self.dataset_attr.kto_tag and isinstance(example[self.dataset_attr.kto_tag], bool):  # kto example
            response = [{"role": Role.ASSISTANT.value, "content": example[self.dataset_attr.response]}]
            if example[self.dataset_attr.kto_tag]:
                response = response + [{"role": Role.ASSISTANT.value, "content": ""}]
            else:
                response = [{"role": Role.ASSISTANT.value, "content": ""}] + response
        elif (
            self.dataset_attr.ranking
            and isinstance(example[self.dataset_attr.chosen], str)
            and isinstance(example[self.dataset_attr.rejected], str)
        ):  # pairwise example
            response = [
                {"role": Role.ASSISTANT.value, "content": example[self.dataset_attr.chosen]},
                {"role": Role.ASSISTANT.value, "content": example[self.dataset_attr.rejected]},
            ]
        elif self.dataset_attr.response and isinstance(example[self.dataset_attr.response], str):  # normal example
            response = [{"role": Role.ASSISTANT.value, "content": example[self.dataset_attr.response]}]
        else:  # unsupervised
            response = []

        output = {
            "_prompt": prompt,
            "_response": response,
            "_system": example[self.dataset_attr.system] if self.dataset_attr.system else "",
            "_tools": example[self.dataset_attr.tools] if self.dataset_attr.tools else "",
            "_images": self._find_medias(example[self.dataset_attr.images]) if self.dataset_attr.images else None,
            "_videos": self._find_medias(example[self.dataset_attr.videos]) if self.dataset_attr.videos else None,
            "_audios": self._find_medias(example[self.dataset_attr.audios]) if self.dataset_attr.audios else None,
        }
        return output


@dataclass
class SharegptDatasetConverter(DatasetConverter):
    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        tag_mapping = {
            self.dataset_attr.user_tag: Role.USER.value,
            self.dataset_attr.assistant_tag: Role.ASSISTANT.value,
            self.dataset_attr.observation_tag: Role.OBSERVATION.value,
            self.dataset_attr.function_tag: Role.FUNCTION.value,
            self.dataset_attr.system_tag: Role.SYSTEM.value,
        }
        odd_tags = (self.dataset_attr.user_tag, self.dataset_attr.observation_tag)
        even_tags = (self.dataset_attr.assistant_tag, self.dataset_attr.function_tag)
        accept_tags = (odd_tags, even_tags)
        messages = example[self.dataset_attr.messages]
        if (
            self.dataset_attr.system_tag
            and len(messages) != 0
            and messages[0][self.dataset_attr.role_tag] == self.dataset_attr.system_tag
        ):
            system = messages[0][self.dataset_attr.content_tag]
            messages = messages[1:]
        else:
            system = example[self.dataset_attr.system] if self.dataset_attr.system else ""

        aligned_messages = []
        broken_data = False
        for turn_idx, message in enumerate(messages):
            if message[self.dataset_attr.role_tag] not in accept_tags[turn_idx % 2]:
                logger.warning_rank0(f"Invalid role tag in {messages}.")
                broken_data = True
                break

            aligned_messages.append(
                {
                    "role": tag_mapping[message[self.dataset_attr.role_tag]],
                    "content": message[self.dataset_attr.content_tag],
                }
            )

        if (not self.dataset_attr.ranking and len(aligned_messages) % 2 != 0) or (
            self.dataset_attr.ranking and len(aligned_messages) % 2 == 0
        ):
            logger.warning_rank0(f"Invalid message count in {messages}.")
            broken_data = True

        if broken_data:
            logger.warning_rank0("Skipping this abnormal example.")
            prompt, response = [], []
        elif self.dataset_attr.kto_tag and isinstance(example[self.dataset_attr.kto_tag], bool):  # kto example
            prompt = aligned_messages[:-1]
            response = aligned_messages[-1:]
            if example[self.dataset_attr.kto_tag]:
                response = response + [{"role": Role.ASSISTANT.value, "content": ""}]
            else:
                response = [{"role": Role.ASSISTANT.value, "content": ""}] + response
        elif (
            self.dataset_attr.ranking
            and isinstance(example[self.dataset_attr.chosen], dict)
            and isinstance(example[self.dataset_attr.rejected], dict)
        ):  # pairwise example
            chosen = example[self.dataset_attr.chosen]
            rejected = example[self.dataset_attr.rejected]
            if (
                chosen[self.dataset_attr.role_tag] not in accept_tags[-1]
                or rejected[self.dataset_attr.role_tag] not in accept_tags[-1]
            ):
                logger.warning_rank0(f"Invalid role tag in {[chosen, rejected]}.")
                broken_data = True

            prompt = aligned_messages
            response = [
                {
                    "role": tag_mapping[chosen[self.dataset_attr.role_tag]],
                    "content": chosen[self.dataset_attr.content_tag],
                },
                {
                    "role": tag_mapping[rejected[self.dataset_attr.role_tag]],
                    "content": rejected[self.dataset_attr.content_tag],
                },
            ]
        else:  # normal example
            prompt = aligned_messages[:-1]
            response = aligned_messages[-1:]

        output = {
            "_prompt": prompt,
            "_response": response,
            "_system": system,
            "_tools": example[self.dataset_attr.tools] if self.dataset_attr.tools else "",
            "_images": self._find_medias(example[self.dataset_attr.images]) if self.dataset_attr.images else None,
            "_videos": self._find_medias(example[self.dataset_attr.videos]) if self.dataset_attr.videos else None,
            "_audios": self._find_medias(example[self.dataset_attr.audios]) if self.dataset_attr.audios else None,
        }
        return output

@dataclass
class Multilingual_SFT_DatasetConverter(DatasetConverter):
    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        prompt = []
        if self.dataset_attr.history and isinstance(example[self.dataset_attr.history], list):
            for old_prompt, old_response in example[self.dataset_attr.history]:
                prompt.append({"role": Role.USER.value, "content": old_prompt})
                prompt.append({"role": Role.ASSISTANT.value, "content": old_response})

        query = []
        if self.dataset_attr.prompt and example[self.dataset_attr.prompt]:
            query.append(example[self.dataset_attr.prompt])

        if self.dataset_attr.query and example[self.dataset_attr.query]:
            query.append(example[self.dataset_attr.query])

        prompt.append({"role": Role.USER.value, "content": "\n".join(query)})  # "prompt\nquery"

        prompt_zh = []
        if self.dataset_attr.prompt_zh and example[self.dataset_attr.prompt_zh]:
            prompt_zh.append({"role": Role.USER.value,"content":example[self.dataset_attr.prompt_zh]})
        prompt_ru = []
        if self.dataset_attr.prompt_ru and example[self.dataset_attr.prompt_ru]:
            prompt_ru.append({"role": Role.USER.value,"content":example[self.dataset_attr.prompt_ru]})
        prompt_ja = []
        if self.dataset_attr.prompt_ja and example[self.dataset_attr.prompt_ja]:
            prompt_ja.append({"role": Role.USER.value,"content":example[self.dataset_attr.prompt_ja]})
        prompt_bn = []
        if self.dataset_attr.prompt_bn and example[self.dataset_attr.prompt_bn]:
            prompt_bn.append({"role": Role.USER.value,"content":example[self.dataset_attr.prompt_bn]})
        prompt_sw = []
        if self.dataset_attr.prompt_sw and example[self.dataset_attr.prompt_sw]:
            prompt_sw.append({"role": Role.USER.value,"content":example[self.dataset_attr.prompt_sw]})
        prompt_ur = []
        if self.dataset_attr.prompt_ur and example[self.dataset_attr.prompt_ur]:
            prompt_ur.append({"role": Role.USER.value,"content":example[self.dataset_attr.prompt_ur]})
        prompt_ps = []
        if self.dataset_attr.prompt_ps and example[self.dataset_attr.prompt_ps]:
            prompt_ps.append({"role": Role.USER.value,"content":example[self.dataset_attr.prompt_ps]})
        prompt_ku = []
        if self.dataset_attr.prompt_ku and example[self.dataset_attr.prompt_ku]:
            prompt_ku.append({"role": Role.USER.value,"content":example[self.dataset_attr.prompt_ku]})
        prompt_ar = []
        if self.dataset_attr.prompt_ar and example[self.dataset_attr.prompt_ar]:
            prompt_ar.append({"role": Role.USER.value,"content":example[self.dataset_attr.prompt_ar]})

        if self.dataset_attr.kto_tag and isinstance(example[self.dataset_attr.kto_tag], bool):  # kto example
            response = [{"role": Role.ASSISTANT.value, "content": example[self.dataset_attr.response]}]
            if example[self.dataset_attr.kto_tag]:
                response = response + [{"role": Role.ASSISTANT.value, "content": ""}]
            else:
                response = [{"role": Role.ASSISTANT.value, "content": ""}] + response
        elif (
            self.dataset_attr.ranking
            and isinstance(example[self.dataset_attr.chosen], str)
            and isinstance(example[self.dataset_attr.rejected], str)
        ):  # pairwise example
            response = [
                {"role": Role.ASSISTANT.value, "content": example[self.dataset_attr.chosen]},
                {"role": Role.ASSISTANT.value, "content": example[self.dataset_attr.rejected]},
            ]
        elif self.dataset_attr.response and isinstance(example[self.dataset_attr.response], str):  # normal example
            response = [{"role": Role.ASSISTANT.value, "content": example[self.dataset_attr.response]}]
        else:  # unsupervised
            response = []

        output = {
            "_prompt": prompt,
            "_prompt_zh": prompt_zh,
            "_prompt_ru": prompt_ru,
            "_prompt_ja": prompt_ja,
            "_prompt_bn": prompt_bn,
            "_prompt_sw": prompt_sw,
            "_prompt_ur": prompt_ur,
            "_prompt_ps": prompt_ps,
            "_prompt_ku": prompt_ku,
            "_prompt_ar": prompt_ar,
            "_response": response,
            "_system": example[self.dataset_attr.system] if self.dataset_attr.system else "",
            "_tools": example[self.dataset_attr.tools] if self.dataset_attr.tools else "",
            "_images": self._find_medias(example[self.dataset_attr.images]) if self.dataset_attr.images else None,
            "_videos": self._find_medias(example[self.dataset_attr.videos]) if self.dataset_attr.videos else None,
            "_audios": self._find_medias(example[self.dataset_attr.audios]) if self.dataset_attr.audios else None,
        }
        return output

@dataclass
class Multilingual_DPO_DatasetConverter(DatasetConverter):
    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        tag_mapping = {
            self.dataset_attr.user_tag: Role.USER.value,
            self.dataset_attr.assistant_tag: Role.ASSISTANT.value,
            self.dataset_attr.observation_tag: Role.OBSERVATION.value,
            self.dataset_attr.function_tag: Role.FUNCTION.value,
            self.dataset_attr.system_tag: Role.SYSTEM.value,
        }
        odd_tags = (self.dataset_attr.user_tag, self.dataset_attr.observation_tag)
        even_tags = (self.dataset_attr.assistant_tag, self.dataset_attr.function_tag)
        accept_tags = (odd_tags, even_tags)

        messages=example[self.dataset_attr.prompt]
        messages_zh=example[self.dataset_attr.prompt_zh]
        messages_ru=example[self.dataset_attr.prompt_ru]
        messages_ja=example[self.dataset_attr.prompt_ja]
        messages_bn=example[self.dataset_attr.prompt_bn]
        messages_sw=example[self.dataset_attr.prompt_sw]
        messages_ur=example[self.dataset_attr.prompt_ur]
        messages_ps=example[self.dataset_attr.prompt_ps]
        messages_ku=example[self.dataset_attr.prompt_ku]
        messages_ar=example[self.dataset_attr.prompt_ar]

        if (
            self.dataset_attr.system_tag
            and len(messages) != 0
            and messages[0][self.dataset_attr.role_tag] == self.dataset_attr.system_tag
        ):
            system = messages[0][self.dataset_attr.content_tag]
            messages = messages[1:]
        else:
            system = example[self.dataset_attr.system] if self.dataset_attr.system else ""

        def get_aligned_messages(msg):
            aligned_messages = []
            broken_data = False
            for turn_idx, message in enumerate(msg):
                if message[self.dataset_attr.role_tag] not in accept_tags[turn_idx % 2]:
                    logger.warning_rank0(f"Invalid role tag in {msg}.")
                    broken_data = True
                    break

                aligned_messages.append(
                    {
                        "role": tag_mapping[message[self.dataset_attr.role_tag]],
                        "content": message[self.dataset_attr.content_tag],
                    }
                )

            if (not self.dataset_attr.ranking and len(aligned_messages) % 2 != 0) or (
                self.dataset_attr.ranking and len(aligned_messages) % 2 == 0
            ):
                logger.warning_rank0(f"Invalid message count in {msg}.")
                broken_data = True
            return aligned_messages,broken_data
        
        aligned_messages,broken_data=get_aligned_messages(messages)

        aligned_messages_zh,broken_data_zh=get_aligned_messages(messages_zh)
        aligned_messages_ru,broken_data_ru=get_aligned_messages(messages_ru)
        aligned_messages_ja,broken_data_ja=get_aligned_messages(messages_ja)
        aligned_messages_bn,broken_data_bn=get_aligned_messages(messages_bn)
        aligned_messages_sw,broken_data_sw=get_aligned_messages(messages_sw)
        aligned_messages_ur,broken_data_ur=get_aligned_messages(messages_ur)
        aligned_messages_ps,broken_data_ps=get_aligned_messages(messages_ps)
        aligned_messages_ku,broken_data_ku=get_aligned_messages(messages_ku)
        aligned_messages_ar,broken_data_ar=get_aligned_messages(messages_ar)
        
        if broken_data:
            logger.warning_rank0("Skipping this abnormal example.")
            prompt, response = [], []
        elif self.dataset_attr.kto_tag and isinstance(example[self.dataset_attr.kto_tag], bool):  # kto example
            prompt = aligned_messages[:-1]
            response = aligned_messages[-1:]
            if example[self.dataset_attr.kto_tag]:
                response = response + [{"role": Role.ASSISTANT.value, "content": ""}]
            else:
                response = [{"role": Role.ASSISTANT.value, "content": ""}] + response
        elif (
            self.dataset_attr.ranking
            and isinstance(example[self.dataset_attr.chosen], dict)
            and isinstance(example[self.dataset_attr.rejected], dict)
        ):  # pairwise example
            chosen = example[self.dataset_attr.chosen]
            rejected = example[self.dataset_attr.rejected]
            if (
                chosen[self.dataset_attr.role_tag] not in accept_tags[-1]
                or rejected[self.dataset_attr.role_tag] not in accept_tags[-1]
            ):
                logger.warning_rank0(f"Invalid role tag in {[chosen, rejected]}.")
                broken_data = True

            prompt = aligned_messages

            prompt_zh = aligned_messages_zh
            prompt_ru = aligned_messages_ru
            prompt_ja = aligned_messages_ja
            prompt_bn = aligned_messages_bn
            prompt_sw = aligned_messages_sw
            prompt_ur = aligned_messages_ur
            prompt_ps = aligned_messages_ps
            prompt_ku = aligned_messages_ku
            prompt_ar = aligned_messages_ar
        

            response = [
                {
                    "role": tag_mapping[chosen[self.dataset_attr.role_tag]],
                    "content": chosen[self.dataset_attr.content_tag],
                },
                {
                    "role": tag_mapping[rejected[self.dataset_attr.role_tag]],
                    "content": rejected[self.dataset_attr.content_tag],
                },
            ]
        else:  # normal example
            prompt = aligned_messages[:-1]
            response = aligned_messages[-1:]

            prompt_zh = aligned_messages_zh[:-1]
            prompt_ru = aligned_messages_ru[:-1]
            prompt_ja = aligned_messages_ja[:-1]
            prompt_bn = aligned_messages_bn[:-1]
            prompt_sw = aligned_messages_sw[:-1]
            prompt_ur = aligned_messages_ur[:-1]
            prompt_ps = aligned_messages_ps[:-1]
            prompt_ku = aligned_messages_ku[:-1]
            prompt_ar = aligned_messages_ar[:-1]

        output = {
            "_prompt": prompt,
            "_prompt_zh": prompt_zh,
            "_prompt_ru": prompt_ru,
            "_prompt_ja": prompt_ja,
            "_prompt_bn": prompt_bn,
            "_prompt_sw": prompt_sw,
            "_prompt_ur": prompt_ur,
            "_prompt_ps": prompt_ps,
            "_prompt_ku": prompt_ku,
            "_prompt_ar": prompt_ar,
            "_response": response,
            "_system": system,
            "_tools": example[self.dataset_attr.tools] if self.dataset_attr.tools else "",
            "_images": self._find_medias(example[self.dataset_attr.images]) if self.dataset_attr.images else None,
            "_videos": self._find_medias(example[self.dataset_attr.videos]) if self.dataset_attr.videos else None,
            "_audios": self._find_medias(example[self.dataset_attr.audios]) if self.dataset_attr.audios else None,
        }
        return output



DATASET_CONVERTERS = {
    "alpaca": AlpacaDatasetConverter,
    "sharegpt": SharegptDatasetConverter,
    "multilingual_dpo": Multilingual_DPO_DatasetConverter,
    "multilingual_sft": Multilingual_SFT_DatasetConverter,
}


def register_dataset_converter(name: str, dataset_converter: type["DatasetConverter"]) -> None:
    r"""Register a new dataset converter."""
    if name in DATASET_CONVERTERS:
        raise ValueError(f"Dataset converter {name} already exists.")

    DATASET_CONVERTERS[name] = dataset_converter


def get_dataset_converter(name: str, dataset_attr: "DatasetAttr", data_args: "DataArguments") -> "DatasetConverter":
    r"""Get a dataset converter."""
    if name not in DATASET_CONVERTERS:
        raise ValueError(f"Dataset converter {name} not found.")

    return DATASET_CONVERTERS[name](dataset_attr, data_args)


def align_dataset(
    dataset: Union["Dataset", "IterableDataset"],
    dataset_attr: "DatasetAttr",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
) -> Union["Dataset", "IterableDataset"]:
    r"""Align the dataset to a specific format.

    Aligned dataset:
    _prompt: [{"role": "user", "content": "..."}] * (2T - 1)
    _response: [{"role": "assistant", "content": "..."}] * N (N > 1 for ranking dataset)
    _system: "..."
    _tools: "..."
    _images: []
    _videos: []
    _audios: []
    """
    column_names = list(next(iter(dataset)).keys())
    kwargs = {}
    if not data_args.streaming:
        kwargs = dict(
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=(not data_args.overwrite_cache) or (training_args.local_process_index != 0),
            desc="Converting format of dataset",
        )

    dataset_converter = get_dataset_converter(dataset_attr.formatting, dataset_attr, data_args)
    return dataset.map(
        dataset_converter,
        batched=False,
        remove_columns=column_names,
        **kwargs,
    )
