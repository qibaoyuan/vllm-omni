# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""Inference-only Qwen2-Audio model compatible with HuggingFace weights."""

import time
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Optional, Union

import torch
import torch.nn as nn
from transformers import BatchFeature, DynamicCache, Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Model as TransformerQwen2Model,
)
from transformers.models.qwen2_audio import Qwen2AudioConfig, Qwen2AudioProcessor
from transformers.models.whisper import WhisperFeatureExtractor
from vllm.config import VllmConfig
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.model_loader.weight_utils import default_weight_loader, maybe_remap_kv_scale_name
from vllm.model_executor.models.interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
from vllm.model_executor.models.utils import (
    init_vllm_registered_model,
    is_pp_missing_parameter,
    maybe_prefix,
    merge_multimodal_embeddings,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    AudioItem,
    ModalityData,
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    AudioProcessorItems,
    DictEmbeddingItems,
    ModalityDataItems,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from vllm_omni.model_executor.models.mimo_audio.config_mimo_audio import MiMoAudioConfig
from vllm_omni.model_executor.models.mimo_audio.myutils import print_shape


@dataclass
class MiMoSampler:
    do_sample: bool | None = None
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None

    def process(self, scores: torch.Tensor):
        if self.temperature is not None:
            scores = scores / self.temperature

        if self.top_k is not None and self.top_k > 0:
            top_k = min(self.top_k, scores.shape[-1])
            indices_to_remove = scores < torch.topk(scores, top_k)[0][:, -1]
            scores = scores.masked_fill(indices_to_remove, float("-inf"))

        if self.top_p is not None and 0.0 < self.top_p <= 1.0:
            top_p = self.top_p if 0.0 < self.top_p <= 1.0 else 1.0
            sorted_logits, sorted_indices = torch.sort(scores)
            cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
            sorted_indices_to_remove[:, -1] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            scores = scores.masked_fill(indices_to_remove, float("-inf"))

        return scores

    def sample(self, scores: torch.Tensor, removed_tokens: list[int] | None = None):
        scores = self.process(scores)
        for t in removed_tokens or []:
            scores[:, t] = float("-inf")

        if self.do_sample:
            probs = scores.softmax(dim=-1)
            return torch.multinomial(probs, num_samples=1).squeeze(-1)

        return torch.argmax(scores, dim=-1)


class MiMoAudioQwen2Model(TransformerQwen2Model):
    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        # if hasattr(self, "embed_tokens"):
        #     del self.embed_tokens


# # === Audio Inputs === #
class Qwen2AudioFeatureInputs(TensorSchema):
    """
    Dimensions:
        - na: Number of audios
        - nmb: Number of mel bins
    """

    type: Literal["audio_features"]
    input_features: Annotated[
        Union[torch.Tensor, list[torch.Tensor]],
        TensorShape("na", "nmb", 3000),
    ]

    feature_attention_mask: Annotated[
        torch.Tensor,
        TensorShape("na", 3000),
    ]


class Qwen2AudioEmbeddingInputs(TensorSchema):
    """
    Dimensions:
        - bn: Batch size
        - naf: Number of audio features
        - hs: Hidden size (must match the hidden size of language model
          backbone)
    """

    type: Literal["audio_embeds"] = "audio_embeds"

    audio_embeds: list[torch.Tensor]

    # audio_embeds: Annotated[
    #     list[torch.Tensor],
    #     TensorShape("bn", "naf", "hs"),
    # ]


Qwen2AudioInputs = Union[Qwen2AudioFeatureInputs, Qwen2AudioEmbeddingInputs]


# === Audio Encoder === #


class Qwen2AudioMultiModalProjector(nn.Module):
    def __init__(self, audio_hidden_size: int, text_hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(audio_hidden_size, text_hidden_size, bias=True)

    def forward(self, audio_features):
        hidden_states = self.linear(audio_features)
        return hidden_states


# From Qwen2AudioEncoder._get_feat_extract_output_lengths
def _get_feat_extract_output_lengths(input_lengths: torch.Tensor):
    feat_lengths = (input_lengths - 1) // 2 + 1
    output_lengths = (feat_lengths - 2) // 2 + 1
    return feat_lengths, output_lengths


class Qwen2AudioProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(Qwen2AudioConfig)

    def get_hf_processor(self, **kwargs: object) -> Qwen2AudioProcessor:
        return self.ctx.get_hf_processor(Qwen2AudioProcessor, **kwargs)

    def get_feature_extractor(self, **kwargs: object) -> WhisperFeatureExtractor:
        hf_processor = self.get_hf_processor(**kwargs)
        feature_extractor = hf_processor.feature_extractor  # type: ignore
        assert isinstance(feature_extractor, WhisperFeatureExtractor)
        return feature_extractor

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"audio": None}


class Qwen2AudioDummyInputsBuilder(BaseDummyInputsBuilder[Qwen2AudioProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)

        hf_processor = self.info.get_hf_processor()
        audio_token = hf_processor.audio_token

        return audio_token * num_audios

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        feature_extractor = self.info.get_feature_extractor()

        sampling_rate = feature_extractor.sampling_rate
        audio_len = feature_extractor.chunk_length * sampling_rate
        num_audios = mm_counts.get("audio", 0)

        return {"audio": self._get_dummy_audios(length=audio_len, num_audios=num_audios)}


def _qwen2audio_field_config(hf_inputs: Mapping[str, torch.Tensor]):
    return dict(
        audio_embeds=MultiModalFieldConfig.batched("audio"),
        input_features=MultiModalFieldConfig.batched("audio"),
        feature_attention_mask=MultiModalFieldConfig.batched("audio"),
    )


class Qwen2AudioMultiModalDataParser(MultiModalDataParser):
    def _parse_audio_data(
        self,
        data: Union[dict[str, torch.Tensor], ModalityData[AudioItem]],
    ) -> Optional[ModalityDataItems[Any, Any]]:
        if isinstance(data, dict):
            return DictEmbeddingItems(
                data,
                modality="audio",
                required_fields={"audio_embeds"},
                fields_factory=_qwen2audio_field_config,
            )

        return super()._parse_audio_data(data)


class Qwen2AudioMultiModalProcessor(BaseMultiModalProcessor[Qwen2AudioProcessingInfo]):
    def _get_data_parser(self) -> MultiModalDataParser:
        feature_extractor = self.info.get_feature_extractor()
        return Qwen2AudioMultiModalDataParser(target_sr=feature_extractor.sampling_rate)

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, Any],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        # NOTE - we rename audios -> audio in mm data because transformers has
        # deprecated audios for the qwen2audio processor and will remove
        # support for it in transformers 4.54.
        audios = mm_data.pop("audios", [])
        if audios:
            mm_data["audio"] = audios

        # Text-only input not supported in composite processor
        if not mm_data.get("audio", []):
            prompt_ids = self.info.get_tokenizer().encode(prompt)
            prompt_ids = self._apply_hf_processor_tokens_only(prompt_ids)
            return BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")

        feature_extractor = self.info.get_feature_extractor(**mm_kwargs)
        mm_kwargs = dict(
            **mm_kwargs,
            sampling_rate=feature_extractor.sampling_rate,
        )

        return super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return _qwen2audio_field_config(hf_inputs)

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()

        # Use getattr with default to be compatible with transformers<4.48
        audio_token = getattr(processor, "audio_token", "<|AUDIO|>")
        audio_bos_token = getattr(processor, "audio_bos_token", "<|audio_bos|>")
        audio_eos_token = getattr(processor, "audio_eos_token", "<|audio_eos|>")

        audio_token_id = vocab[audio_token]
        audio_bos_id = vocab[audio_bos_token]
        audio_eos_id = vocab[audio_eos_token]

        out_mm_data = out_mm_kwargs.get_data()
        feature_attention_mask = out_mm_data.get("feature_attention_mask")
        if feature_attention_mask is None:
            audio_output_lengths = []
        else:
            assert isinstance(feature_attention_mask, torch.Tensor)
            _, audio_output_lens = _get_feat_extract_output_lengths(feature_attention_mask.sum(-1))

            audio_output_lengths = audio_output_lens.tolist()

        def get_replacement_qwen2_audio(item_idx: int):
            if audio_output_lengths:
                num_features = audio_output_lengths[item_idx]
            else:
                audio_embeds = out_mm_data["audio_embeds"][item_idx]
                assert len(audio_embeds.shape) == 2, "audio_embeds must be a 2D tensor"
                num_features = audio_embeds.shape[0]

            if num_features == 0:
                audios = mm_items.get_items("audio", AudioProcessorItems)
                audio_len = audios.get_audio_length(item_idx)

                raise ValueError(f"The audio (len={audio_len}) is too short to be represented inside the model")

            audio_tokens = [audio_token_id] * num_features

            return PromptUpdateDetails.select_token_id(
                [audio_bos_id] + audio_tokens + [audio_eos_id],
                embed_token_id=audio_token_id,
            )

        return [
            PromptReplacement(
                modality="audio",
                target=audio_token,
                replacement=get_replacement_qwen2_audio,
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(
    Qwen2AudioMultiModalProcessor, info=Qwen2AudioProcessingInfo, dummy_inputs=Qwen2AudioDummyInputsBuilder
)
class MiMoAudioLLMForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP):
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        if modality.startswith("audio"):
            return f"Audio {i}: <|audio_bos|><|AUDIO|><|audio_eos|>"

        raise ValueError("Only audio modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        # 特殊 token IDs 定义（与 added_tokens.json 对应）
        self.empty_token_id = 151667  # <|empty|>
        self.sostm_token_id = 151670  # <|sostm|>
        self.eostm_token_id = 151671  # <|eostm|>
        self.sosp_token_id = 151665  # <|sosp|>
        self.eosp_token_id = 151666  # <|eosp|>

        self.stop_token_ids = {151643, 151645, self.eostm_token_id}
        config = vllm_config.model_config.hf_config
        config = MiMoAudioConfig(**vars(config)) if isinstance(config, Qwen2Config) else config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.logits_processor = LogitsProcessor(config.vocab_size)

        self.config = config
        self.lora_config = lora_config

        self.quant_config = quant_config
        print("vllm_config", vllm_config)
        # self.model = Qwen2Model(vllm_config=vllm_config,
        #                         prefix=maybe_prefix(prefix, "model"))

        vllm_config.model_config.hf_config = self.config
        setattr(
            vllm_config.model_config.hf_config,
            "rope_scaling",
            {"mrope_section": [16, 24, 24], "rope_type": "default", "type": "default"},
        )
        self.model = init_vllm_registered_model(
            vllm_config=vllm_config,
            # hf_config=config,
            prefix=maybe_prefix(prefix, "model"),
            # hf_config=thinker_config.text_config,
            architectures=["Qwen2ForCausalLM"],
        )

        self.global_sampler = MiMoSampler(do_sample=False, temperature=0.6, top_p=0.95)
        self.removed_tokens = None

        print("self.model", self.model)

        print("vllm_config", vllm_config)

        self.speech_vocab_sizes = config.parsed_speech_vocab_sizes()
        self.speech_empty_ids = config.parsed_speech_empty_ids()
        self.delay_pattern = config.parsed_delay_pattern()
        self.group_size = config.group_size
        self.audio_channels = config.audio_channels

        self.local_config = config.local_config()
        self.input_local_config = config.input_local_config()

        self.speech_group_downcast = ColumnParallelLinear(
            self.input_local_config.hidden_size * config.group_size,
            config.hidden_size,
            bias=False,
            return_bias=False,
        )
        self.hidden_states_downcast = ColumnParallelLinear(
            config.hidden_size,
            self.local_config.hidden_size,
            bias=False,
        )

        # self.speech_group_downcast = nn.Linear(
        #     self.input_local_config.hidden_size * config.group_size,
        #     config.hidden_size,
        #     bias=False,
        # )
        # self.hidden_states_downcast = nn.Linear(
        #     config.hidden_size,
        #     self.local_config.hidden_size,
        #     bias=False,
        # )

        self.input_local_config = config.input_local_config()
        self.input_local_transformer = MiMoAudioQwen2Model(self.input_local_config)
        self.input_local_transformer.embed_tokens = None
        print("self.input_local_config", self.input_local_config)
        print("input_local_transformer", self.input_local_transformer)

        ###other parts

        self.local_transformer = MiMoAudioQwen2Model(self.local_config)
        self.local_transformer.embed_tokens = None
        self.local_transformer_lm_heads = nn.ModuleList(
            [
                nn.Linear(
                    self.local_config.hidden_size,
                    self.speech_vocab_sizes[i],
                    bias=False,
                )
                for i in range(self.audio_channels)
            ]
        )

        self.speech_embeddings = nn.ModuleList(
            [
                nn.Embedding(
                    self.speech_vocab_sizes[i],
                    self.input_local_config.hidden_size,
                    padding_idx=self.speech_empty_ids[i],
                )
                for i in range(self.audio_channels)
            ]
        )

        if self.input_local_config.hidden_size != self.local_config.hidden_size:
            self.speech_embeddings_to_local = nn.Linear(
                self.input_local_config.hidden_size,
                self.local_config.hidden_size,
                bias=False,
            )
        else:
            self.speech_embeddings_to_local = None
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors
        # 用于存储每个请求的 new_audio_emb (per-request state)
        # key: request_id (str), value: new_audio_emb (torch.Tensor)
        self._cached_new_audio_emb_by_req: dict[str, torch.Tensor] = {}
        # 用于存储每个请求累积的 audio_codes（离散 token 形式）
        # key: request_id (str), value: list of [audio_channels, T] tensors
        self._cached_audio_codes_by_req: dict[str, list[torch.Tensor]] = {}
        print("init MiMoAudioLLMForConditionalGeneration ended")

    def _validate_and_reshape_mm_tensor(self, mm_input: object, name: str) -> torch.Tensor:
        if not isinstance(mm_input, (torch.Tensor, list)):
            raise ValueError(f"Incorrect type of {name}. Got type: {type(mm_input)}")
        if isinstance(mm_input, torch.Tensor):
            return mm_input.reshape(-1, *mm_input.shape[2:])
        else:
            return mm_input
            # return torch.concat(mm_input)

    def _parse_and_validate_audio_input(self, **kwargs: object) -> Optional[Qwen2AudioInputs]:
        input_features = kwargs.pop("input_features", None)
        audio_embeds = kwargs.pop("audio_embeds", None)
        feature_attention_mask = kwargs.pop("feature_attention_mask", None)

        if input_features is None and audio_embeds is None:
            return None

        if audio_embeds is not None:
            if not isinstance(audio_embeds, (torch.Tensor, list)):
                raise ValueError(f"Incorrect type of audio embeds. Got type: {type(audio_embeds)}")
            audio_embeds = self._validate_and_reshape_mm_tensor(audio_embeds, "audio_embeds")
            return Qwen2AudioEmbeddingInputs(type="audio_embeds", audio_embeds=audio_embeds)

        if input_features is not None:
            input_features = self._validate_and_reshape_mm_tensor(input_features, "input_features")
            return Qwen2AudioFeatureInputs(
                type="audio_features", input_features=input_features, feature_attention_mask=feature_attention_mask
            )

        raise AssertionError("This line should be unreachable.")

    def _process_audio_input(self, audio_input: Qwen2AudioInputs) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:
        if audio_input["type"] == "audio_embeds":
            audio_embeds = audio_input["audio_embeds"]
            return tuple(audio_embeds)

        input_features = audio_input["input_features"]
        feature_attention_mask = audio_input["feature_attention_mask"]

        audio_feat_lengths, audio_output_lengths = self.audio_tower._get_feat_extract_output_lengths(
            feature_attention_mask.sum(-1)
        )

        batch_size, _, max_mel_seq_len = input_features.shape
        max_seq_len = (max_mel_seq_len - 2) // 2 + 1
        # Create a sequence tensor of shape (batch_size, max_seq_len)
        seq_range = (
            torch.arange(0, max_seq_len, dtype=audio_feat_lengths.dtype, device=audio_feat_lengths.device)
            .unsqueeze(0)
            .expand(batch_size, max_seq_len)
        )
        lengths_expand = audio_feat_lengths.unsqueeze(-1).expand(batch_size, max_seq_len)
        # Create mask
        padding_mask = seq_range >= lengths_expand

        audio_attention_mask_ = padding_mask.view(batch_size, 1, 1, max_seq_len).expand(
            batch_size, 1, max_seq_len, max_seq_len
        )
        audio_attention_mask = audio_attention_mask_.to(
            dtype=self.audio_tower.conv1.weight.dtype, device=self.audio_tower.conv1.weight.device
        )
        audio_attention_mask[audio_attention_mask_] = float("-inf")

        audio_outputs = self.audio_tower(input_features, attention_mask=audio_attention_mask)
        selected_audio_feature = audio_outputs.last_hidden_state
        audio_features = self.multi_modal_projector(selected_audio_feature)
        num_audios, max_audio_tokens, embed_dim = audio_features.shape
        audio_output_lengths = audio_output_lengths.unsqueeze(1)
        audio_features_mask = (
            torch.arange(max_audio_tokens).expand(num_audios, max_audio_tokens).to(audio_output_lengths.device)
            < audio_output_lengths
        )
        masked_audio_features = audio_features[audio_features_mask].view(-1, embed_dim)

        # Split to tuple of embeddings for individual audio input.
        return torch.split(masked_audio_features, audio_output_lengths.flatten().tolist())

    def get_language_model(self) -> torch.nn.Module:
        return self.model

    def get_multimodal_embeddings(self, **kwargs: object) -> MultiModalEmbeddings:
        if kwargs.get("mimo_audio_codes_processing") is None:
            kwargs["mimo_audio_codes_processing"] = True if kwargs.get("audio_embeds") is not None else False
        audio_input = self._parse_and_validate_audio_input(**kwargs)

        if audio_input is None:
            return []
        masked_audio_features = self._prepare_input_audio_embeds(audio_input, **kwargs)
        # masked_audio_features = self._process_audio_input(audio_input)
        return masked_audio_features

    def input_convert(self, input_ids):
        input_ids = input_ids
        return input_ids

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        # print("get_input_embeddings,input_ids", input_ids.shape, input_ids)
        input_ids = self.input_convert(input_ids)
        # print("after get_input_embeddings,input_ids", input_ids.shape, input_ids)
        inputs_embeds = self.model.get_input_embeddings(input_ids)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        if len(input_ids[0]) == 1 and input_ids[0] == self.empty_token_id:
            inputs_embeds = torch.zeros_like(inputs_embeds)

        print_shape(inputs_embeds=inputs_embeds)
        if multimodal_embeddings is not None and len(multimodal_embeddings) != 0:
            if len(input_ids[0]) == 1 and input_ids[0] == self.empty_token_id:
                # multimodal_embeddings 是一个 tuple，需要正确处理
                # 对于单个 empty_token，应该使用最后一个 embedding（通常是 prev_new_audio_emb）
                # 或者将所有 embeddings 相加
                if len(multimodal_embeddings) == 1:
                    # 单个 embedding，确保形状匹配 [1, hidden_size]
                    emb = multimodal_embeddings[0]
                    if emb.dim() == 1:
                        emb = emb.unsqueeze(0)  # [hidden_size] -> [1, hidden_size]
                    elif emb.dim() == 2 and emb.shape[0] > 1:
                        # 如果有多行，只取最后一行（最新的 embedding）
                        emb = emb[-1:].unsqueeze(0) if emb.shape[0] > 1 else emb
                    inputs_embeds = inputs_embeds + emb
                else:
                    # 多个 embedding segments，使用最后一个（最新的 prev_new_audio_emb）
                    # 或者将它们相加（取决于实际需求）
                    last_emb = multimodal_embeddings[-1]
                    if last_emb.dim() == 1:
                        last_emb = last_emb.unsqueeze(0)
                    elif last_emb.dim() == 2 and last_emb.shape[0] > 1:
                        last_emb = last_emb[-1:].unsqueeze(0) if last_emb.shape[0] > 1 else last_emb
                    inputs_embeds = inputs_embeds + last_emb
            else:
                input_ids = input_ids.squeeze(0) if input_ids.dim() == 2 else input_ids
                inputs_embeds = merge_multimodal_embeddings(
                    input_ids, inputs_embeds, multimodal_embeddings, self.empty_token_id
                )
        inputs_embeds = inputs_embeds.to(torch.bfloat16)
        print_shape(id="gie_ie", inputs_embeds=inputs_embeds)
        return inputs_embeds

    def local_forward(
        self,
        # input_ids: torch.Tensor,
        # positions: torch.Tensor,
        local_embeds: torch.FloatTensor,  # [  1, hidden_size]
        tokens_dtype: torch.dtype = torch.int64,
        tokens_device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        local_sampler: MiMoSampler | None = None,
    ):
        delay_iters = self.group_size + max(self.delay_pattern)

        local_tokens = torch.zeros(
            (self.group_size, self.audio_channels),
            dtype=tokens_dtype,
            device=tokens_device,
        )
        if local_sampler is None:
            local_sampler = MiMoSampler(do_sample=False, temperature=0.9, top_p=0.95)

        print_shape(delay_iters=delay_iters, audio_channels=self.audio_channels, local_embeds=local_embeds)
        # return torch.randn((self.group_size,self.audio_channels), )  # [B, group_size, audio_channels]
        ##11
        past_key_values = DynamicCache()
        for t in range(delay_iters):
            # (1, 1, 4096)
            print_shape(id=f"lf_{t}", local_embeds=local_embeds)
            ##qwen2
            output = self.local_transformer(
                inputs_embeds=local_embeds,
                past_key_values=past_key_values,
                return_dict=True,
                use_cache=True,
            )
            hidden_state = output.last_hidden_state
            print_shape(id=f"lf_{t}_hs", hidden_state=hidden_state)
            past_key_values = output.past_key_values

            local_embeds = torch.zeros_like(local_embeds)
            for idx in range(self.audio_channels):  # 8
                cur_start = self.delay_pattern[idx]
                cur_end = cur_start + self.group_size
                cur_empty = self.speech_empty_ids[idx]
                if cur_start <= t < cur_end:
                    cur_lm_head = self.local_transformer_lm_heads[idx]

                    print_shape(id="before cur_lm_head", hidden_state=hidden_state)
                    cur_scores: torch.Tensor = cur_lm_head(hidden_state)[-1, :]  # （1，1025）
                    print_shape(id="\tcur_scores", cur_scores=cur_scores)
                    print_shape(id=f"lf_{t}_{idx}_cur_scores", cur_scores=cur_scores)

                    # tgt_cur_scores_file = f"tgt_cur_scores_file_{t}_{time.time()}.pth"
                    # torch.save(cur_scores, tgt_cur_scores_file)
                    # print("save cur_scores file", tgt_cur_scores_file)
                    print_shape(id="local_sampler", local_sampler=local_sampler)
                    # [ vocab_size]
                    cur_token = local_sampler.sample(
                        cur_scores,
                        [cur_empty],
                    )
                    print_shape(id=f"\tcur_token,t:{t},id:{idx}", cur_token=cur_token)

                    local_tokens[t - cur_start, idx] = cur_token
                    cur_input_embed = self.speech_embeddings[idx](cur_token)

                    print_shape(id=f"\tcur_input_embed,t:{t},id:{idx}", cur_input_embed=cur_input_embed)
                    if self.speech_embeddings_to_local is not None:
                        cur_input_embed = self.speech_embeddings_to_local(cur_input_embed)
                        print_shape(id="speech_embeddings_to_local not none", cur_input_embed=cur_input_embed)
                    local_embeds += cur_input_embed
                    print_shape(id=f"\tlocal_embeds,t:{t},id:{idx}", local_embeds=local_embeds)

        return local_tokens, local_embeds  # [B, group_size, audio_channels]

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        ori_input_ids = None
        # 获取 request_ids 和 request_token_spans 用于多请求支持
        request_ids: Optional[list[str]] = kwargs.get("request_ids")
        request_token_spans: Optional[list[tuple[int, int]]] = kwargs.get("request_token_spans")
        print_shape(
            id="mim_forward",
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            request_ids=request_ids,
            request_token_spans=request_token_spans,
        )
        # 从 intermediate_tensors 或 per-request 缓存中读取前一次的 new_audio_emb
        # 在多请求场景下，需要根据 request_id 来获取对应的 new_audio_emb
        prev_new_audio_emb_by_req: dict[str, torch.Tensor] = {}

        if (
            torch.is_tensor(input_ids)
            and input_ids.view(-1).numel() == 1  # flatten 后只有一个 token
            and input_ids.view(-1)[0].item() == self.empty_token_id  # 这个 token == 151667
            and torch.all(inputs_embeds == 0)
        ):
            kwargs["audio_embeds"] = inputs_embeds
            inputs_embeds = None
        else:
            kwargs["audio_embeds"] = torch.zeros((1, 1, input_ids.shape[1], 4096)).to(input_ids.device)

        kwargs["mimo_audio_codes_processing"] = False

        # 1. 从 intermediate_tensors 中读取（如果存在）
        if intermediate_tensors is not None:
            # intermediate_tensors 可能包含所有请求的 new_audio_emb
            # 格式可能是 dict[str, torch.Tensor] 或单个 tensor
            cached_emb = intermediate_tensors.get("new_audio_emb_by_req", None)
            if cached_emb is not None and isinstance(cached_emb, dict):
                prev_new_audio_emb_by_req = cached_emb
                print_shape(
                    id="retrieved_prev_new_audio_emb_from_intermediate_tensors",
                    prev_new_audio_emb_by_req=prev_new_audio_emb_by_req,
                )

        # 2. 从实例变量中读取（如果存在）
        if hasattr(self, "_cached_new_audio_emb_by_req"):
            for req_id, cached_emb in self._cached_new_audio_emb_by_req.items():
                if req_id not in prev_new_audio_emb_by_req:
                    prev_new_audio_emb_by_req[req_id] = cached_emb
            if prev_new_audio_emb_by_req:
                print_shape(
                    id="retrieved_prev_new_audio_emb_from_instance", prev_new_audio_emb_by_req=prev_new_audio_emb_by_req
                )

        # 3. 如果存在前一次的 new_audio_emb，需要将其合并到 inputs_embeds 中
        # 注意：当 intermediate_tensors is not None 时，inputs_embeds 应该已经准备好了
        # 关键修复：prev_new_audio_emb 应该替换 input_ids 中 empty_token 位置的 embedding
        # 而不是简单地拼接，这样才能保持与 mimo_audio_raw 一致的行为
        if (
            prev_new_audio_emb_by_req
            and inputs_embeds is not None
            and input_ids is not None
            and request_ids is not None
        ):
            batch_size = inputs_embeds.shape[0]
            seq_len = inputs_embeds.shape[1]

            # 检查 input_ids 中是否有 empty_token，如果有，应该用 prev_new_audio_emb 替换对应位置的 embedding
            # 对于单请求场景
            if len(request_ids) == 1 and batch_size == 1:
                req_id = request_ids[0]
                if req_id in prev_new_audio_emb_by_req:
                    prev_emb = prev_new_audio_emb_by_req[req_id]
                    # prev_emb 的形状应该是 [1, hidden_size] 或 [hidden_size]
                    if prev_emb.dim() == 1:
                        prev_emb = prev_emb.unsqueeze(0)  # [1, hidden_size]

                    # 检查 input_ids 的最后一个位置是否是 empty_token
                    # 如果是，用 prev_emb 替换 inputs_embeds 的最后一个位置
                    if input_ids.shape[1] > 0 and input_ids[0, -1].item() == self.empty_token_id:
                        # 替换最后一个位置的 embedding
                        inputs_embeds[:, -1:, :] = prev_emb.unsqueeze(0)
                        print_shape(
                            id="replaced_last_empty_token_embedding",
                            inputs_embeds=inputs_embeds,
                            prev_emb=prev_emb,
                        )
                    else:
                        # 如果没有 empty_token，说明这是第一次生成，不需要替换
                        print_shape(
                            id="no_empty_token_to_replace",
                            input_ids=input_ids,
                        )
            else:
                # 多请求场景：需要根据 request_token_spans 来处理
                # 这里简化处理，只处理第一个请求
                if request_ids and len(request_ids) > 0:
                    req_id = request_ids[0]
                    if req_id in prev_new_audio_emb_by_req:
                        prev_emb = prev_new_audio_emb_by_req[req_id]
                        if prev_emb.dim() == 1:
                            prev_emb = prev_emb.unsqueeze(0)
                        # 对于多请求，暂时只处理第一个请求的最后一个位置
                        if input_ids.shape[1] > 0 and input_ids[0, -1].item() == self.empty_token_id:
                            inputs_embeds[:, -1:, :] = prev_emb.unsqueeze(0)
                            print_shape(
                                id="replaced_last_empty_token_embedding_multi_req",
                                inputs_embeds=inputs_embeds,
                            )

        # NOTE: In v1, inputs_embeds is always generated at model runner, this
        # condition is for v0 compatibility.
        elif inputs_embeds is None:
            ##todo: mock data
            # forward_input_ids torch.Size([1, 1]) [[25612]]
            print("forward_input_ids", input_ids.shape, input_ids.tolist())

            print_shape(
                multimodal_embeddings=kwargs["audio_embeds"],
            )  # (1,20,4096)
            multimodal_embeddings = self.get_multimodal_embeddings(**kwargs)
            # ["multimodal_embeddings=(tensor([[0., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0'),)"]
            print_shape(
                multimodal_embeddings=multimodal_embeddings,
            )

            # 如果存在前一次的 new_audio_emb，需要正确处理
            # 关键修复：prev_new_audio_emb 应该作为下一个 empty_token 位置的 embedding
            # 检查 input_ids 中是否有 empty_token
            if prev_new_audio_emb_by_req and input_ids is not None:
                # 简化处理：如果只有一个请求，使用该请求的 prev_new_audio_emb
                if request_ids and len(request_ids) == 1:
                    req_id = request_ids[0]
                    prev_new_audio_emb = prev_new_audio_emb_by_req.get(req_id)
                elif prev_new_audio_emb_by_req:
                    # 多个请求时，使用第一个找到的
                    prev_new_audio_emb = next(iter(prev_new_audio_emb_by_req.values()))
                else:
                    prev_new_audio_emb = None

                if prev_new_audio_emb is not None:
                    print_shape(id="merging_prev_new_audio_emb", prev_new_audio_emb=prev_new_audio_emb)
                    # 确保 prev_new_audio_emb 的形状正确：[seq_len, hidden_size]
                    if prev_new_audio_emb.dim() == 1:
                        prev_new_audio_emb = prev_new_audio_emb.unsqueeze(0)  # [1, hidden_size]

                    # 检查 input_ids 中是否有 empty_token
                    # 如果有，应该将 prev_new_audio_emb 作为该位置的 embedding
                    # 这里通过将 prev_new_audio_emb 添加到 multimodal_embeddings 来实现
                    # 但需要确保在 get_input_embeddings 中正确处理
                    if multimodal_embeddings and len(multimodal_embeddings) > 0:
                        # 将 prev_new_audio_emb 添加到 multimodal_embeddings tuple 中
                        # 注意：这里不应该直接相加，而是应该作为新的 embedding segment
                        # 但为了兼容现有逻辑，我们将其作为新的 audio embedding
                        multimodal_embeddings = multimodal_embeddings + (prev_new_audio_emb,)
                    else:
                        multimodal_embeddings = (prev_new_audio_emb,)
                    print_shape(id="after_merging_prev_new_audio_emb", multimodal_embeddings=multimodal_embeddings)

            inputs_embeds = self.get_input_embeddings(input_ids, multimodal_embeddings)

            print_shape(multimodal_embeddings=multimodal_embeddings, inputs_embeds=inputs_embeds)
            input_ids = None

        # (1, 1, 4096),
        print_shape(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        # inputs_embeds = inputs_embeds.to(torch.float32)
        hidden_states = self.model.model(input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds)

        logits = self.model.compute_logits(hidden_states)
        # logits=logits[-1,:]
        print_shape(id="compute_logits", logits=logits)
        next_ids = self.global_sampler.sample(logits[-1:, :], removed_tokens=self.removed_tokens)
        print_shape(id="compute_logits", next_ids=next_ids)
        print("forward res , hidden_states:", hidden_states)
        # return hidden_states

        ie, hs = inputs_embeds, hidden_states.unsqueeze(0)
        print_shape(
            id="llm_forward",
            ie=ie,
            hs=hs,
        )

        new_audio_emb = None
        next_speech_tokens = None
        if next_ids is not None and len(next_ids) == 1 and int(next_ids[0]) == self.empty_token_id:
            # if id is empty_token_id, then will be use hs to do local forward
            print_shape(id=f"input_ids0={self.empty_token_id}", input_ids=next_ids, hs=hs)
            hs_downsampled = self.hidden_states_downcast(hs)
            print_shape(id="hs_downsampled", hs_downsampled=hs_downsampled)

            next_speech_tokens, audio_emb = self.local_forward(
                local_embeds=hs_downsampled[0],  ###传输的内容（做下一个stage？）
            )

            print_shape(id="after,local_forward", next_speech_tokens=next_speech_tokens, audio_emb=audio_emb)

            # 4,8,4096
            print_shape(id="after,get_input_embeddings", audio_emb=audio_emb)
            print_shape(id="self.input_local_transformer_shape", input_local_transformer=self.input_local_transformer)

            new_audio_emb = torch.zeros(
                (
                    1,
                    1,
                    self.group_size,
                    self.input_local_config.hidden_size,
                ),
                device=torch.device("cuda"),
                dtype=torch.bfloat16,
            )

            next_speech_tokens = next_speech_tokens.to(torch.int32).T.unsqueeze(0).unsqueeze(0)
            print_shape(next_speech_tokens=next_speech_tokens)
            for idx in range(self.audio_channels):
                cur_empty = self.speech_empty_ids[idx]
                cur_embed = self.speech_embeddings[idx]
                # ['id=forlloop_0', 'next_speech_tokens=(4, 8),values=[[526.0, 190.0, 55.0, 86.0, 10.0, 0.0, 75.0, 71.0]...']

                cur_speech_ids = next_speech_tokens[:, :, idx, :]
                cur_speech_embeds: torch.Tensor = cur_embed(cur_speech_ids)
                # [B, T_groups, group_size, hidden_size]

                cur_mask = cur_speech_ids == cur_empty
                cur_speech_embeds.masked_fill_(cur_mask.unsqueeze(-1), 0.0)

                new_audio_emb += cur_speech_embeds
                print_shape(id=f"forlloop_{idx}", new_audio_emb=new_audio_emb, cur_speech_embeds=cur_speech_embeds)
            print_shape(id="before,input_local_transformer", new_audio_emb=new_audio_emb)

            print("model info")
            print(self.input_local_transformer.dtype)
            print(next(self.input_local_transformer.parameters()).dtype)
            print(self.input_local_transformer.config.rope_theta)
            print(self.input_local_transformer.config.rope_scaling)
            setattr(self.input_local_transformer.config, "rope_scaling", None)
            new_audio_emb_1 = self.input_local_transformer(
                inputs_embeds=new_audio_emb.squeeze(0),
                return_dict=True,
                is_causal=False,
            )

            print_shape(id="after,1input_local_transformer", new_audio_emb_1=new_audio_emb_1.last_hidden_state)
            new_audio_emb_2 = new_audio_emb_1.last_hidden_state.view(1, 1, -1)
            print_shape(
                id="after,input_local_transformer,before,speech_group_downcast",
                new_audio_emb_2=new_audio_emb_2,
                old_audio_emb=audio_emb,
            )
            new_audio_emb_3 = self.speech_group_downcast(new_audio_emb_2)[0]
            print_shape(id="after,input_local_transformer", new_audio_emb_3=new_audio_emb_3)
            new_audio_emb = new_audio_emb_3.clone()

        # 如果生成了 new_audio_emb，需要将其存储以便下一轮使用
        # 在多请求场景下，需要根据 request_id 来存储每个请求的 new_audio_emb
        if new_audio_emb is not None:
            # 确定当前请求的 request_id
            # 如果只有一个请求，使用该请求的 request_id
            # 如果有多个请求，需要根据实际情况确定（这里简化处理）
            if request_ids and len(request_ids) == 1:
                req_id = request_ids[0]
                # 存储到 per-request 缓存中
                self._cached_new_audio_emb_by_req[req_id] = new_audio_emb
                print_shape(id="stored_new_audio_emb_to_instance", req_id=req_id, new_audio_emb=new_audio_emb)

                # 同时，如果 intermediate_tensors 存在，也尝试存储到其中
                if intermediate_tensors is not None:
                    # 创建或更新 new_audio_emb_by_req 字典
                    if "new_audio_emb_by_req" not in intermediate_tensors:
                        intermediate_tensors["new_audio_emb_by_req"] = {}
                    intermediate_tensors["new_audio_emb_by_req"][req_id] = new_audio_emb
                    print_shape(
                        id="stored_new_audio_emb_to_intermediate_tensors", req_id=req_id, new_audio_emb=new_audio_emb
                    )
            elif request_ids and len(request_ids) > 1:
                # 多请求场景：需要确定 new_audio_emb 属于哪个请求
                # 这里简化处理：假设 new_audio_emb 属于第一个请求
                # TODO: 根据实际需求确定 new_audio_emb 与 request_id 的对应关系
                req_id = request_ids[0]
                self._cached_new_audio_emb_by_req[req_id] = new_audio_emb
                print_shape(
                    id="stored_new_audio_emb_to_instance_multi_req",
                    req_id=req_id,
                    new_audio_emb=new_audio_emb,
                    total_reqs=len(request_ids),
                )

                if intermediate_tensors is not None:
                    if "new_audio_emb_by_req" not in intermediate_tensors:
                        intermediate_tensors["new_audio_emb_by_req"] = {}
                    intermediate_tensors["new_audio_emb_by_req"][req_id] = new_audio_emb
                    print_shape(
                        id="stored_new_audio_emb_to_intermediate_tensors_multi_req",
                        req_id=req_id,
                        new_audio_emb=new_audio_emb,
                    )
            else:
                # 没有 request_ids 的情况（向后兼容）
                # 使用默认 key 或第一个可用的 key
                if not self._cached_new_audio_emb_by_req:
                    default_key = "default"
                else:
                    default_key = next(iter(self._cached_new_audio_emb_by_req.keys()))
                self._cached_new_audio_emb_by_req[default_key] = new_audio_emb
                print_shape(
                    id="stored_new_audio_emb_to_instance_no_req_id",
                    default_key=default_key,
                    new_audio_emb=new_audio_emb,
                )

                if intermediate_tensors is not None:
                    if "new_audio_emb_by_req" not in intermediate_tensors:
                        intermediate_tensors["new_audio_emb_by_req"] = {}
                    intermediate_tensors["new_audio_emb_by_req"][default_key] = new_audio_emb
                    print_shape(
                        id="stored_new_audio_emb_to_intermediate_tensors_no_req_id",
                        default_key=default_key,
                        new_audio_emb=new_audio_emb,
                    )

        return next_speech_tokens, hs

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        lg = self.model.compute_logits(hidden_states)

        print_shape(
            id="llm_logit",
            lg=lg,
        )
        return lg

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))

        # print("params_dict.keys", params_dict.keys())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if name.startswith("model.") or name.startswith("lm_head."):
                name = "model." + name
            if self.quant_config is not None and (scale_name := self.quant_config.get_cache_scale(name)):
                # Loading kv cache quantization scales
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                loaded_weight = loaded_weight if loaded_weight.dim() == 0 else loaded_weight[0]
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if name.startswith("input_local_transformer.") or name.startswith("local_transformer."):
                    continue
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                print("loaded_weight,name,fused", name)
                if weight_loader == default_weight_loader:
                    weight_loader(param, loaded_weight)
                else:
                    weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    print("biasname is None", name, " not in model define")
                    continue
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    print("name is None", name, " not in model define")
                    continue
                if is_pp_missing_parameter(name, self):
                    print("is_pp_missing_parameterweight_name_key", name, " not in model define")
                    continue
                if name not in params_dict:
                    print("weight_name_key", name, " not in model define")
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                print("loaded_weight,name", name)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params

    def apply_input_local_transformer(self, speech_embeddings: torch.Tensor) -> torch.Tensor:
        B, T_groups, group_size, hidden_size = speech_embeddings.shape

        # Process each group independently: [B*T_groups, group_size, hidden_size]
        input_embeddings = speech_embeddings.reshape(B * T_groups, group_size, hidden_size)

        # Apply input local transformer
        output = self.input_local_transformer(
            inputs_embeds=input_embeddings.to(speech_embeddings.device).to(torch.bfloat16),
            return_dict=True,
            is_causal=not self.config.input_full_attention,
        )
        encoded_embeddings = output.last_hidden_state

        # Reshape back to [B, T_groups, group_size, hidden_size]
        return encoded_embeddings.reshape(B, T_groups, group_size, hidden_size)

    def _prepare_input_audio_embeds(
        self,
        audio_input: Qwen2AudioInputs,  # [B, audio_channels + 1, new_T]
        **kwargs: Any,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:
        prompt_ids_length = kwargs.get("prompt_ids_length", None)
        _is_first_audio_codes = False if prompt_ids_length is None else True
        if audio_input["type"] == "audio_embeds":
            audio_embeds = audio_input["audio_embeds"]
        if (
            not kwargs.get("mimo_audio_codes_processing")
            or (isinstance(audio_embeds, torch.Tensor) and audio_embeds.shape[0] > 1)
            or not _is_first_audio_codes
        ):
            return tuple(audio_embeds)

        prompt_ids_length = prompt_ids_length[0].item()
        T_groups = prompt_ids_length
        mm_offset = kwargs.get("mm_offset").squeeze()
        audio_lengths = kwargs.get("audio_lengths").squeeze()
        group_size = self.group_size
        audio_codes_list = audio_embeds

        dtype = audio_codes_list[0].dtype
        device = audio_codes_list[0].device
        B = audio_codes_list[0].shape[0]

        speech_input_ids = torch.zeros(
            (B, self.audio_channels, prompt_ids_length * group_size), dtype=dtype, device=device
        )
        for i, idx in enumerate(self.speech_empty_ids):
            speech_input_ids[:, i, :] = idx

        speech_input_ids = self._overlay_audio_codes_by_offset(
            speech_input_ids, audio_codes_list, mm_offset, prompt_ids_length, self.audio_channels
        )

        speech_input_ids = speech_input_ids[:, :, : T_groups * group_size].view(
            B, self.audio_channels, T_groups, group_size
        )

        # Transpose to [B, T_groups, audio_channels, group_size]
        speech_input_ids = speech_input_ids.transpose(1, 2)

        # Determine which positions are speech (text token == empty_idx)
        audio_lengths = audio_lengths // group_size  # 4
        is_speech = self._build_is_speech(mm_offset, audio_lengths, T_groups, B, device)  # [B, T_groups]

        # Initialize speech embeddings: [B, T_groups, group_size, hidden_size]
        speech_embeds = torch.zeros(
            (B, T_groups, group_size, self.input_local_config.hidden_size),
            device=device,
            dtype=torch.bfloat16,
        )

        # Process each audio channel
        for idx in range(self.audio_channels):
            cur_empty = self.speech_empty_ids[idx]
            cur_embed = self.speech_embeddings[idx]

            # Get speech tokens for this channel: [B, T_groups, group_size]
            cur_speech_ids = speech_input_ids[:, :, idx, :]

            # Convert to embeddings: [B, T_groups, group_size, hidden_size]
            cur_speech_embeds: torch.Tensor = cur_embed(cur_speech_ids)

            # Mask out empty tokens
            cur_mask = cur_speech_ids == cur_empty
            cur_speech_embeds.masked_fill_(cur_mask.unsqueeze(-1), 0.0)

            # Accumulate embeddings across channels
            speech_embeds += cur_speech_embeds

        # Apply mask to zero out non-speech positions
        speech_embeds = speech_embeds * is_speech.unsqueeze(-1).unsqueeze(-1)

        # Apply input local transformer if configured
        speech_embeds = self.apply_input_local_transformer(speech_embeds)

        # Re-apply mask after transformer
        speech_embeds = speech_embeds * is_speech.unsqueeze(-1).unsqueeze(-1)

        # Downcast grouped speech embeddings: [B, T_groups, hidden_size]
        speech_grouped_embeds: torch.Tensor = self.speech_group_downcast(
            speech_embeds.view(B, speech_embeds.shape[1], -1)
        )

        mm_offset = mm_offset.tolist()
        audio_lengths = audio_lengths.tolist()

        speech_embeds_split = []

        for off, length in zip(mm_offset, audio_lengths):
            if length <= 0:
                continue

            start = int(off)
            end = int(off + length)

            start = max(start, 0)
            end = min(end, speech_grouped_embeds.size(1))

            if start >= end:
                continue

            # [B, length, hidden_size]
            seg_embeds = speech_grouped_embeds[:, start:end, :]
            speech_embeds_split.append(seg_embeds)

        audio_embeds_list = [
            speech_embeds_grouped.reshape(B * speech_embeds_grouped.shape[1], -1)
            for speech_embeds_grouped in speech_embeds_split
        ]

        return tuple(audio_embeds_list)

    def _overlay_audio_codes_by_offset(
        self,
        speech_input_ids: torch.Tensor,
        audio_codes_list: list,
        mm_offset: torch.Tensor,
        prompt_ids_length: int,
        audio_channels: int,
    ):
        total_length = prompt_ids_length * self.group_size

        if mm_offset.numel() != len(audio_codes_list):
            raise ValueError(
                f"mm_offset length ({mm_offset.numel()}) != audio_codes_list length ({len(audio_codes_list)})"
            )

        mm_offset = mm_offset.to(speech_input_ids.device)

        for seg_i, audio_codes in enumerate(audio_codes_list):
            T = audio_codes.shape[2]

            off = int(mm_offset[seg_i].item()) * self.group_size
            if off >= total_length:
                continue

            end = min(off + T, total_length)
            valid_T = end - off
            if valid_T <= 0:
                continue

            speech_input_ids[:, :audio_channels, off:end] = audio_codes[:, :audio_channels, :valid_T]

        return speech_input_ids

    def _build_is_speech(
        self,
        mm_offset: torch.Tensor,
        audio_lengths: torch.Tensor,
        T_groups: int,
        B: int,
        device=None,
    ):
        t_idx = torch.arange(T_groups, device=device)  # (T_groups,)

        seg_mask = (t_idx[None, :] >= mm_offset[:, None]) & (t_idx[None, :] < (mm_offset + audio_lengths)[:, None])

        return seg_mask.any(dim=0).unsqueeze(0).expand(B, -1)
