# Copyright 2025 Xiaomi Corporation.
"""Inference-only Qwen2-Audio model compatible with HuggingFace weights."""

import time
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Optional, Union, List

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
    _merge_multimodal_embeddings as merge_multimodal_embeddings,
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

        # Special token IDs definition (corresponds to added_tokens.json)
        self.empty_token_id = 151667  # <|empty|>
        self.sostm_token_id = 151670  # <|sostm|>
        self.eostm_token_id = 151671  # <|eostm|>
        self.sosp_token_id = 151665  # <|sosp|>
        self.eosp_token_id = 151666  # <|eosp|>
        self.endoftext_token_id = 151643  # <|endoftext|>
        self.im_end_token_id = 151645  # <|im_end|>

        config = vllm_config.model_config.hf_config
        config = MiMoAudioConfig(**vars(config)) if isinstance(config, Qwen2Config) else config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.logits_processor = LogitsProcessor(config.vocab_size)

        self.config = config
        self.lora_config = lora_config
        self.quant_config = quant_config

        vllm_config.model_config.hf_config = self.config
        self.model = MiMoAudioQwen2Model(config=self.config)

        self.global_sampler = MiMoSampler(do_sample=False, temperature=0.6, top_p=0.95)
        self.local_sampler = MiMoSampler(do_sample=False, temperature=0.9, top_p=0.95)
        self.removed_tokens = None

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

        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            return_bias=False,
        )

        self.input_local_config = config.input_local_config()
        self.input_local_transformer = MiMoAudioQwen2Model(self.input_local_config)
        self.input_local_transformer.embed_tokens = None

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

        self._cached_new_audio_emb_by_req: dict[str, torch.Tensor] = {}
        self._cached_past_key_values_by_req: dict[str, Optional[DynamicCache]] = {}

        # Pre-allocate audio_embeds buffer for CUDA graph capture to avoid dynamic allocation
        # Maximum sequence length set to 8192, can be adjusted according to actual needs
        self._max_audio_embeds_seq_len = 8192
        self.register_buffer(
            "_audio_embeds_buffer",
            torch.zeros((1, 1, self._max_audio_embeds_seq_len, 4096), dtype=torch.bfloat16),
            persistent=False,
        )
        # Pre-allocate attention_mask buffer
        self._max_attn_len = 16384
        self.register_buffer(
            "_attention_mask_buffer",
            torch.ones((1, self._max_attn_len), dtype=torch.bool),
            persistent=False,
        )
        # Pre-allocate new_audio_emb buffer for processing after local_forward
        self.register_buffer(
            "_new_audio_emb_buffer",
            torch.zeros((1, 1, self.group_size, self.input_local_config.hidden_size), dtype=torch.bfloat16),
            persistent=False,
        )

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

    def get_language_model(self) -> torch.nn.Module:
        return self.model

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        if kwargs.get("mimo_audio_codes_processing") is None:
            kwargs["mimo_audio_codes_processing"] = True if kwargs.get("audio_embeds") is not None else False
        audio_input = self._parse_and_validate_audio_input(**kwargs)

        if audio_input is None:
            return []
        masked_audio_features = self._prepare_input_audio_embeds(audio_input, **kwargs)
        return masked_audio_features

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        if len(input_ids[0]) == 1 and input_ids[0] == self.empty_token_id:
            inputs_embeds = torch.zeros_like(inputs_embeds)

        if multimodal_embeddings is not None and len(multimodal_embeddings) != 0:
            if len(input_ids[0]) == 1 and input_ids[0] == self.empty_token_id:
                inputs_embeds = inputs_embeds + multimodal_embeddings
            else:
                input_ids = input_ids.squeeze(0) if input_ids.dim() == 2 else input_ids
                is_multimodal = input_ids == self.empty_token_id
                inputs_embeds = merge_multimodal_embeddings(inputs_embeds, multimodal_embeddings, is_multimodal)

        inputs_embeds = inputs_embeds.to(torch.bfloat16)
        return inputs_embeds

    def local_forward(
        self,
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
            local_sampler = MiMoSampler(do_sample=False, temperature=0.6, top_p=0.9)

        past_key_values = DynamicCache()
        for t in range(delay_iters):
            # (1, 1, 4096)
            output = self.local_transformer(
                inputs_embeds=local_embeds,
                past_key_values=past_key_values,
                return_dict=True,
                use_cache=True,
            )
            hidden_state = output.last_hidden_state
            past_key_values = output.past_key_values

            local_embeds = torch.zeros_like(local_embeds)
            for idx in range(self.audio_channels):  # 8
                cur_start = self.delay_pattern[idx]
                cur_end = cur_start + self.group_size
                cur_empty = self.speech_empty_ids[idx]
                if cur_start <= t < cur_end:
                    cur_lm_head = self.local_transformer_lm_heads[idx]
                    cur_scores: torch.Tensor = cur_lm_head(hidden_state)[:, -1, :]  # （1，1025）

                    # [ vocab_size]
                    cur_token = local_sampler.sample(
                        cur_scores,
                        [cur_empty],
                    )

                    local_tokens[t - cur_start, idx] = cur_token
                    cur_input_embed = self.speech_embeddings[idx](cur_token)

                    if self.speech_embeddings_to_local is not None:
                        cur_input_embed = self.speech_embeddings_to_local(cur_input_embed)
                    local_embeds += cur_input_embed

        return local_tokens  # [B, group_size, audio_channels]

    def _prepare_input_parameters(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor],
    ) -> int:
        if input_ids is not None and torch.is_tensor(input_ids):
            if input_ids.ndim == 1:
                new_len = int(input_ids.shape[0])
            else:
                new_len = int(input_ids.shape[1])
        else:
            new_len = int(inputs_embeds.shape[1])
        return new_len

    def _should_merge_multimodal_embedding(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor],
        is_capturing: bool,
        kwargs: dict,
    ) -> tuple[bool, dict]:
        _merge_multimodal_embedding = False

        # Only for multimodal inputs audio generation processing(Input_ids=151667), inputs_embeds will be all zeros
        seq_len = input_ids.shape[1] if input_ids.ndim == 2 else input_ids.shape[0]

        # Determine if it is audio-only input (avoid calling .item() during capture)
        if not is_capturing:
            # Can safely check conditions in non-capture mode
            if (
                torch.is_tensor(input_ids)
                and input_ids.view(-1).numel() == 1
                and input_ids.view(-1)[0].item() == self.empty_token_id
                and torch.all(inputs_embeds == 0)
            ):
                _merge_multimodal_embedding = True

        if _merge_multimodal_embedding:
            kwargs["audio_embeds"] = inputs_embeds
        else:
            # kwargs['audio_embeds'] = torch.zeros((1, 1, input_ids.shape[1], 4096)).to(input_ids.device)
            kwargs["audio_embeds"] = self._audio_embeds_buffer[:, :, :seq_len, :]

        kwargs["mimo_audio_codes_processing"] = False

        return _merge_multimodal_embedding, kwargs

    def _load_cached_state(
        self,
        request_ids: Optional[list[str]],
    ) -> tuple[Optional[DynamicCache], dict[str, torch.Tensor]]:
        past_key_values: Optional[DynamicCache] = None
        prev_new_audio_emb_by_req: dict[str, torch.Tensor] = {}

        if hasattr(self, "_cached_new_audio_emb_by_req"):
            for req_id, cached_emb in self._cached_new_audio_emb_by_req.items():
                if req_id not in prev_new_audio_emb_by_req:
                    prev_new_audio_emb_by_req[req_id] = cached_emb

        if hasattr(self, "_cached_past_key_values_by_req"):
            for req_id in request_ids or []:
                if req_id in self._cached_past_key_values_by_req:
                    past_key_values = self._cached_past_key_values_by_req[req_id]

        return past_key_values, prev_new_audio_emb_by_req

    def _prepare_multimodal_embeddings_with_cache(
        self,
        input_ids: torch.Tensor,
        request_ids: Optional[list[str]],
        prev_new_audio_emb_by_req: dict[str, torch.Tensor],
        kwargs: dict,
    ) -> torch.Tensor:
        # This multimodal_embeddings is zero-valued, will later retrieve previously generated audio codes embeddings
        multimodal_embeddings = self.embed_multimodal(**kwargs)
        # If previous new_audio_emb exists, add it to multimodal_embeddings
        # In multi-request scenarios, need to select corresponding prev_new_audio_emb based on request_id
        if prev_new_audio_emb_by_req:
            # Simplified handling: if there is only one request, use that request's prev_new_audio_emb
            # If there are multiple requests, use the first request's (or select according to actual needs)
            if request_ids and len(request_ids) == 1:
                req_id = request_ids[0]
                prev_new_audio_emb = prev_new_audio_emb_by_req.get(req_id)
            elif prev_new_audio_emb_by_req:
                # When there are multiple requests, use the first one found (or select according to actual needs)
                prev_new_audio_emb = next(iter(prev_new_audio_emb_by_req.values()))
            else:
                prev_new_audio_emb = None
            if prev_new_audio_emb is not None:
                # Add prev_new_audio_emb to multimodal_embeddings
                # multimodal_embeddings is a tuple, each element is a tensor
                if multimodal_embeddings and len(multimodal_embeddings) > 0:
                    # Add prev_new_audio_emb as a new audio embedding to the list
                    multimodal_embeddings = multimodal_embeddings[0].unsqueeze(0) + prev_new_audio_emb
                else:
                    multimodal_embeddings = prev_new_audio_emb

        inputs_embeds = self.embed_input_ids(input_ids, multimodal_embeddings)
        return inputs_embeds

    def _generate_speech_tokens_and_audio_embeddings(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        next_speech_tokens = None
        new_audio_emb = None

        # if id is empty_token_id, then will be use hs to do local forward
        hs_downsampled = self.hidden_states_downcast(hidden_states[:, -1:, :])

        next_speech_tokens = self.local_forward(
            local_embeds=hs_downsampled[0],
            local_sampler=self.local_sampler,
        )

        # 4,8,4096 - Use pre-allocated buffer and zero it to avoid dynamic allocation
        new_audio_emb = self._new_audio_emb_buffer.zero_()

        next_speech_tokens = next_speech_tokens.to(torch.int32).T.unsqueeze(0).unsqueeze(0)
        for idx in range(self.audio_channels):
            cur_empty = self.speech_empty_ids[idx]
            cur_embed = self.speech_embeddings[idx]

            cur_speech_ids = next_speech_tokens[:, :, idx, :]
            cur_speech_embeds: torch.Tensor = cur_embed(cur_speech_ids)

            cur_mask = cur_speech_ids == cur_empty
            cur_speech_embeds.masked_fill_(cur_mask.unsqueeze(-1), 0.0)

            new_audio_emb += cur_speech_embeds

        new_audio_emb_input_lf = self.input_local_transformer(
            inputs_embeds=new_audio_emb.squeeze(0),
            return_dict=True,
            is_causal=False,
        )  # 1,1,4,1024

        new_audio_emb_last = new_audio_emb_input_lf.last_hidden_state.view(1, 1, -1)
        new_audio_emb_downcast = self.speech_group_downcast(new_audio_emb_last)[0]
        new_audio_emb = new_audio_emb_downcast.clone()

        return next_speech_tokens, new_audio_emb

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        B = input_ids.shape[0]
        request_ids: Optional[list[str]] = kwargs.get("request_ids")
        is_capturing = torch.cuda.is_current_stream_capturing()
        new_len = self._prepare_input_parameters(input_ids, inputs_embeds)

        _merge_multimodal_embedding, kwargs = self._should_merge_multimodal_embedding(
            input_ids, inputs_embeds, is_capturing, kwargs
        )

        past_key_values, prev_new_audio_emb_by_req = self._load_cached_state(request_ids)

        past_len = self._get_past_len(past_key_values)
        attn_len = past_len + new_len

        # Use pre-allocated buffer or dynamically create in non-capture mode
        if is_capturing:
            # Use pre-allocated buffer during CUDA graph capture
            if not hasattr(self, "_attention_mask_buffer") or self._attention_mask_buffer.shape[1] < attn_len:
                # This branch should not execute during capture, but as a safety measure
                attention_mask = torch.ones((B, attn_len), device=input_ids.device, dtype=torch.bool)
            else:
                attention_mask = self._attention_mask_buffer[:B, :attn_len]
        else:
            attention_mask = torch.ones(
                (B, attn_len),
                device=input_ids.device,
                dtype=torch.bool,
            )

        # NOTE: In v1, inputs_embeds is always generated at model runner, this
        # condition is for v0 compatibility.
        if _merge_multimodal_embedding:
            inputs_embeds = self._prepare_multimodal_embeddings_with_cache(
                input_ids, request_ids, prev_new_audio_emb_by_req, kwargs
            )

        # Do not pass attention_mask during CUDA graph capture to avoid HuggingFace internal .all() calls
        model_attention_mask = None if is_capturing else attention_mask

        outputs = self.model(
            attention_mask=model_attention_mask,
            position_ids=positions,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=True,
        )
        hidden_states = outputs.last_hidden_state
        new_past_key_values = outputs.past_key_values

        logits = self.compute_logits(hidden_states)
        next_ids = self.global_sampler.sample(logits, removed_tokens=self.removed_tokens)

        new_audio_emb = None
        next_speech_tokens = None

        # Skip this conditional branch during CUDA graph capture, as int(next_ids[0]) will trigger GPU-CPU sync
        should_do_local_forward = False
        if not is_capturing and next_ids is not None and len(next_ids) == 1 and int(next_ids[0]) == self.empty_token_id:
            should_do_local_forward = True

        if should_do_local_forward:
            next_speech_tokens, new_audio_emb = self._generate_speech_tokens_and_audio_embeddings(hidden_states)

        self._update_request_caches(request_ids, new_past_key_values, new_audio_emb)

        return next_speech_tokens, hidden_states

    def _update_request_caches(
        self,
        request_ids: Optional[list[str]],
        new_past_key_values: Optional[DynamicCache],
        new_audio_emb: Optional[torch.Tensor],
    ) -> None:
        if new_past_key_values is not None:
            if request_ids and len(request_ids) == 1:
                req_id = request_ids[0]
                self._cached_past_key_values_by_req[req_id] = new_past_key_values

        # If new_audio_emb is generated, need to store it for next round use
        # In multi-request scenarios, need to store each request's new_audio_emb based on request_id
        if new_audio_emb is not None:
            # Determine current request's request_id
            # If there is only one request, use that request's request_id
            # If there are multiple requests, need to determine based on actual situation (simplified handling here)
            if request_ids and len(request_ids) == 1:
                req_id = request_ids[0]
                self._cached_new_audio_emb_by_req[req_id] = new_audio_emb

            elif request_ids and len(request_ids) > 1:
                # TODO: Determine correspondence between new_audio_emb and request_id based on actual needs
                req_id = request_ids[0]
                self._cached_new_audio_emb_by_req[req_id] = new_audio_emb

            else:
                # Case without request_ids (backward compatibility)
                # Use default key or first available key
                if not self._cached_new_audio_emb_by_req:
                    default_key = "default"
                else:
                    default_key = next(iter(self._cached_new_audio_emb_by_req.keys()))
                self._cached_new_audio_emb_by_req[default_key] = new_audio_emb

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if hidden_states.ndim == 2:
            text_logits: torch.Tensor = self.lm_head(hidden_states)
            logits = text_logits.clone()
        else:
            text_logits: torch.Tensor = self.lm_head(hidden_states[:, -1:, :])
            logits = text_logits[:, -1, :].clone()
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))

        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
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
                if name.startswith("model."):
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
                if weight_loader == default_weight_loader:
                    weight_loader(param, loaded_weight)
                else:
                    weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
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
        prompt_ids = kwargs.get("prompt_ids", None)
        _is_first_audio_codes = False if prompt_ids is None else True
        # Original TTS correct running logic
        if audio_input["type"] == "audio_embeds":
            audio_embeds = audio_input["audio_embeds"]
        if (
            not kwargs.get("mimo_audio_codes_processing")
            or (isinstance(audio_embeds, torch.Tensor) and audio_embeds.shape[0] > 1)
            or not _is_first_audio_codes
        ):
            return tuple(audio_embeds)

        prompt_ids = prompt_ids[0][0]
        prompt_ids_length = len(prompt_ids.tolist())
        prompt_ids_expand = self._expand_ids_4x_pad_and_nonpad(
            prompt_ids,
            empty_token_id=self.empty_token_id,
            ignore_id=-100,
        )
        T_groups = prompt_ids_length
        mm_offset = kwargs.get("mm_offset").squeeze()
        audio_lengths = kwargs.get("audio_lengths").squeeze()
        group_size = self.group_size
        audio_codes_list = audio_embeds

        # Convert list-format audio codes to tensor format [B, C, T]
        # Input may be nested list [[c0_codes], [c1_codes], ...] or tensor
        converted_audio_codes_list = []
        for codes in audio_codes_list:
            if isinstance(codes, (list, tuple)):
                codes_tensor = torch.tensor(codes, dtype=torch.long, device=prompt_ids_expand.device)
                if codes_tensor.dim() == 2:
                    codes_tensor = codes_tensor.unsqueeze(0)  # [C, T] -> [1, C, T]
                converted_audio_codes_list.append(codes_tensor)
            else:
                if codes.dim() == 2:
                    codes = codes.unsqueeze(0)  # [C, T] -> [1, C, T]
                converted_audio_codes_list.append(codes)
        audio_codes_list = converted_audio_codes_list

        dtype = audio_codes_list[0].dtype
        device = audio_codes_list[0].device
        B = audio_codes_list[0].shape[0]

        speech_input_ids = torch.zeros(
            (B, self.audio_channels, prompt_ids_length * group_size), dtype=dtype, device=device
        )
        for i, idx in enumerate(self.speech_empty_ids):
            speech_input_ids[:, i, :] = idx

        speech_input_ids = self._overlay_audio_codes_by_prompt_pad_positions(
            speech_input_ids, prompt_ids_expand, audio_codes_list, mm_offset, device
        )

        speech_input_ids = speech_input_ids[:, :, : T_groups * group_size].view(
            B, self.audio_channels, T_groups, group_size
        )

        # Transpose to [B, T_groups, audio_channels, group_size]
        speech_input_ids = speech_input_ids.transpose(1, 2)

        # Determine which positions are speech (text token == empty_idx)
        audio_lengths = audio_lengths // group_size  # 4
        is_speech = (prompt_ids == self.empty_token_id).unsqueeze(0).expand(B, -1)  # [B, T_groups]

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

        if isinstance(audio_lengths, torch.Tensor):
            audio_lengths = audio_lengths.tolist()
        if isinstance(audio_lengths, list):
            seg_lengths = audio_lengths
        else:
            seg_lengths = [audio_lengths]

        speech_embeds_split = self._split_grouped_embeds_by_speech_flag(
            speech_grouped_embeds=speech_grouped_embeds,  # [B, T_groups, H]
            is_speech_1d=(prompt_ids == self.empty_token_id),
            seg_lengths=seg_lengths,
            device=device,
        )

        # To pass sanity_check_mm_encoder_outputs check with dim = 2
        audio_embeds_list = [
            speech_embeds_grouped.reshape(B * speech_embeds_grouped.shape[1], -1)
            for speech_embeds_grouped in speech_embeds_split
        ]

        return tuple(audio_embeds_list)

    def _expand_ids_4x_pad_and_nonpad(
        self,
        prompt_ids: torch.Tensor,
        empty_token_id: int,
        ignore_id: int = -100,
    ) -> torch.Tensor:
        device = prompt_ids.device
        dtype = prompt_ids.dtype

        repeats = torch.full_like(prompt_ids, 4, dtype=torch.long, device=device)
        expanded = torch.repeat_interleave(prompt_ids, repeats)  # [4*T]

        within = torch.arange(expanded.numel(), device=device) % 4

        is_nonpad_expanded = expanded != empty_token_id
        mask_ignore = is_nonpad_expanded & (within != 0)

        expanded = expanded.clone()
        expanded[mask_ignore] = torch.tensor(ignore_id, device=device, dtype=dtype)
        return expanded

    def _overlay_audio_codes_by_prompt_pad_positions(
        self,
        speech_input_ids: torch.Tensor,  # [B, C, L]
        prompt_ids_expand: torch.Tensor,  # [L]  (L == speech_input_ids.shape[-1])
        audio_codes_list: List[torch.Tensor],  # each [B, C, T_i]
        mm_offset_groups: Optional[torch.Tensor] = None,
        device: torch.device = None,
    ) -> torch.Tensor:
        B, C, L = speech_input_ids.shape
        assert prompt_ids_expand.numel() == L, (
            f"Length mismatch: prompt_ids_expand={prompt_ids_expand.numel()} vs L={L}"
        )

        prompt_ids_expand = prompt_ids_expand.to(device)

        pad_positions = (prompt_ids_expand == self.empty_token_id).nonzero(as_tuple=True)[0]  # [K]

        # (Avoid list order not being chronological)
        if mm_offset_groups is not None:
            mm_offset_groups = mm_offset_groups.to(device).long()
            assert mm_offset_groups.numel() == len(audio_codes_list), (
                f"mm_offset_groups ({mm_offset_groups.numel()}) != num_segs ({len(audio_codes_list)})"
            )

            order = torch.argsort(mm_offset_groups)
            order_indices = order.flatten().tolist()
            audio_codes_list = [audio_codes_list[int(i)] for i in order_indices]

        cat_codes = torch.cat(audio_codes_list, dim=2).to(device)

        assert cat_codes.shape[0] == B, f"Batch mismatch: cat_codes.B={cat_codes.shape[0]} vs B={B}"
        assert cat_codes.shape[1] == C, f"Channel mismatch: cat_codes.C={cat_codes.shape[1]} vs C={C}"

        K = pad_positions.numel()
        T_total = cat_codes.shape[2]
        N = min(K, T_total)

        if N <= 0:
            return speech_input_ids

        speech_input_ids[:, :, pad_positions[:N]] = cat_codes[:, :, :N]

        return speech_input_ids

    def _split_grouped_embeds_by_speech_flag(
        self,
        speech_grouped_embeds: torch.Tensor,
        is_speech_1d: torch.Tensor,
        seg_lengths: List[int],
        device: torch.device = None,
    ) -> List[torch.Tensor]:
        assert speech_grouped_embeds.dim() == 3, f"expect [B,T,H], got {speech_grouped_embeds.shape}"
        B, T_groups, H = speech_grouped_embeds.shape

        assert is_speech_1d.dim() == 1 and is_speech_1d.numel() == T_groups, (
            f"is_speech_1d should be [T_groups], got {is_speech_1d.shape} vs T_groups={T_groups}"
        )

        is_speech_1d = is_speech_1d.to(device)

        speech_pos = is_speech_1d.nonzero(as_tuple=True)[0]  # [K]
        K = speech_pos.numel()

        segments: List[torch.Tensor] = []
        cursor = 0
        for seg_len in seg_lengths:
            seg_len = int(seg_len)
            if seg_len <= 0:
                continue

            end = cursor + seg_len
            if cursor >= K:
                break

            end = min(end, K)

            pos = speech_pos[cursor:end]
            seg = torch.index_select(speech_grouped_embeds, dim=1, index=pos)
            segments.append(seg)

            cursor = end

        return segments

    def _get_past_len(self, past_key_values: Optional[DynamicCache]) -> int:
        if past_key_values is None:
            return 0

        if hasattr(past_key_values, "get_seq_length"):
            try:
                pl = past_key_values.get_seq_length()
                if pl is not None:
                    return int(pl)
            except Exception:
                pass

        if hasattr(past_key_values, "seen_tokens"):
            try:
                return int(past_key_values.seen_tokens)
            except Exception:
                pass

        try:
            if hasattr(past_key_values, "layers") and len(past_key_values.layers) > 0:
                k = past_key_values.layers[0].keys
                return int(k.shape[-2])

            if hasattr(past_key_values, "key_cache") and len(past_key_values.key_cache) > 0:
                k = past_key_values.key_cache[0]
                return int(k.shape[-2])
        except Exception:
            pass

        return 0
