import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional

import torch
from transformers import BatchFeature, Qwen2AudioProcessor, Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Model as TransformerQwen2Model,
)
from vllm.multimodal import MultiModalDataDict
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargs
from vllm.multimodal.parse import AudioProcessorItems, MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder, ProcessorInputs
from vllm.utils import LRUCache


class MiMoAudioQwen2Model(TransformerQwen2Model):
    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        if hasattr(self, "embed_tokens"):
            del self.embed_tokens


# Helper function for feature extraction output lengths
# Note: MiMoAudio uses a different audio processing pipeline than Qwen2Audio
# This is a placeholder - may need to be adjusted based on actual MiMoAudio tokenizer
def _get_feat_extract_output_lengths(input_lengths: torch.Tensor):
    """
    Calculate output lengths for audio feature extraction.
    For MiMoAudio, this might need to be adjusted based on the actual tokenizer.
    """
    # Placeholder implementation - adjust based on MiMoAudio tokenizer's actual behavior
    # This is similar to Qwen2Audio's implementation but may need modification
    feat_lengths = (input_lengths - 1) // 2 + 1
    output_lengths = (feat_lengths - 2) // 2 + 1
    return feat_lengths, output_lengths


class MiMoAudioLLMProcessingInfo(
    BaseProcessingInfo,
):
    def get_hf_config(self):
        return self.ctx.get_hf_config(Qwen2Config)

    def get_hf_processor(
        self,
        *,
        # Ignored in initialization
        sampling_rate: Optional[int] = None,
        **kwargs: object,
    ):
        return self.ctx.get_hf_processor(
            Qwen2AudioProcessor,
            use_fast=kwargs.pop("use_fast", True),
            **kwargs,
        )

    def get_feature_extractor(
        self,
        *,
        # Ignored in initialization
        sampling_rate: Optional[int] = None,
        **kwargs: object,
    ):
        hf_processor = self.get_hf_processor(**kwargs)
        feature_extractor = hf_processor.feature_extractor  # type: ignore
        # assert isinstance(feature_extractor, WhisperFeatureExtractor)
        return feature_extractor

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"audio": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        # MiMoAudio has different tokenization than Qwen2Audio
        # This is a placeholder - adjust based on actual MiMoAudio tokenizer limits
        hf_config = self.get_hf_config()
        # For now, return a large value or None
        # This should be adjusted based on MiMoAudio's actual constraints
        return {"audio": 1}


class MiMoAudioLLMDummyInputsBuilder(BaseDummyInputsBuilder[MiMoAudioLLMProcessingInfo]):
    _processor_inputs_cache: LRUCache = LRUCache(capacity=1024)

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return ""

    def get_dummy_mm_data(self, seq_len: int, mm_counts: Mapping[str, int]) -> MultiModalDataDict:
        return {"audio": 0}

    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> ProcessorInputs:
        num_audios = mm_counts.get("audio", 0)
        cache_key = f"mimo_audio_dummy_processor_inputs_{seq_len}_{num_audios}"
        dummy_processor_inputs = MiMoAudioLLMDummyInputsBuilder._processor_inputs_cache.get(cache_key)
        if dummy_processor_inputs is not None:
            return dummy_processor_inputs

        # Generate dummy audio data for MiMoAudio
        # Note: MiMoAudio uses a different audio format than Qwen2Audio
        # The actual format depends on MiMoAudioTokenizer's requirements
        # For now, we create dummy audio data similar to Qwen2Audio

        # Get sampling rate from config or use default
        # MiMoAudioTokenizer typically uses 24000 Hz sampling rate
        sampling_rate = 24000
        # Use a reasonable chunk length for dummy audio (e.g., 1 second)
        audio_len = sampling_rate

        mm_data = {"audio": self._get_dummy_audios(length=audio_len, num_audios=num_audios)} if num_audios > 0 else {}

        # Create dummy prompt text
        # MiMoAudio may use different audio tokens than Qwen2Audio
        # Adjust based on actual tokenizer vocabulary
        prompt_text = "" if num_audios == 0 else "<|AUDIO|>" * num_audios

        dummy_processor_inputs = ProcessorInputs(
            prompt=prompt_text,
            mm_data=mm_data,
        )

        MiMoAudioLLMDummyInputsBuilder._processor_inputs_cache.put(cache_key, dummy_processor_inputs)
        return dummy_processor_inputs


class MiMoAudioLLMMultiModalProcessor(BaseMultiModalProcessor[MiMoAudioLLMProcessingInfo]):
    def _get_data_parser(self) -> MultiModalDataParser:
        # MiMoAudio uses MiMoAudioTokenizer which typically uses 24000 Hz sampling rate
        # Adjust this based on the actual tokenizer configuration
        sampling_rate = 24000  # Default for MiMoAudioTokenizer
        return MultiModalDataParser(target_sr=sampling_rate)

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, Any],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        mm_data = dict(mm_data)
        audios = mm_data.pop("audios", [])

        # NOTE: WhisperFeatureExtractor cannot handle empty list of audios
        if audios:
            # NOTE: Qwen2.5-Omni processor accept "audio"
            mm_data["audio"] = audios
            mm_kwargs = dict(
                **mm_kwargs,
            )

        hf_inputs = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )
        return hf_inputs

        # Text-only input not supported in composite processor
        # mm_data['audio']=[]
        # if not mm_data.get("audio", []):
        #     prompt_ids = self.info.get_tokenizer().encode(prompt)
        #     prompt_ids = self._apply_hf_processor_tokens_only(prompt_ids)
        #
        #     return BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")
        #
        # # feature_extractor = self.info.get_feature_extractor(**mm_kwargs)
        # mm_kwargs = dict(
        #     **mm_kwargs,
        #     sampling_rate=24000
        # )
        # # mm_kwargs={}
        #
        # return super()._call_hf_processor(
        #     prompt=prompt,
        #     mm_data=mm_data,
        #     mm_kwargs=mm_kwargs,
        #     tok_kwargs=tok_kwargs,
        # )

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        # MiMoAudio may have different field names than Qwen2Audio
        # Adjust based on actual MiMoAudio model requirements
        # For now, return empty dict or adjust based on actual needs
        return {}

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        # MiMoAudio doesn't use a standard HF processor like Qwen2Audio
        # Get tokenizer and vocabulary directly
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()

        # MiMoAudio may use different audio tokens
        # Adjust these based on actual tokenizer vocabulary
        # Common tokens in MiMoAudio might be different from Qwen2Audio
        audio_token = "<|AUDIO|>"  # Adjust if different
        audio_bos_token = "<|audio_bos|>"  # Adjust if different
        audio_eos_token = "<|audio_eos|>"  # Adjust if different

        # Get token IDs if they exist in vocabulary
        audio_token_id = vocab.get(audio_token, None)
        audio_bos_id = vocab.get(audio_bos_token, None)
        audio_eos_id = vocab.get(audio_eos_token, None)

        # If tokens don't exist, try alternative tokens or use None
        if audio_token_id is None:
            # Try alternative token names or skip audio token replacement
            return []

        audio_feature_lengths = out_mm_kwargs.get("audio_feature_lengths")
        feature_attention_mask = out_mm_kwargs.get("feature_attention_mask")

        if audio_feature_lengths is None and feature_attention_mask is None:
            audio_output_lengths = []
        elif audio_feature_lengths is not None:
            _, audio_output_lens = _get_feat_extract_output_lengths(audio_feature_lengths)
            audio_output_lengths = audio_output_lens.tolist()
        elif feature_attention_mask is not None:
            assert isinstance(feature_attention_mask, torch.Tensor)
            _, audio_output_lens = _get_feat_extract_output_lengths(feature_attention_mask.sum(-1))
            audio_output_lengths = audio_output_lens.tolist()
        else:
            audio_output_lengths = []

        def get_replacement_mimo_audio(item_idx: int):
            if item_idx >= len(audio_output_lengths):
                # If no length info, use a default or skip
                return PromptUpdateDetails.select_token_id(
                    [audio_token_id],
                    embed_token_id=audio_token_id,
                )

            num_features = audio_output_lengths[item_idx]
            if num_features == 0:
                try:
                    audios = mm_items.get_items("audio", AudioProcessorItems)
                    audio_len = audios.get_audio_length(item_idx)
                    raise ValueError(f"The audio (len={audio_len}) is too short to be represented inside the model")
                except (AttributeError, KeyError):
                    # If AudioProcessorItems is not available, use default
                    num_features = 1

            audio_tokens = [audio_token_id] * num_features

            # Build replacement tokens following Qwen2Audio pattern
            # [audio_bos_id] + audio_tokens + [audio_eos_id]
            replacement_tokens = []
            if audio_bos_id is not None:
                replacement_tokens.append(audio_bos_id)
            replacement_tokens.extend(audio_tokens)
            if audio_eos_id is not None:
                replacement_tokens.append(audio_eos_id)

            # If no bos/eos tokens, just use audio tokens
            if not replacement_tokens:
                replacement_tokens = audio_tokens

            return PromptUpdateDetails.select_token_id(
                replacement_tokens,
                embed_token_id=audio_token_id,
            )

        return [
            PromptReplacement(
                modality="audio",
                target=audio_token,
                replacement=get_replacement_mimo_audio,
            )
        ]


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


PRINT_SHAPE = os.environ.get("PRINT_SHAPE", "1")


def print_shape(**tensors):
    if PRINT_SHAPE == "0":
        return
    shape_list = []
    for name, tensor in tensors.items():
        if tensor is None:
            shape_list.append(f"{name}=None")
        elif isinstance(tensor, torch.Tensor):
            shape_list.append(
                f"{name}={tuple(tensor.shape)},"
                f"dtype={tensor.dtype},"
                f"device={tensor.device},"
                f"sum={tensor.sum()},"
                f"values={str(tensor.tolist())[:200]}...{str(tensor.tolist())[-200:]}"
            )
        else:
            shape_list.append(f"{name}={tensor}")
    print("print_shape", shape_list)
