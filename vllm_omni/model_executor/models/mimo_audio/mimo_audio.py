import os
import time
import numpy as np
from collections.abc import Iterable, Mapping, Sequence
from functools import cached_property
from typing import Optional, Union, Any, List

import torch
from torch import nn
from transformers import Qwen2Config, BatchFeature
from vllm.config import VllmConfig

# from vllm.sequence import IntermediateTensors, SequenceData
from vllm.model_executor.layers.logits_processor import LogitsProcessor

# from vllm.model_executor import SamplingMetadata
from vllm.utils import is_list_of
from vllm.multimodal.inputs import AudioItem, ModalityData
from vllm.model_executor.models import SupportsPP
from vllm.model_executor.models.utils import init_vllm_registered_model, maybe_prefix
from vllm.sequence import IntermediateTensors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

from transformers.models.qwen2_audio import Qwen2AudioProcessor
from vllm_omni.model_executor.models.mimo_audio.config_mimo_audio import MiMoAudioConfig
from vllm_omni.model_executor.models.qwen2_5_omni.qwen2_5_omni import OmniOutput
from vllm_omni.model_executor.models.mimo_audio.mimo_audio_code2wav import MiMoAudioTokenizerWorker
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.model_executor.models.interfaces import SupportsMultiModal
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
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargs,
)

_TOKENIZER_WORKER_CACHE: dict[tuple[str, str, str], MiMoAudioTokenizerWorker] = {}


def _get_tokenizer_worker(
    device: torch.device,
    config_path: str,
    audio_tokenizer_path: str,
) -> MiMoAudioTokenizerWorker:
    key = (str(device), config_path, audio_tokenizer_path)
    if key not in _TOKENIZER_WORKER_CACHE:
        _TOKENIZER_WORKER_CACHE[key] = MiMoAudioTokenizerWorker(
            device_str=str(device),
            config_path=config_path,
            audio_tokenizer_path=audio_tokenizer_path,
        )
    return _TOKENIZER_WORKER_CACHE[key]


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
        # MiMoAudio doesn't use a standard HF processor like Qwen2Audio
        # Instead it uses MiMoAudioTokenizer directly
        # This method may return None or a dummy processor
        # depending on the actual implementation requirements
        return None

    def get_feature_extractor(
        self,
        *,
        # Ignored in initialization
        sampling_rate: Optional[int] = None,
    ):
        # MiMoAudio uses MiMoAudioTokenizer instead of WhisperFeatureExtractor
        # This method may need to return the tokenizer or None
        # depending on the actual implementation requirements
        return None

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
        num_audios = mm_counts.get("audio", 0)
        return "<|empty|>" * num_audios

    def get_dummy_mm_data(self, seq_len: int, mm_counts: Mapping[str, int]) -> MultiModalDataDict:
        num_audios = mm_counts.get("audio", 0)
        if num_audios == 0:
            return {}
        # Return dummy raw audio data (not encoded codes)
        # This will be processed by _parse_audio_data like real audio
        # Use 1 second of audio at target_sr (24000 Hz)
        dummy_audio_length = 100  # 1 second at 24kHz
        dummy_audio = np.zeros((dummy_audio_length,), dtype=np.float32)
        return {"audio": [(dummy_audio, 24000)] * num_audios}

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

        dummy_text = self.get_dummy_text(mm_counts)
        dummy_mm_data = self.get_dummy_mm_data(seq_len, mm_counts)

        dummy_processor_inputs = ProcessorInputs(
            prompt=dummy_text,
            mm_data=dummy_mm_data,
        )

        MiMoAudioLLMDummyInputsBuilder._processor_inputs_cache.put(cache_key, dummy_processor_inputs)
        return dummy_processor_inputs


class MiMoAudioDataParser(MultiModalDataParser):
    def __init__(self, target_sr: int, use_mono_channel: bool = True):
        super().__init__(target_sr=target_sr)
        self.use_mono_channel = use_mono_channel
        self.target_sr = target_sr

        tokenizer_device = os.environ.get("MIMO_AUDIO_TOKENIZER_DEVICE", None)

        if tokenizer_device:
            if tokenizer_device.lower() == "cpu":
                self.device = torch.device("cpu")
            else:
                # 支持 "cuda", "cuda:0", "cuda:6" 等格式
                self.device = torch.device(tokenizer_device)
        else:
            # 默认使用 cuda（会使用 GPU 0）
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.audio_tokenizer_path = os.environ.get("MIMO_AUDIO_TOKENIZER_PATH", None)
        if not self.audio_tokenizer_path:
            raise ValueError(
                "Audio tokenizer path is not set. Provide "
                "`model_config.audio_tokenizer_path` in the stage config "
                "or export MIMO_AUDIO_TOKENIZER_PATH."
            )

        self.tokenizer_config_path = os.environ.get("MIMO_AUDIO_TOKENIZER_CONFIG_PATH", None)

        self.mimo_tokenizer = _get_tokenizer_worker(
            device=self.device,
            config_path=self.tokenizer_config_path,
            audio_tokenizer_path=self.audio_tokenizer_path,
        )

    def _parse_audio_data(
        self,
        data: ModalityData[AudioItem],
    ):
        if data is None:
            return AudioProcessorItems(None)

        # also check single audio item with sampling rate
        if self._is_empty(data) or (isinstance(data, tuple) and self._is_empty(data[0])):
            return None

        if (
            is_list_of(data, float)
            or isinstance(data, (np.ndarray, torch.Tensor))
            and data.ndim == 1
            or isinstance(data, tuple)
        ):
            data_items = [data]
        elif isinstance(data, (np.ndarray, torch.Tensor)):
            data_items = [elem for elem in data]
        else:
            data_items = data

        new_audios = list[np.ndarray]()
        for data_item in data_items:
            if not isinstance(data_item, tuple) or not isinstance(data_item[0], np.ndarray):
                new_audios.append(data_item)
                continue

            audio, orig_sr = self._get_audio_with_sr(data_item)
            if orig_sr is None:
                new_audio = audio
            else:
                new_audio = self.audio_resampler.resample(audio, orig_sr=orig_sr)
            wav_tensor = torch.from_numpy(new_audio.astype(np.float32))
            if wav_tensor.dim() == 1:
                wav_tensor = wav_tensor.unsqueeze(0)
            if self.use_mono_channel:
                if wav_tensor.size(0) > 1:
                    wav_mono = wav_tensor.mean(dim=0)  # [samples]
                else:
                    wav_mono = wav_tensor[0]
            audio_codes = self.mimo_tokenizer.encode(audio=(wav_mono, self.target_sr))
            print("audio_codes11", audio_codes.shape, "orig_sr", orig_sr, "self.target_sr", self.target_sr)
            new_audio = audio_codes.detach().cpu().numpy().astype(np.int64).tolist()
            new_audios.append(new_audio)

        return AudioProcessorItems(new_audios)


class MiMoAudioLLMMultiModalProcessor(BaseMultiModalProcessor[MiMoAudioLLMProcessingInfo]):
    def _get_data_parser(self) -> MultiModalDataParser:
        # MiMoAudio uses MiMoAudioTokenizer which typically uses 24000 Hz sampling rate
        # Adjust this based on the actual tokenizer configuration
        sampling_rate = 24000  # Default for MiMoAudioTokenizer
        return MiMoAudioDataParser(target_sr=sampling_rate)

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, Any],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        sampling_rate = 24000
        audios = mm_data.pop("audios", [])
        tokenizer = self.info.get_tokenizer()
        if audios:
            mm_data["audio"] = audios

        if isinstance(prompt, str):
            prompt_ids = tokenizer.encode(prompt)
        else:
            prompt_ids = prompt

        # Text-only input not supported in composite processor
        if not mm_data.get("audio", []):
            prompt_ids = self._apply_hf_processor_tokens_only(prompt_ids)

            return BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")

        # feature_extractor = self.info.get_feature_extractor(**mm_kwargs)
        mm_kwargs = dict(
            **mm_kwargs,
            sampling_rate=sampling_rate,
        )

        audio_lengths = [len(x) for x in audios]
        if all(audio_length == 8 for audio_length in audio_lengths):
            audio_lengths = [len(x[0]) for x in audios]

        return BatchFeature(
            data={
                "input_ids": torch.tensor([prompt_ids], dtype=torch.int64),
                "audio_embeds": audios,
                "audio_lengths": audio_lengths,
                "mm_data": mm_data,
                "mm_kwargs": mm_kwargs,
                "tok_kwargs": tok_kwargs,
            },
            tensor_type=None,
        )

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        # MiMoAudio may have different field names than Qwen2Audio
        # Adjust based on actual MiMoAudio model requirements
        # For now, return empty dict or adjust based on actual needs
        return {
            "audio_embeds": MultiModalFieldConfig.batched("audio"),
            "audio_lengths": MultiModalFieldConfig.batched("audio"),
        }

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
        audio_token = "<|empty|>"

        audio_bos_id = None
        audio_eos_id = None
        audio_token_id = vocab.get(audio_token, None)

        # If tokens don't exist, try alternative tokens or use None
        if audio_token_id is None:
            # Try alternative token names or skip audio token replacement
            return []

        audio_kwargs = out_mm_kwargs.get("audio", [])
        # feature_attention_mask = out_mm_kwargs.get("feature_attention_mask", None)

        if audio_kwargs:
            audio_output_lengths = [item["audio_lengths"].data for item in audio_kwargs]
        else:
            audios = mm_items.get_items("audio", AudioProcessorItems)
            if audios is not None:
                num_audios = audios.get_count()
                audio_output_lengths = [audios.get_audio_length(i) for i in range(num_audios)]
            else:
                audio_output_lengths = []

        def get_replacement_mimo_audio(item_idx: int):
            if item_idx >= len(audio_output_lengths):
                # If no length info, use a default or skip
                return PromptUpdateDetails.select_token_id(
                    [audio_token_id],
                    embed_token_id=audio_token_id,
                )

            num_features = audio_output_lengths[item_idx] // 4
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


@MULTIMODAL_REGISTRY.register_processor(
    MiMoAudioLLMMultiModalProcessor,
    info=MiMoAudioLLMProcessingInfo,
    dummy_inputs=MiMoAudioLLMDummyInputsBuilder,
)
class MiMoAudioForConditionalGeneration(
    nn.Module,
    SupportsPP,
    SupportsMultiModal,
):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        if modality.startswith("image"):
            return "<|vision_start|><|IMAGE|><|vision_end|>"
        if modality.startswith("video"):
            return "<|vision_start|><|VIDEO|><|vision_end|>"
        if modality.startswith("audio"):
            return f"Audio {i}: <|sostm|><|empty|><|eostm|>"

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()

        self.have_multimodal_outputs = True
        config = vllm_config.model_config.hf_config
        config = MiMoAudioConfig(**vars(config)) if isinstance(config, Qwen2Config) else config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.logits_processor = LogitsProcessor(config.vocab_size)

        self.config = config
        self.lora_config = lora_config

        self.quant_config = quant_config
        self.vllm_config = vllm_config
        print("vllm_config", vllm_config)

        self.llm_prev_hidden_state = None

        if "model_stage" in os.environ:
            self.model_stage = os.environ["model_stage"]
        else:
            self.model_stage = vllm_config.model_config.model_stage
        print("model_stage", self.model_stage)
        t0 = time.perf_counter()
        if self.model_stage == "llm":
            # Initialize llm model (multimodal processing)
            self.llm = init_vllm_registered_model(
                vllm_config=vllm_config,
                # prefix=maybe_prefix(prefix, "model"),
                hf_config=vllm_config.model_config.hf_config,
                # Use registry architecture key
                architectures=["MiMoAudioLLMForConditionalGeneration"],
            )
            print("init llm, finished", self.llm, "time cost", time.perf_counter() - t0)
            t0 = time.perf_counter()
            self.localforward = None
            print("init localforward, finished", self.localforward, "time cost", time.perf_counter() - t0)
            self.token2wav = None
            self.model = self.llm

        elif self.model_stage == "localforward":
            self.llm = None
            # Initialize talker model wrapper (handles projection + LM)
            self.localforward = init_vllm_registered_model(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "localforward"),
                # hf_config=talker_config,
                # Use registry architecture key
                architectures=["MiMoAudioLocalforwardModel"],
            )
            # self.talker.init_multi_modal(thinker_config)
            self.model = self.localforward
            self.token2wav = None
        elif self.model_stage == "code2wav":
            self.llm = None
            self.localforward = None
            # Initialize token2wav (code->mel->wav) like thinker/talker
            self.token2wav = init_vllm_registered_model(
                vllm_config=vllm_config,
                # prefix=maybe_prefix(prefix, "token2wav"),
                # hf_config=self.token2wav_config,
                architectures=["MiMoAudioToken2WavModel"],
            )

            self.model = self.token2wav
        else:
            raise ValueError("Invalid model stage")

        # Set up intermediate tensors
        self.make_empty_intermediate_tensors = (
            (self.llm.make_empty_intermediate_tensors) if self.model_stage == "llm" else lambda: None
        )

    # -------------------- Device utilities --------------------
    @staticmethod
    def _module_device(module: nn.Module) -> torch.device:
        try:
            return next(module.parameters()).device
        except StopIteration:
            # No parameters; fall back to buffers or cpu
            for _, buf in module.named_buffers(recurse=True):
                return buf.device
            return torch.device("cpu")

    def move_submodules_to_devices(
        self,
        *,
        llm_device: Optional[Union[str, torch.device]] = None,
        localforward_device: Optional[Union[str, torch.device]] = None,
        token2wav_device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """Optionally move thinker/talker/token2wav to different devices.

        Example:
            model.move_submodules_to_devices(
                thinker_device='cuda:0',
                talker_device='cuda:1',
                token2wav_device='cpu',
            )
        """
        if llm_device is not None and self.llm is not None:
            self.llm.to(llm_device)
        if localforward_device is not None and self.localforward is not None:
            self.localforward.to(localforward_device)
        if token2wav_device is not None and self.token2wav is not None:
            self.token2wav.to(token2wav_device)

    @cached_property
    def sampler(self):
        if hasattr(self.model, "sampler"):
            return self.model.sampler
        return Sampler()

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
    ) -> torch.Tensor:
        if self.model_stage == "code2wav":
            return torch.zeros_like(input_ids).reshape(-1, 1).repeat(1, self.vllm_config.model_config.get_hidden_size())
        return self.model.get_input_embeddings(input_ids, multimodal_embeddings)

    def get_multimodal_embeddings(self, **kwargs):
        # Delegate to thinker model for multimodal processing
        return self.model.get_multimodal_embeddings(**kwargs)

    def last_index_of(self, list, value):
        return len(list) - 1 - list[::-1].index(value)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights for all components of the omni model."""
        loaded_weights = set()
        llm_weights = []
        localforward_weights = []
        token2wav_weights = []
        for k, v in weights:
            # print('loading weights for {}'.format(k))
            if (
                k.startswith("model.")
                or k.startswith("lm_head.")
                or k.startswith("input_local_transformer.")
                or k.startswith("hidden_states_downcast.")
                or k.startswith("local_transformer")
                or k.startswith("speech_embeddings")
                or k.startswith("speech_group_downcast")
            ):
                if self.llm:
                    llm_weights.append((k, v))
            elif k.startswith("local_transformer") or k.startswith("speech_embeddings"):
                if self.localforward:
                    localforward_weights.append((k, v))
            elif k.startswith("mel_transform."):
                if self.token2wav:
                    token2wav_weights.append((k, v))
            else:
                pass

        # Load llm weights
        if self.llm:
            if llm_weights:
                llm_loaded = self.llm.load_weights(llm_weights)
            else:
                llm_loaded = set([k for k, v in llm_weights])
            # thinker_loaded = add_prefix_to_loaded_weights(thinker_loaded, "thinker")
            loaded_weights.update(llm_loaded)
        # Load localforward weights
        if self.localforward:
            localforward_loaded = self.localforward.load_weights(localforward_weights)
            loaded_weights.update(localforward_loaded)

        ## load code2wav
        if self.token2wav:
            token2wav_loaded = self.token2wav.load_weights(
                token2wav_weights,
                os.environ["MIMO_AUDIO_TOKENIZER_PATH"],
            )

            loaded_weights.update(token2wav_loaded)

    def apply_input_local_transformer(self, speech_embeddings: torch.Tensor):
        T_groups, group_size, hidden_size = speech_embeddings.shape

        # Process each group independently: [B*T//group_size, group_size, hidden_size]
        input_embeddings = speech_embeddings.reshape(T_groups, group_size, hidden_size)

        output = self.llm.input_local_transformer(
            inputs_embeds=input_embeddings,
            return_dict=True,
            is_causal=False,  # for SDPA
        )
        encoded_embeddings = output.last_hidden_state

        # Reshape back to original format
        # [B*T//group_size, group_size, hidden_size] -> [B, T//group_size, group_size, hidden_size]
        encoded_embeddings = encoded_embeddings.reshape(T_groups, group_size, hidden_size)

        return encoded_embeddings

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
    ) -> torch.Tensor:
        if self.model_stage == "code2wav":
            return torch.zeros_like(input_ids).reshape(-1, 1).repeat(1, self.vllm_config.model_config.get_hidden_size())
        return self.model.get_input_embeddings(input_ids, multimodal_embeddings)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        generate_audio: bool = True,
        voice_type: str = "柚子",
        codec: Optional[torch.Tensor] = None,
        sampling_metadata: Optional[SamplingMetadata] = None,
        logits_index: Optional[int] = None,
        sampler=None,
        additional_information: Optional[dict[str, object]] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors, OmniOutput]:
        """
        Workflow:
        1) llm: multimodal understanding → text hidden states.
        2) If audio requested and codec not provided, use talker to derive codec.
        3) If audio requested (or codec provided), use token2wav to synthesize waveform.
        4) Return text hidden states (and audio when applicable).
        """
        # print("model_stage forward:", self.model_stage)
        if self.model_stage == "llm":
            # Normalize to batched inputs if caller provides 1D/2D unbatched tensors
            added_batch_dim = False
            if input_ids is not None and input_ids.ndim == 1:
                input_ids = input_ids.unsqueeze(0)
                added_batch_dim = True
            if positions is not None and positions.ndim == 1:
                positions = positions.unsqueeze(0)
                added_batch_dim = True
            if inputs_embeds is not None and inputs_embeds.ndim == 2:
                inputs_embeds = inputs_embeds.unsqueeze(0)
                added_batch_dim = True
            llm_dev = self._module_device(self.llm)

            # if input_ids is None, set it to a zero tensor, in the length of the
            # same as the embedding seq length
            if input_ids is None:
                input_ids = torch.zeros(inputs_embeds.shape[1], dtype=torch.long, device=llm_dev).unsqueeze(0)  # (1, 0)
                added_batch_dim = True

            # 1) Thinker (ensure inputs on thinker's device)
            if input_ids is not None and input_ids.device != llm_dev:
                input_ids = input_ids.to(llm_dev)
            if positions is not None and positions.device != llm_dev:
                positions = positions.to(llm_dev)
            if inputs_embeds is not None and inputs_embeds.device != llm_dev:
                inputs_embeds = inputs_embeds.to(llm_dev)

            # Run llm
            llm_output = self.model(
                input_ids=input_ids,
                positions=positions[0],
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )

            next_speech_tokens = None
            if isinstance(llm_output, tuple):
                if len(llm_output) == 2:
                    next_speech_tokens, text_hidden_states = llm_output
                elif len(llm_output) == 3:
                    ids, next_speech_tokens, text_hidden_states = llm_output
            else:
                text_hidden_states = llm_output

            multimodal_outputs: dict = None

            # Text-only path
            return OmniOutput(
                text_hidden_states=(text_hidden_states.squeeze(0) if added_batch_dim else text_hidden_states),
                multimodal_outputs={"code": next_speech_tokens},
            )

        if self.model_stage == "code2wav":
            code = (
                input_ids
                if input_ids is not None
                else torch.zeros(
                    inputs_embeds.shape[0],
                    dtype=torch.long,
                    device=inputs_embeds.device,
                )
            )

            audio_tensor = self.generate_audio(code, voice_type)
            return OmniOutput(text_hidden_states=None, multimodal_outputs={"audio": audio_tensor})

    def generate_audio(self, code, voice_type):
        token2wav_dev = self._module_device(self.token2wav)
        if isinstance(code, torch.Tensor):
            code_tensor = code.to(dtype=torch.long, device=token2wav_dev)
        else:
            code_tensor = torch.as_tensor(code, dtype=torch.long, device=token2wav_dev)
        if code_tensor.ndim == 2 and code_tensor.shape[0] == 1:
            code_tensor = code_tensor.squeeze(0)

        with torch.inference_mode():
            audio_tensor = self.token2wav(
                codes=code_tensor,
            )

        return audio_tensor

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        # sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states

        # Use thinker model for logits computation
        return self.model.compute_logits(hidden_states)
