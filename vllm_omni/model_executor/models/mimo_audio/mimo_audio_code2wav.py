import logging
import os
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Optional, Union

import torch
import torchaudio
from torch import nn
from torchaudio.transforms import MelSpectrogram
from transformers import AutoTokenizer, Qwen2Config
from vllm.config import VllmConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.models import SupportsPP
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

from vllm_omni.model_executor.models.mimo_audio.config_mimo_audio import MiMoAudioConfig
from vllm_omni.model_executor.models.mimo_audio.modeling_audio_tokenizer import MiMoAudioTokenizer
from vllm_omni.model_executor.models.mimo_audio.myutils import print_shape

logger = logging.getLogger(__name__)


class MiMoAudioTokenizerWorker:
    def __init__(
        self,
        device_str: str,
        config_path: str,
        audio_tokenizer_path: str,
    ):
        self.device = device_str
        logger.info("[tokenizer worker] Loading MiMoAudioConfig from %s", config_path)
        load_cfg_start = time.monotonic()
        self.config: MiMoAudioConfig = MiMoAudioConfig.from_pretrained(config_path)
        logger.info(
            "[tokenizer worker] Config loaded in %.2fs",
            time.monotonic() - load_cfg_start,
        )

        if device_str == "cpu":
            model_dtype = torch.float32  # CPU 必须使用 float32
        else:
            model_dtype = torch.bfloat16  # GPU 使用 bfloat16

        logger.info(
            "[tokenizer worker] Loading MiMo-Audio Tokenizer from %s (device=%s, dtype=%s)",
            audio_tokenizer_path,
            self.device,
            model_dtype,
        )
        start_loading_slm_tokenizer_time = time.monotonic()
        self.audio_tokenizer = MiMoAudioTokenizer.from_pretrained(
            audio_tokenizer_path,
            dtype=model_dtype,
            device_map={"": self.device},
        )

        logger.info(
            f"Audio Tokenizers loaded in {time.monotonic() - start_loading_slm_tokenizer_time:.2f} seconds, device: {self.device}"
        )

        # print("self.audio_tokenizer_codebook", self.audio_tokenizer.codebook_size)
        logger.info(
            "[tokenizer worker] Building MelSpectrogram transform (sr=%s, n_fft=%s)",
            self.audio_tokenizer.config.sampling_rate,
            self.audio_tokenizer.config.nfft,
        )
        mel_start = time.monotonic()
        self.mel_transform = (
            MelSpectrogram(
                sample_rate=self.audio_tokenizer.config.sampling_rate,
                n_fft=self.audio_tokenizer.config.nfft,
                hop_length=self.audio_tokenizer.config.hop_length,
                win_length=self.audio_tokenizer.config.window_size,
                f_min=self.audio_tokenizer.config.fmin,
                f_max=self.audio_tokenizer.config.fmax,
                n_mels=self.audio_tokenizer.config.n_mels,
                power=1.0,
                center=True,
            )
            .to(self.device)
            .to(torch.float32)
        )
        logger.info(
            "[tokenizer worker] MelSpectrogram ready in %.2fs",
            time.monotonic() - mel_start,
        )

        self.group_size = self.config.group_size
        self.audio_channels = self.config.audio_channels
        self.sample_rate = self.audio_tokenizer.config.sampling_rate

        # Warmup (skip for CPU due to potential shape mismatch issues)
        if device_str != "cpu":
            logger.info("[tokenizer worker] Running warmup encode/decode...")
            warmup_start = time.monotonic()
            warmup_dtype = torch.float32 if device_str == "cpu" else torch.bfloat16
            try:
                self.encode_wav_base(torch.zeros(self.sample_rate, dtype=warmup_dtype))
                self.decode(torch.zeros(self.audio_channels, self.group_size, dtype=torch.long))
                logger.info(
                    "[tokenizer worker] Warmup finished in %.2fs",
                    time.monotonic() - warmup_start,
                )
            except Exception as e:
                logger.warning("[tokenizer worker] Warmup failed (non-critical): %s", str(e))
        else:
            logger.info("[tokenizer worker] Skipping warmup for CPU device")

    def resample_audio_if_needed(self, wav_tensor: torch.Tensor, original_sr: int):
        """Resample audio if sample rate doesn't match config"""
        target_sr = self.sample_rate
        if original_sr != target_sr:
            wav_tensor = torchaudio.functional.resample(wav_tensor, original_sr, target_sr)
        return wav_tensor

    def wav2mel(self, wav: torch.Tensor):
        """Convert waveform to mel spectrogram using consistent processing"""
        wav = wav.to(torch.float32)
        spec = self.mel_transform(wav[None, :])
        return torch.log(torch.clip(spec, min=1e-7)).squeeze()

    @torch.inference_mode()
    def encode_wav_base(
        self,
        wav: torch.Tensor,  # [samples] cpu
    ) -> torch.Tensor:
        """Run this in cuda stream if available"""
        wav = wav.to(self.device)

        print("wavbeforewav2mel", wav.shape)
        mel = self.wav2mel(wav).transpose(0, 1)  # (seq_len, n_mels)
        input_features = mel  # [seq_len, n_mels]
        input_lens = torch.tensor([mel.shape[0]], device=self.device)
        codes_packed, _ = self.audio_tokenizer.encoder.encode(
            input_features=input_features,
            input_lens=input_lens,
            return_codes_only=True,
        )
        codes = codes_packed.transpose(0, 1).detach()
        audio_codes = codes[:, : self.audio_channels]

        # Pad the sequence to be a multiple of group_size by repeating the last frame
        T = audio_codes.shape[0]
        if T % self.group_size != 0:
            pad = self.group_size - (T % self.group_size)
            last_tokens = audio_codes[-1, :]  # Keep dim for repeat
            padding_tokens = last_tokens.expand(pad, -1)
            audio_codes = torch.cat([audio_codes, padding_tokens], dim=0)

        audio_codes = audio_codes.transpose(0, 1).cpu()

        return audio_codes  # [audio_channels, T] cpu

    @torch.inference_mode()
    def encode(self, audio: tuple[torch.Tensor, int], max_length: float | None = None) -> torch.Tensor:
        """wav: [samples] cpu, sample_rate = 24000"""
        wav, original_sr = audio
        wav = self.resample_audio_if_needed(wav, original_sr)
        if max_length is not None:
            wav = wav[: int(self.sample_rate * max_length)]
        audio_codes = torch.empty((self.audio_channels, 0), dtype=torch.long)  # [audio_channels, T] cpu
        for wav_slice in wav.split(40 * self.sample_rate):
            if wav_slice.shape[0] < self.sample_rate:
                wav_slice = torch.cat(
                    [wav_slice, torch.zeros(self.sample_rate - wav_slice.shape[0])]
                )  # pad to 1 second
            audio_codes_slice = self.encode_wav_base(wav_slice)
            audio_codes = torch.cat([audio_codes, audio_codes_slice], dim=1)
        # audio_codes = self.encode_wav_base(wav)
        return audio_codes.contiguous()  # [audio_channels, T] cpu

    @torch.inference_mode()
    def decode(
        self,
        tokens: torch.Tensor,  # [audio_channels, T] cpu
    ) -> torch.Tensor:
        """Decode audio tokens to waveform using the tokenizer's decoder"""
        tokens = tokens.to(self.device)
        with torch.no_grad():
            decoded_audio: torch.Tensor = self.audio_tokenizer.decode(tokens)
        decoded_audio = decoded_audio.float().reshape(-1).detach().cpu()
        return decoded_audio  # [samples] cpu


@dataclass
class AudioStreamerConfig:
    group_size: int
    audio_channels: int


@dataclass
class MiMoAudioCodes:
    sosp: int
    eosp: int
    sostm: int
    eostm: int
    im_end: int
    pad: int
    eot: int
    empty: int

    @classmethod
    def from_tokenizer(cls, tokenizer: AutoTokenizer) -> "MiMoAudioCodes":
        def idx(token: str) -> int:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if not isinstance(token_id, int):
                raise ValueError(f"Token {token} is not mapped to a single id.")
            return token_id

        return cls(
            sosp=idx("<|sosp|>"),
            eosp=idx("<|eosp|>"),
            sostm=idx("<|sostm|>"),
            eostm=idx("<|eostm|>"),
            im_end=idx("<|im_end|>"),
            pad=idx("<|endoftext|>"),
            eot=idx("<|eot|>"),
            empty=idx("<|empty|>"),
        )


def extract_audio_code_tensor(
    flat_codes: torch.Tensor,
    group_size: int,
    audio_channels: int,
    codes: MiMoAudioCodes,
) -> Optional[torch.Tensor]:
    """Convert flattened talker output into [audio_channels, T] codes."""
    if flat_codes.numel() == 0:
        return None

    group_width = group_size * (audio_channels + 1)
    usable = (flat_codes.numel() // group_width) * group_width
    if usable == 0:
        return None

    groups = flat_codes[:usable].view(-1, group_size, audio_channels + 1)
    audio_buffer: list[torch.Tensor] = []

    for group in groups:
        text_token = int(group[0, 0].item())
        if text_token == codes.empty:
            audio_buffer.append(group[:, 1:])
        elif text_token == codes.eostm:
            break

    if not audio_buffer:
        return None

    audio_tokens = torch.cat(audio_buffer, dim=0).transpose(0, 1).contiguous()
    return audio_tokens  # [audio_channels, T]


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


class MiMoAudioToken2WavForConditionalGenerationVLLM(nn.Module, SupportsPP):
    """Decode MiMo audio codes to waveform for the code2wav stage."""

    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        config = MiMoAudioConfig(**vars(config)) if isinstance(config, Qwen2Config) else config
        self.config = config
        self.vllm_config = vllm_config
        self.quant_config = vllm_config.quant_config
        self.lora_config = vllm_config.lora_config

        self.logits_processor = LogitsProcessor(config.vocab_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sample_rate = getattr(config, "audio_sample_rate", 24000)

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.name_or_path, trust_remote_code=True)
        self.codes = MiMoAudioCodes.from_tokenizer(self.tokenizer)
        self.streamer_config = AudioStreamerConfig(
            group_size=self.config.group_size,
            audio_channels=self.config.audio_channels,
        )

        self.audio_tokenizer_path = getattr(vllm_config.model_config, "audio_tokenizer_path", None) or os.environ.get(
            "MIMO_AUDIO_TOKENIZER_PATH"
        )
        if not self.audio_tokenizer_path:
            raise ValueError(
                "Audio tokenizer path is not set. Provide "
                "`model_config.audio_tokenizer_path` in the stage config "
                "or export MIMO_AUDIO_TOKENIZER_PATH."
            )

        self.tokenizer_config_path = (
            getattr(vllm_config.model_config, "audio_tokenizer_config_path", None)
            or os.environ.get("MIMO_AUDIO_TOKENIZER_CONFIG_PATH")
            or self.config.name_or_path
        )

        self._tokenizer_service: Optional[MiMoAudioTokenizerWorker] = _get_tokenizer_worker(
            device=self.device,
            config_path=self.tokenizer_config_path,
            audio_tokenizer_path=self.audio_tokenizer_path,
        )
        self.debug_echo_codes = os.environ.get("MIMO_AUDIO_ECHO_CODES", "0") not in ("0", "", "false", "False", "FALSE")

    # @property
    # def tokenizer_service(self) -> MiMoAudioTokenizerWorker:
    #     if self._tokenizer_service is None:
    #         self._tokenizer_service = _get_tokenizer_worker(
    #             device=self.device,
    #             config_path=self.tokenizer_config_path,
    #             audio_tokenizer_path=self.audio_tokenizer_path,
    #         )
    #     return self._tokenizer_service

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]], audio__dict_path: str) -> set[str]:
        # Decoder has no trainable weights to load in this stage.
        return set()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        codes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        # Support both input_ids (from vLLM) and codes (from mimo_audio.py)
        code_tensor = codes if codes is not None else input_ids

        print_shape(id="code2wav_forward", code_tensor=code_tensor)
        if code_tensor is None:
            raise ValueError("Either input_ids or codes must be provided")

        if code_tensor.numel() == 0:
            raise ValueError("code_tensor is empty.")

        if self.debug_echo_codes:
            echo = self._echo_code_group(code_tensor)
            if echo.numel() > 0:
                print("[mimo_audio_code2wav.forward] Debug echo mode: returning code group tensor.")
                print(f"  Echo shape: {echo.shape}, dtype: {echo.dtype}, device: {echo.device}")
                return echo

        waveform = self._decode_waveform_from_codes(code_tensor)

        # For warmup/dummy run: if waveform is empty, return dummy hidden_states
        # _dummy_run expects a tensor that can be processed by extract_multimodal_outputs
        if waveform.numel() == 0:
            # Return dummy hidden_states with shape [seq_len, hidden_size]
            # This is needed for warmup to avoid NoneType errors
            hidden_size = getattr(self.config, "hidden_size", None)
            if hidden_size is None:
                # Fallback: try to get from vllm_config.model_config
                hidden_size = getattr(
                    self.vllm_config.model_config,
                    "hidden_size",
                    4096,  # Default fallback
                )

            # Return [seq_len=1, hidden_size] tensor for warmup
            dummy_hidden = torch.zeros(
                (1, hidden_size),
                dtype=torch.bfloat16,
                device=self.device,
            )
            print("[mimo_audio_code2wav.forward] Returning dummy_hidden:")
            print(f"  Type: {type(dummy_hidden)}")
            print(f"  Shape: {dummy_hidden.shape}")
            print(f"  Dtype: {dummy_hidden.dtype}")
            print(f"  Device: {dummy_hidden.device}")
            print(
                f"  Min: {dummy_hidden.min().item():.6f}, Max: {dummy_hidden.max().item():.6f}, Mean: {dummy_hidden.mean().item():.6f}"
            )
            return dummy_hidden

        print("[mimo_audio_code2wav.forward] Returning waveform:")
        print(f"  Type: {type(waveform)}")
        print(f"  Shape: {waveform.shape}")
        print(f"  Dtype: {waveform.dtype}")
        print(f"  Device: {waveform.device}")
        print(f"  Numel: {waveform.numel()}")
        if waveform.numel() > 0:
            print(
                f"  Min: {waveform.min().item():.6f}, Max: {waveform.max().item():.6f}, Mean: {waveform.mean().item():.6f}"
            )
            if waveform.numel() <= 20:
                print(f"  Values: {waveform.cpu().numpy()}")
            else:
                print(f"  First 10 values: {waveform.flatten()[:10].cpu().numpy()}")
        return waveform

    def _get_full_code_sequence(
        self,
        code_tensor: Optional[torch.Tensor],
        kwargs: dict,
    ) -> Optional[torch.Tensor]:
        """Gather the full prompt token ids (code sequence) if available."""
        if code_tensor is not None and code_tensor.numel() > 1:
            return code_tensor

        sampling_metadata = kwargs.get("sampling_metadata")
        prompt_token_ids = getattr(sampling_metadata, "prompt_token_ids", None) if sampling_metadata else None

        if prompt_token_ids is None:
            return code_tensor

        if isinstance(prompt_token_ids, torch.Tensor):
            return prompt_token_ids.detach().to(torch.long).view(-1)

        if isinstance(prompt_token_ids, list):
            if len(prompt_token_ids) == 0:
                return code_tensor
            if isinstance(prompt_token_ids[0], list):
                flat = prompt_token_ids[0]
            else:
                flat = prompt_token_ids
            if len(flat) == 0:
                return code_tensor
            return torch.tensor(flat, dtype=torch.long)

        return code_tensor

    def _echo_code_group(self, code_tensor: torch.Tensor) -> torch.Tensor:
        """Return the input code group without decoding (debug path)."""
        group_width = self.streamer_config.audio_channels + 1
        if group_width <= 0 or code_tensor.numel() < group_width:
            return torch.zeros(0, dtype=torch.float32, device=self.device)

        max_steps = code_tensor.numel() // group_width
        if max_steps == 0:
            return torch.zeros(0, dtype=torch.float32, device=self.device)

        steps_to_use = min(self.streamer_config.group_size, max_steps)
        trimmed = code_tensor[: steps_to_use * group_width]
        groups = trimmed.view(steps_to_use, group_width)
        return groups.to(dtype=torch.float32, device=self.device)

    def _decode_waveform_from_codes(self, code_tensor: torch.Tensor) -> torch.Tensor:
        print("[_decode_waveform_from_codes] Input code_tensor:")
        print(f"  Type: {type(code_tensor)}")
        print(f"  Shape: {code_tensor.shape if code_tensor is not None else None}")
        print(f"  Numel: {code_tensor.numel() if code_tensor is not None else 0}")
        # print("tokens shape =", tokens.shape)
        # print("tokens min =", tokens.min().item())
        # print("tokens max =", tokens.max().item())
        # print("codebook_size =", self.audio_tokenizer.codebook_size)

        if code_tensor is None or code_tensor.numel() == 0:
            print("[_decode_waveform_from_codes] code_tensor is None or empty, returning empty tensor")
            return torch.zeros(0, dtype=torch.float32, device=self.device)

        if code_tensor.ndim > 1:
            if code_tensor.shape[0] == 1:
                code_tensor = code_tensor.squeeze(0)
                print(f"[_decode_waveform_from_codes] Squeezed to shape: {code_tensor.shape}")
            elif code_tensor.shape[0] == self.streamer_config.audio_channels + 1:
                code_tensor = code_tensor.transpose(0, 1).contiguous().view(-1)
                print(f"[_decode_waveform_from_codes] Reshaped to shape: {code_tensor.shape}")
            else:
                raise ValueError(
                    f"code2wav expects shape [L], [1, L] or [audio_channels+1, T], got {code_tensor.shape}"
                )

        flat_codes = code_tensor.detach().to(torch.long).cpu()
        print(
            f"[_decode_waveform_from_codes] flat_codes shape: {flat_codes.shape}, first 20 values: {flat_codes[:20].tolist()}"
        )
        print(
            f"[_decode_waveform_from_codes] streamer_config.group_size: {self.streamer_config.group_size}, audio_channels: {self.streamer_config.audio_channels}"
        )

        audio_codes = extract_audio_code_tensor(
            flat_codes,
            self.streamer_config.group_size,
            self.streamer_config.audio_channels,
            self.codes,
        )
        print("[_decode_waveform_from_codes] audio_codes:")
        print(f"  Type: {type(audio_codes)}")
        print(f"  Shape: {audio_codes.shape if audio_codes is not None else None}")
        print(f"  Numel: {audio_codes.numel() if audio_codes is not None else 0}")

        if audio_codes is None or audio_codes.numel() == 0:
            print("[_decode_waveform_from_codes] audio_codes is None or empty, returning empty tensor")
            return torch.zeros(0, dtype=torch.float32, device=self.device)

        print(
            f"[_decode_waveform_from_codes] Calling tokenizer_service.decode() with audio_codes shape: {audio_codes.shape}"
        )
        print(f"[_decode_waveform_from_codes] audio_codes: {audio_codes}")
        decoded_audio = self._tokenizer_service.decode(audio_codes)
        print("[_decode_waveform_from_codes] decoded_audio:")
        print(f"  Type: {type(decoded_audio)}")
        print(f"  Shape: {decoded_audio.shape if decoded_audio is not None else None}")
        print(f"  Numel: {decoded_audio.numel() if decoded_audio is not None else 0}")

        result = decoded_audio.to(self.device)
        print(f"[_decode_waveform_from_codes] Returning result with shape: {result.shape}, numel: {result.numel()}")
        return result

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        # code2wav 阶段不需要实际的 logits，但需要返回有效的张量
        # 返回一个 dummy logits 张量，形状为 [batch_size, vocab_size]
        if hidden_states.numel() == 0:
            return None

        batch_size = hidden_states.shape[0] if hidden_states.ndim > 0 else 1
        vocab_size = self.config.vocab_size

        # 创建一个全零的 logits 张量（采样器会选择第一个 token，通常是 pad token）
        # 这对于 code2wav 阶段是安全的，因为我们不依赖采样结果
        dummy_logits = torch.zeros(
            (batch_size, vocab_size),
            dtype=hidden_states.dtype,
            device=self.device,
        )
        return dummy_logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        # code2wav 阶段不需要采样，但需要返回有效的 SamplerOutput 避免 CUDA 错误
        # 使用 Sampler 来处理采样，即使结果不会被使用
        if logits is None or logits.numel() == 0:
            return None

        # 使用 Sampler 进行采样（虽然结果不会被使用）
        # 这确保了采样流程能够正常完成，避免 CUDA 错误
        sampler = Sampler()
        sampler_output = sampler(
            logits=logits,
            sampling_metadata=sampling_metadata,
        )
        return sampler_output
