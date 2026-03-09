# Copyright 2025 Xiaomi Corporation.
import math
import threading
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers.activations import ACT2FN
from transformers.modeling_utils import PreTrainedModel
from vllm.logger import init_logger

from .config_mimo_audio import MiMoAudioTokenizerConfig
from .modeling_rope_utils import ROPE_INIT_FUNCTIONS, apply_rotary_pos_emb, dynamic_rope_update
from .quantization import ResidualVectorQuantizer

logger = init_logger(__name__)
is_flash_atth_available = False
try:
    from flash_attn import flash_attn_varlen_func

    is_flash_atth_available = True
except Exception:
    logger.warning("flash_attn not installed")


def get_sequence_mask(inputs, inputs_length):
    if inputs.dim() == 3:
        bsz, tgt_len, _ = inputs.size()
    else:
        bsz, tgt_len = inputs_length.shape[0], torch.max(inputs_length)
    sequence_mask = torch.arange(0, tgt_len).to(inputs.device)
    sequence_mask = torch.lt(sequence_mask, inputs_length.reshape(bsz, 1)).view(bsz, tgt_len, 1)
    unpacking_index = torch.cumsum(sequence_mask.to(torch.int64).view(-1), dim=0) - 1
    return sequence_mask, unpacking_index


def unpack_hidden_states(hidden_states, lengths, sequence_mask=None, unpacking_index=None):
    bsz = lengths.shape[0]
    if sequence_mask is None or unpacking_index is None:
        sequence_mask, unpacking_index = get_sequence_mask(hidden_states, lengths)
    hidden_states = torch.index_select(hidden_states, 0, unpacking_index).view(
        bsz, torch.max(lengths), hidden_states.shape[-1]
    )
    hidden_states = torch.where(sequence_mask, hidden_states, 0)
    return hidden_states


def get_position_ids(lengths):
    total_len = lengths.sum()
    offset = torch.cat([torch.zeros(1).to(lengths), lengths[:-1].cumsum(dim=0)])
    offset = torch.repeat_interleave(offset, lengths)
    position_ids = torch.arange(0, total_len).to(offset) - offset
    return position_ids


@dataclass
class StreamingConfig:
    seg_point: int = field(default=60 * 25)
    process_seg_point: bool = field(default=True)
    left_overlap: int = field(default=10 * 25)
    right_overlap: int = field(default=40)
    seg_point_left_overlap: int = field(default=0)


@dataclass
class StreamingCache:
    hidden_states: list[torch.Tensor] = field(default=None)
    processed_lengths: list[int] = field(default=None)


class ISTFT(nn.Module):
    """
    Custom implementation of ISTFT since torch.istft doesn't allow custom padding (other than `center=True`) with
    windowing. This is because the NOLA (Nonzero Overlap Add) check fails at the edges.
    See issue: https://github.com/pytorch/pytorch/issues/62323
    Specifically, in the context of neural vocoding we are interested in "same" padding analogous to CNNs.
    The NOLA constraint is met as we trim padded samples anyway.

    Args:
        n_fft (int): Size of Fourier transform.
        hop_length (int): The distance between neighboring sliding window frames.
        win_length (int): The size of window frame and STFT filter.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
    """

    def __init__(self, n_fft: int, hop_length: int, win_length: int, padding: str = "same"):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        window = torch.hann_window(win_length)
        self.register_buffer("window", window)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Compute the Inverse Short Time Fourier Transform (ISTFT) of a complex spectrogram.

        Args:
            spec (Tensor): Input complex spectrogram of shape (B, N, T), where B is the batch size,
                            N is the number of frequency bins, and T is the number of time frames.

        Returns:
            Tensor: Reconstructed time-domain signal of shape (B, L), where L is the length of the output signal.
        """
        if self.padding == "center":
            # Fallback to pytorch native implementation
            return torch.istft(
                spec,
                self.n_fft,
                self.hop_length,
                self.win_length,
                self.window,
                center=True,
            )
        elif self.padding == "same":
            pad = (self.win_length - self.hop_length) // 2
        else:
            raise ValueError("Padding must be 'center' or 'same'.")

        assert spec.dim() == 3, "Expected a 3D tensor as input"
        B, N, T = spec.shape

        # Inverse FFT
        ifft = torch.fft.irfft(spec, self.n_fft, dim=1, norm="backward")
        ifft = ifft * self.window[None, :, None]

        # Overlap and Add
        output_size = (T - 1) * self.hop_length + self.win_length
        y = torch.nn.functional.fold(
            ifft,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        )[:, 0, 0, pad:-pad]

        # Window envelope
        window_sq = self.window.square().expand(1, T, -1).transpose(1, 2)
        window_envelope = torch.nn.functional.fold(
            window_sq,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        ).squeeze()[pad:-pad]

        # Normalize
        y = y / window_envelope.clamp(min=1e-11)

        return y


class ISTFTHead(nn.Module):
    """
    ISTFT Head module for predicting STFT complex coefficients.

    Args:
        dim (int): Hidden dimension of the model.
        n_fft (int): Size of Fourier transform.
        hop_length (int): The distance between neighboring sliding window frames, which should align with
                          the resolution of the input features.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
    """

    def __init__(self, dim: int, n_fft: int, hop_length: int, padding: str = "same"):
        super().__init__()
        out_dim = n_fft + 2
        self.out = torch.nn.Linear(dim, out_dim)
        self.istft = ISTFT(n_fft=n_fft, hop_length=hop_length, win_length=n_fft, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ISTFTHead module.

        Args:
            x (Tensor): Input tensor of shape (B, L, H), where B is the batch size,
                        L is the sequence length, and H denotes the model dimension.

        Returns:
            Tensor: Reconstructed time-domain audio signal of shape (B, T), where T is the length of the output signal.
        """
        x = self.out(x).transpose(1, 2)
        mag, p = x.chunk(2, dim=1)
        mag = torch.exp(mag)
        mag = torch.clip(mag, max=1e2)  # safeguard to prevent excessively large magnitudes
        # wrapping happens here. These two lines produce real and imaginary value
        x = torch.cos(p)
        y = torch.sin(p)
        # recalculating phase here does not produce anything new
        # only costs time
        # phase = torch.atan2(y, x)
        # S = mag * torch.exp(phase * 1j)
        # better directly produce the complex value
        original_dtype = x.dtype
        S = mag.float() * (x.float() + 1j * y.float())
        audio = self.istft(S)
        audio = audio.to(original_dtype)
        return audio


class RotaryEmbedding(nn.Module):
    def __init__(self, base, dim, max_seq_len, rope_type="default", device=None):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.rope_type = rope_type

        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(device=device, base=base, dim=dim)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[:, None].float().expand(-1, 1).to(x.device)
        position_ids_expanded = position_ids[None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(0, 1)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


LAYER_NORM = {"LayerNorm": nn.LayerNorm, "RMSNorm": RMSNorm}


class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size=(-1, -1), causal=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.causal = causal

    def forward(
        self,
        hidden_states: torch.Tensor,
        seq_len: torch.Tensor,
        rope_position_embeddings=None,
    ):
        total_seq_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(total_seq_len, self.num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).view(total_seq_len, self.num_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(total_seq_len, self.num_heads, self.head_dim)

        if rope_position_embeddings is not None:
            cos, sin = rope_position_embeddings
            query_states = apply_rotary_pos_emb(query_states, cos, sin)
            key_states = apply_rotary_pos_emb(key_states, cos, sin)

        if hidden_states.is_cuda and is_flash_atth_available is True:
            # === Use flash-attn in GPU mode ===
            cu_len = F.pad(torch.cumsum(seq_len, dim=0), (1, 0), "constant", 0).to(torch.int32)
            max_seqlen = torch.max(seq_len).to(torch.int32).detach()
            attn_output = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_len,
                cu_len,
                max_seqlen,
                max_seqlen,
                causal=self.causal,
                window_size=self.window_size,
            )
            attn_output = attn_output.reshape(total_seq_len, self.embed_dim)

        else:
            # === Fallback implementation in CPU / Eager mode ===
            cu_len = F.pad(torch.cumsum(seq_len, dim=0), (1, 0), "constant", 0).to(torch.long)
            attn_output = torch.zeros_like(hidden_states)

            for i, slen in enumerate(seq_len.tolist()):
                start_idx = cu_len[i].item()
                end_idx = cu_len[i + 1].item()

                q = query_states[start_idx:end_idx]  # [slen, num_heads, head_dim]
                k = key_states[start_idx:end_idx]
                v = value_states[start_idx:end_idx]

                q = q.transpose(0, 1)  # [num_heads, slen, head_dim]
                k = k.transpose(0, 1)
                v = v.transpose(0, 1)

                attn_scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim**0.5)
                if self.causal:
                    mask = torch.tril(torch.ones_like(attn_scores))
                    attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))
                attn_probs = F.softmax(attn_scores, dim=-1)
                attn_out = torch.matmul(attn_probs, v)  # [num_heads, slen, head_dim]

                attn_out = attn_out.transpose(0, 1).reshape(slen, self.embed_dim)
                attn_output[start_idx:end_idx] = attn_out

        attn_output = self.out_proj(attn_output)
        return attn_output

    @property
    def needs_window_mask(self) -> bool:
        left_win, right_win = self.window_size
        return (left_win >= 0) or (right_win >= 0)

    @staticmethod
    def build_window_mask(
        seq_len: int,
        window_size: tuple[int, int],
        causal: bool,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Build a [1, 1, L, L] bool attention mask (True = attend, False = mask out)."""
        left_win, right_win = window_size
        need_window = (left_win >= 0) or (right_win >= 0)
        if not need_window and not causal:
            return None

        row = torch.arange(seq_len, device=device)
        col = torch.arange(seq_len, device=device)
        diff = row[:, None] - col[None, :]

        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
        if causal:
            mask = mask & (diff >= 0)
        if left_win >= 0:
            mask = mask & (diff <= left_win)
        if right_win >= 0:
            mask = mask & (-diff <= right_win)

        return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, L, L]

    def forward_fixed_length(
        self,
        hidden_states: torch.Tensor,
        rope_position_embeddings=None,
        attn_mask: torch.Tensor | None = None,
    ):
        """
        CUDA Graph compatible forward with fixed-shape 3D input [B, L, D].
        Uses F.scaled_dot_product_attention instead of flash_attn_varlen_func
        to avoid dynamic shapes.
        """
        bsz, seq_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim)

        if rope_position_embeddings is not None:
            cos, sin = rope_position_embeddings
            query_states = apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2)
            key_states = apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2)

        query_states = query_states.transpose(1, 2)  # [B, num_heads, L, head_dim]
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        use_is_causal = attn_mask is None and self.causal and not self.needs_window_mask
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attn_mask,
            is_causal=use_is_causal,
        )

        attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output


class TransformerLayer(nn.Module):
    def __init__(
        self,
        act,
        d_model,
        encoder_attention_heads,
        encoder_ffn_dim,
        causal,
        ln_type="LayerNorm",
        attn_window_size=(-1, -1),
    ):
        super().__init__()
        self.embed_dim = d_model
        self.self_attn = Attention(self.embed_dim, encoder_attention_heads, attn_window_size, causal)

        self.self_attn_layer_norm = LAYER_NORM[ln_type](self.embed_dim)

        self.activation_fn = act
        self.fc1 = nn.Linear(self.embed_dim, encoder_ffn_dim)
        self.fc2 = nn.Linear(encoder_ffn_dim, self.embed_dim)

        self.final_layer_norm = LAYER_NORM[ln_type](self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        seq_len: torch.Tensor,
        rope_position_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, seq_len, rope_position_embeddings=rope_position_embeddings)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        if (hidden_states.dtype == torch.float16 or hidden_states.dtype == torch.bfloat16) and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        return hidden_states

    def forward_fixed_length(
        self,
        hidden_states: torch.Tensor,
        rope_position_embeddings: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """CUDA Graph compatible forward with fixed-shape 3D input [B, L, D]."""
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn.forward_fixed_length(
            hidden_states, rope_position_embeddings=rope_position_embeddings, attn_mask=attn_mask
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        # Unconditional nan_to_num + clamp avoids data-dependent branching
        # (.any()) which would break CUDA Graph capture.
        if hidden_states.dtype == torch.float16 or hidden_states.dtype == torch.bfloat16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.nan_to_num(hidden_states, nan=0.0, posinf=clamp_value, neginf=-clamp_value)
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        return hidden_states


class TransformerVocos(nn.Module):
    def __init__(self, config: MiMoAudioTokenizerConfig):
        super().__init__()
        self.config = config
        self.max_source_positions = self.config.max_audio_seconds * self.config.sampling_rate // self.config.hop_length
        self.embeddings = nn.Linear(config.n_mels, config.vocoder_dim, bias=False)

        self.position_embedding = RotaryEmbedding(
            config.rope_theta,
            config.vocoder_dim // config.vocoder_attention_heads,
            self.max_source_positions,
            self.config.rope_type,
        )

        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    ACT2FN[self.config.activation_function],
                    self.config.vocoder_dim,
                    self.config.vocoder_attention_heads,
                    self.config.vocoder_intermediate_dim,
                    causal=False,
                    ln_type=self.config.ln_type,
                    attn_window_size=self.config.vocoder_attn_window_size,
                )
                for _ in range(self.config.vocoder_num_layers)
            ]
        )

        self.layer_norm = LAYER_NORM[self.config.ln_type](self.config.vocoder_dim)
        self.hop_size = self.config.hop_length
        self.head = ISTFTHead(
            self.config.vocoder_dim,
            self.config.nfft,
            self.config.hop_length,
            self.config.vocoder_padding,
        )

    def forward(self, x: torch.Tensor, input_length):
        x = x.transpose(1, 2)
        attention_mask, unpacking_index = get_sequence_mask(x, input_length)
        x = torch.masked_select(x, attention_mask).view(torch.sum(input_length), self.config.n_mels)
        x = self.embeddings(x)
        position_ids = torch.arange(0, x.size(0), device=x.device, dtype=torch.long)
        rope_position_embeddings = self.position_embedding(x, position_ids)
        for idx, layer in enumerate(self.layers):
            x = layer(x, input_length, rope_position_embeddings=rope_position_embeddings)

        x = self.layer_norm(x)
        x = unpack_hidden_states(x, input_length, attention_mask, unpacking_index)
        x = self.head(x)
        output_length = input_length * self.hop_size
        return x[:, None, :], output_length

    def _get_or_create_window_mask(self, seq_len: int, device: torch.device) -> torch.Tensor | None:
        """Get cached bool window mask or create one.  Must be called OUTSIDE
        ``torch.cuda.graph()`` capture context (i.e. during warmup) so that the
        tensor already exists when the graph is recorded."""
        first_layer_attn = self.layers[0].self_attn
        if not first_layer_attn.needs_window_mask:
            return None
        cache_attr = "_vocoder_window_mask_cache"
        if not hasattr(self, cache_attr):
            setattr(self, cache_attr, {})
        cache: dict = getattr(self, cache_attr)
        if seq_len in cache:
            return cache[seq_len]
        mask = Attention.build_window_mask(
            seq_len,
            first_layer_attn.window_size,
            first_layer_attn.causal,
            device,
        )
        cache[seq_len] = mask
        return mask

    def forward_fixed_length(self, x: torch.Tensor):
        """
        CUDA Graph compatible forward with fixed-shape 3D input [B, n_mels, L].
        Operates entirely with fixed-shape tensors (no masked_select / dynamic pack).
        """
        x = x.transpose(1, 2)  # [B, L, n_mels]
        bsz, seq_len, _ = x.size()
        x = self.embeddings(x)  # [B, L, vocoder_dim]
        position_ids = torch.arange(0, seq_len, device=x.device, dtype=torch.long)
        rope_position_embeddings = self.position_embedding(x, position_ids)
        rope_cos, rope_sin = rope_position_embeddings
        rope_position_embeddings = (rope_cos.unsqueeze(0), rope_sin.unsqueeze(0))

        attn_mask = self._get_or_create_window_mask(seq_len, x.device)
        for layer in self.layers:
            x = layer.forward_fixed_length(x, rope_position_embeddings=rope_position_embeddings, attn_mask=attn_mask)

        x = self.layer_norm(x)
        x = self.head(x)
        return x[:, None, :]


class AudioEncoder(nn.Module):
    def __init__(self, config: MiMoAudioTokenizerConfig):
        super().__init__()
        config._attn_implementation = "flash_attention_2"
        self.config = config
        self.max_source_positions = (
            config.max_audio_seconds * config.sampling_rate // config.hop_length
        ) // config.stride_size
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        self.skip_layer_idx = config.encoder_skip_layer_id
        self.conv1 = nn.Conv1d(config.n_mels, config.d_model, kernel_size=config.kernel_size, padding=1)
        self.conv2 = nn.Conv1d(
            config.d_model,
            config.d_model,
            kernel_size=config.kernel_size,
            stride=config.stride_size,
            padding=1,
        )

        self.position_embedding = RotaryEmbedding(
            config.rope_theta,
            config.d_model // config.encoder_attention_heads,
            self.max_source_positions,
            config.rope_type,
        )

        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    ACT2FN[config.activation_function],
                    config.d_model,
                    config.encoder_attention_heads,
                    config.encoder_ffn_dim,
                    causal=self.config.encoder_causal,
                    ln_type=self.config.ln_type,
                    attn_window_size=self.config.encoder_attn_window_size,
                )
                for _ in range(config.encoder_layers)
            ]
        )

        self.layer_norm = LAYER_NORM[config.ln_type](config.d_model)

        if self.config.avg_pooler != 1:
            self.down_sample_layer = nn.Sequential(
                nn.Conv1d(
                    config.d_model,
                    config.d_model,
                    config.avg_pooler,
                    config.avg_pooler,
                    bias=False,
                ),
                nn.GELU(),
            )
            self.down_sample_norm = LAYER_NORM[config.ln_type](config.d_model)
        else:
            self.down_sample_layer = None

        if self.config.num_quantizers != 0:
            self.quantizer = ResidualVectorQuantizer(
                dimension=self.config.d_model,
                n_q=self.config.num_quantizers,
                bins=self.config.codebook_size,
                threshold_ema_dead_code=self.config.threshold_ema_dead_code,
            )
        else:
            self.quantizer = None

    def get_features(self, input_features, output_length):
        input_features = input_features.to(self.conv1.weight)
        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))
        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        bsz, tgt_len, _ = inputs_embeds.size()

        hidden_states = inputs_embeds

        position_ids = get_position_ids(output_length).long().to(input_features.device)
        rope_position_embeddings = self.position_embedding(input_features, position_ids)

        attention_mask, unpacking_index = get_sequence_mask(hidden_states, output_length)

        hidden_states = torch.masked_select(hidden_states, attention_mask).view(
            torch.sum(output_length), self.config.d_model
        )

        skip_connect_hidden_states = 0.0
        for idx, encoder_layer in enumerate(self.layers):
            hidden_states = encoder_layer(
                hidden_states,
                output_length,
                rope_position_embeddings=rope_position_embeddings,
            )
            if (self.skip_layer_idx is not None) and idx == self.skip_layer_idx - 1:
                skip_connect_hidden_states = hidden_states.clone()

        hidden_states += skip_connect_hidden_states
        hidden_states = self.layer_norm(hidden_states)

        if self.down_sample_layer is not None:
            hidden_states = torch.index_select(hidden_states, 0, unpacking_index).view(
                bsz, tgt_len, self.config.d_model
            )
            if hidden_states.size(1) % self.config.avg_pooler:
                pad_len = self.config.avg_pooler - hidden_states.size(1) % self.config.avg_pooler
                hidden_states = torch.nn.functional.pad(hidden_states, (0, 0, 0, pad_len), mode="constant", value=0.0)
                tgt_len += pad_len
            tgt_len = tgt_len // self.config.avg_pooler
            hidden_states = self.down_sample_layer(hidden_states.transpose(1, 2))
            output_length = (
                output_length // self.config.avg_pooler + (output_length % self.config.avg_pooler != 0).int()
            )
            hidden_states = hidden_states.transpose(1, 2)
            attention_mask, unpacking_index = get_sequence_mask(hidden_states, output_length)
            hidden_states = torch.masked_select(hidden_states, attention_mask).view(
                torch.sum(output_length), self.config.d_model
            )
            hidden_states = self.down_sample_norm(hidden_states)

        return (
            hidden_states,
            output_length,
            attention_mask,
            unpacking_index,
            tgt_len,
            bsz,
        )

    def get_output_length(self, mel_len):
        tgt_len = mel_len + 3 - self.config.kernel_size
        return (tgt_len + 2 - self.config.kernel_size) // self.config.stride_size + 1

    @torch.no_grad()
    def encode(
        self,
        input_features,
        input_lens=None,
        output_length=None,
        return_codes_only=False,
        n_q=None,
        use_quantizer=True,
    ):
        if output_length is None:
            output_length = self.get_output_length(input_lens)
        input_features = unpack_hidden_states(input_features, input_lens)
        hidden_states, output_length, attention_mask, unpacking_index, tgt_len, bsz = self.get_features(
            input_features=input_features.transpose(1, 2),
            output_length=output_length,
        )

        dtype = hidden_states.dtype

        if use_quantizer and self.quantizer is not None:
            self.quantizer.float()

            codes = self.quantizer.encode(hidden_states.float(), n_q=n_q)
            if return_codes_only:
                return codes, output_length
            hidden_states = self.quantizer.decode(codes)
            hidden_states = hidden_states.to(dtype)
        else:
            codes = None

        hidden_states_packed = hidden_states.clone()

        # unpacking
        hidden_states = torch.index_select(hidden_states, 0, unpacking_index).view(bsz, tgt_len, self.config.d_model)
        hidden_states = torch.where(attention_mask, hidden_states, 0)
        return hidden_states, hidden_states_packed, output_length, codes

    @torch.no_grad()
    def decode_vq(self, codes):
        self.quantizer.float()
        hidden_states = self.quantizer.decode(codes)

        return hidden_states


class CausalConvTranspose1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride)
        self.norm = nn.GroupNorm(1, out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, hidden_states, input_length, output_dim=None):
        kernel_size = self.conv.kernel_size[0]
        stride = self.conv.stride[0]
        bsz = input_length.shape[0]

        if output_dim is None:
            output_dim = hidden_states.dim()
        if hidden_states.dim() <= 2:  # unpack sequence to 3d
            sequence_mask, unpacking_index = get_sequence_mask(hidden_states, input_length)
            hidden_states = torch.index_select(hidden_states, 0, unpacking_index).view(
                bsz, torch.max(input_length), self.in_channels
            )
            hidden_states = torch.where(sequence_mask, hidden_states, 0)

        hidden_states = hidden_states.transpose(2, 1)  # (N, L, C) -> (N, C, L)
        hidden_states = self.conv(hidden_states)
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.transpose(2, 1)  # (N, C, L) -> (N, L, C)

        casual_padding_right = max(0, kernel_size - stride)
        hidden_states = hidden_states[:, : hidden_states.shape[1] - casual_padding_right, :]
        output_length = (input_length - 1) * stride + kernel_size - casual_padding_right
        sequence_mask, _ = get_sequence_mask(hidden_states, output_length)
        if output_dim <= 2:
            hidden_states = torch.masked_select(hidden_states, sequence_mask).view(-1, self.out_channels)
        else:
            hidden_states = torch.where(sequence_mask, hidden_states, 0)
            hidden_states = hidden_states[:, : torch.max(output_length), :]
        return hidden_states, output_length

    def forward_fixed_length(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        CUDA Graph compatible forward with fixed-shape 3D input [B, L, C].
        Returns [B, L_out, C_out] where L_out is deterministic from L.
        """
        kernel_size = self.conv.kernel_size[0]
        stride = self.conv.stride[0]

        hidden_states = hidden_states.transpose(2, 1)  # (B, L, C) -> (B, C, L)
        hidden_states = self.conv(hidden_states)
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.transpose(2, 1)  # (B, C, L) -> (B, L, C)

        casual_padding_right = max(0, kernel_size - stride)
        if casual_padding_right > 0:
            hidden_states = hidden_states[:, : hidden_states.shape[1] - casual_padding_right, :]
        return hidden_states


class AudioDecoder(nn.Module):
    def __init__(self, config: MiMoAudioTokenizerConfig):
        super().__init__()
        self.config = config
        self.max_source_positions = self.config.max_audio_seconds * self.config.sampling_rate // self.config.hop_length

        if self.config.avg_pooler != 1:
            self.dconv1 = CausalConvTranspose1d(
                self.config.d_model,
                self.config.d_model,
                self.config.avg_pooler,
                self.config.avg_pooler,
            )
        else:
            self.dconv1 = None

        self.position_embedding = RotaryEmbedding(
            config.rope_theta,
            config.d_model // config.decoder_attention_heads,
            self.max_source_positions,
            config.rope_type,
        )

        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    ACT2FN[self.config.activation_function],
                    self.config.d_model,
                    self.config.decoder_attention_heads,
                    self.config.decoder_ffn_dim,
                    causal=self.config.decoder_causal,
                    ln_type=self.config.ln_type,
                    attn_window_size=self.config.decoder_attn_window_size,
                )
                for _ in range(self.config.decoder_layers)
            ]
        )
        self.layer_norm = LAYER_NORM[config.ln_type](self.config.d_model)
        self.dconv2 = CausalConvTranspose1d(
            self.config.d_model,
            self.config.n_mels,
            self.config.decoder_kernel_size,
            self.config.decoder_stride_size,
        )
        self.vocoder = TransformerVocos(config)

    def forward(
        self,
        audio_embed,
        input_length,
    ):
        assert audio_embed.shape[-1] == self.config.d_model
        audio_embed = audio_embed.to(self.layer_norm.weight)

        if self.dconv1 is not None:
            audio_embed, output_length = self.dconv1(audio_embed, input_length, output_dim=3)
        else:
            output_length = input_length

        hidden_states = audio_embed

        position_ids = get_position_ids(output_length).long().to(hidden_states.device)
        rope_position_embeddings = self.position_embedding(hidden_states, position_ids)

        # packing hidden states
        attention_mask, _ = get_sequence_mask(hidden_states, output_length)
        hidden_states = torch.masked_select(hidden_states, attention_mask).view(
            torch.sum(output_length), self.config.d_model
        )

        for idx, encoder_layer in enumerate(self.layers):
            hidden_states = encoder_layer(
                hidden_states,
                output_length,
                rope_position_embeddings=rope_position_embeddings,
            )

        hidden_states = self.layer_norm(hidden_states)

        coarse_mel, output_length = self.dconv2(hidden_states, output_length, output_dim=3)

        recon_wav, wav_length = self.vocoder(
            x=coarse_mel.transpose(1, 2),
            input_length=output_length,
        )

        return recon_wav

    def _get_or_create_decoder_mask(self, seq_len: int, device: torch.device) -> torch.Tensor | None:
        """Get cached bool decoder attention mask or create one."""
        first_layer_attn = self.layers[0].self_attn
        if not first_layer_attn.needs_window_mask and not first_layer_attn.causal:
            return None
        # Decoder with causal=True but window_size=(-1,-1): use is_causal flag instead
        if not first_layer_attn.needs_window_mask and first_layer_attn.causal:
            return None
        cache_attr = "_decoder_window_mask_cache"
        if not hasattr(self, cache_attr):
            setattr(self, cache_attr, {})
        cache: dict = getattr(self, cache_attr)
        if seq_len in cache:
            return cache[seq_len]
        mask = Attention.build_window_mask(
            seq_len,
            first_layer_attn.window_size,
            first_layer_attn.causal,
            device,
        )
        cache[seq_len] = mask
        return mask

    def forward_fixed_length(
        self,
        audio_embed: torch.Tensor,
    ) -> torch.Tensor:
        """
        CUDA Graph compatible forward with fixed-shape 3D input [B, L, d_model].
        Avoids all dynamic shape operations (masked_select, get_sequence_mask, etc.).
        """
        assert audio_embed.shape[-1] == self.config.d_model
        audio_embed = audio_embed.to(self.layer_norm.weight)

        if audio_embed.dim() == 2:
            audio_embed = audio_embed.unsqueeze(0)

        if self.dconv1 is not None:
            audio_embed = self.dconv1.forward_fixed_length(audio_embed)

        hidden_states = audio_embed
        bsz, seq_len, _ = hidden_states.size()

        position_ids = torch.arange(0, seq_len, device=hidden_states.device, dtype=torch.long)
        rope_position_embeddings = self.position_embedding(hidden_states, position_ids)
        rope_cos, rope_sin = rope_position_embeddings
        rope_position_embeddings = (rope_cos.unsqueeze(0), rope_sin.unsqueeze(0))

        decoder_mask = self._get_or_create_decoder_mask(seq_len, hidden_states.device)
        for encoder_layer in self.layers:
            hidden_states = encoder_layer.forward_fixed_length(
                hidden_states,
                rope_position_embeddings=rope_position_embeddings,
                attn_mask=decoder_mask,
            )

        hidden_states = self.layer_norm(hidden_states)

        coarse_mel = self.dconv2.forward_fixed_length(hidden_states)

        recon_wav = self.vocoder.forward_fixed_length(coarse_mel.transpose(1, 2))

        return recon_wav


class MiMoAudioTokenizer(PreTrainedModel):
    config_class = MiMoAudioTokenizerConfig

    def __init__(self, config: MiMoAudioTokenizerConfig):
        super().__init__(config)
        self.config = config
        self.sampling_rate = config.sampling_rate
        self.encoder = AudioEncoder(config=config)
        self.decoder = AudioDecoder(config=config)
        self.downsample_rate = int(self.config.hop_length * 2 * self.config.avg_pooler)

    def get_output_length(self, mel_len):
        tgt_len = mel_len + 3 - self.config.kernel_size
        return (tgt_len + 2 - self.config.kernel_size) // self.config.stride_size + 1

    @torch.no_grad()
    def encode(self, mels, input_lens, use_quantizer=True):
        input_features = mels
        hidden_states, hidden_states_packed, encoder_output_length, codes = self.encoder.encode(
            input_features, input_lens=input_lens, use_quantizer=use_quantizer
        )
        return hidden_states, hidden_states_packed, encoder_output_length, codes

    @torch.no_grad()
    def decode(self, codes):
        hidden_states = self.encoder.decode_vq(codes)
        output = self.decoder(
            hidden_states,
            torch.tensor([hidden_states.size(0)], device=hidden_states.device),
        )
        return output

    @torch.no_grad()
    def decode_fixed_length(self, codes: torch.Tensor) -> torch.Tensor:
        """
        CUDA Graph compatible decode: VQ decode + AudioDecoder fixed-length forward.
        Input codes shape: [n_q, seq_len].
        The quantizer.decode is a pure embedding lookup (no dynamic shapes),
        and the decoder uses fixed-length forward to avoid shape-dependent ops.
        """
        hidden_states = self.encoder.quantizer.decode(codes)
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)  # [1, seq_len, d_model]
        output = self.decoder.forward_fixed_length(hidden_states)
        return output

    @torch.no_grad()
    def streaming_decode(
        self,
        codes_chunks,
        chunk_input_lengths,
        history_cache=StreamingCache(),
        streaming_config=StreamingConfig(),
        last_chunk=False,
    ):
        hidden_states = self.encoder.decode_vq(codes_chunks)
        input_lengths = []
        input_hidden_states = []
        start_idx = 0
        cache_hidden_states = []
        for i, input_length in enumerate(chunk_input_lengths):
            sample_hidden_states = hidden_states[start_idx : start_idx + input_length]
            start_idx += input_length
            if history_cache.hidden_states is not None:
                sample_hidden_states = torch.cat([history_cache.hidden_states[i], sample_hidden_states], dim=0)
                input_length += history_cache.hidden_states[i].size(0)
            input_hidden_states.append(sample_hidden_states)
            cache_hidden_states.append(sample_hidden_states.clone())
            input_lengths.append(input_length)
        input_hidden_states = torch.cat(input_hidden_states, dim=0)
        input_lengths = torch.tensor(input_lengths, device=hidden_states.device)
        output = self.decoder(input_hidden_states, input_lengths)
        return_wavs = []
        frames_per_token = self.config.avg_pooler * self.config.stride_size * self.config.hop_length
        processed_lengths = []
        for i, wav in enumerate(output):
            wav = wav.float().detach().cpu()
            start_idx = history_cache.processed_lengths[i] if history_cache.processed_lengths is not None else 0
            if last_chunk:
                return_wavs.append(wav[:, start_idx * frames_per_token :])
                new_processed_length = input_lengths[i].item()
            elif input_lengths[i].item() <= streaming_config.right_overlap:
                return_wavs.append(None)
                new_processed_length = 0
            else:
                end_idx = input_lengths[i].item() - streaming_config.right_overlap
                wav = wav[:, start_idx * frames_per_token : end_idx * frames_per_token]
                return_wavs.append(wav)
                new_processed_length = end_idx
                if input_lengths[i].item() > streaming_config.left_overlap:
                    cache_hidden_states[i] = cache_hidden_states[i][-streaming_config.left_overlap :]
                    new_processed_length -= input_lengths[i].item() - streaming_config.left_overlap
            processed_lengths.append(new_processed_length)
        history_cache.hidden_states = cache_hidden_states
        history_cache.processed_lengths = processed_lengths

        return return_wavs, history_cache


class CUDAGraphAudioTokenizerWrapper:
    """
    CUDA Graph wrapper for MiMoAudioTokenizer decode path.

    Captures the entire decode pipeline (VQ decode -> AudioDecoder -> Vocoder)
    into CUDA Graphs for different pre-defined input sizes (bucket sizes).
    During inference, input codes are padded to the nearest bucket size and
    the corresponding graph is replayed, eliminating kernel launch overhead.

    This wrapper only accelerates the decode path since it is the hot path
    during streaming inference (called per chunk), while encode is called
    only once per input.
    """

    DEFAULT_CAPTURE_SIZES = [200]  # DEBUG: single size to isolate graph issue

    def __init__(
        self,
        tokenizer: MiMoAudioTokenizer,
        capture_sizes: list[int] | None = None,
        enabled: bool = True,
    ):
        self.tokenizer = tokenizer
        self.capture_sizes = sorted(capture_sizes or self.DEFAULT_CAPTURE_SIZES)
        self.enabled = enabled
        self.num_quantizers = tokenizer.config.num_quantizers

        self._graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._static_inputs: dict[int, torch.Tensor] = {}
        self._static_outputs: dict[int, torch.Tensor] = {}

        self._warmed_up = False
        self._device: torch.device | None = None
        self._lock = threading.Lock()
        self._graph_pool_handles: dict[int, int] = {}

    def _get_padded_size(self, actual_size: int) -> int | None:
        for size in self.capture_sizes:
            if actual_size <= size:
                return size
        return None

    @torch.no_grad()
    def warmup(self, device: torch.device, dtype: torch.dtype = torch.long):
        if device.type != "cuda":
            logger.info("CUDAGraph warmup skipped: device %s is not CUDA", device)
            return

        if not self.enabled:
            logger.info("CUDAGraph is disabled for audio tokenizer, skipping warmup")
            return

        if self._warmed_up:
            logger.warning("CUDAGraph already warmed up for audio tokenizer, skipping")
            return

        self._device = device
        self.tokenizer.eval()

        # Pre-convert quantizer to float32 so it stays constant during graph capture
        if self.tokenizer.encoder.quantizer is not None:
            self.tokenizer.encoder.quantizer.float()

        logger.info(
            "Starting CUDAGraph warmup for audio tokenizer decode, %d sizes: %s",
            len(self.capture_sizes),
            self.capture_sizes,
        )

        for size in self.capture_sizes:
            dummy_codes = torch.zeros(
                self.num_quantizers,
                size,
                dtype=dtype,
                device=device,
            )
            with torch.no_grad():
                self.tokenizer.decode_fixed_length(dummy_codes)

        torch.cuda.synchronize(device)

        for size in self.capture_sizes:
            try:
                self._capture_graph(size, device, dtype)
                logger.info("  Captured CUDAGraph for audio tokenizer decode, size=%d", size)
            except Exception:
                logger.warning(
                    "  Failed to capture CUDAGraph for audio tokenizer decode, size=%d",
                    size,
                    exc_info=True,
                )

        self._warmed_up = True
        logger.info(
            "CUDAGraph warmup complete for audio tokenizer. Captured %d graphs.",
            len(self._graphs),
        )

    def _capture_graph(self, size: int, device: torch.device, dtype: torch.dtype):
        static_input = torch.zeros(
            self.num_quantizers,
            size,
            dtype=dtype,
            device=device,
        )

        # Multiple warmup runs to ensure all lazy initializations (mask caches,
        # cuDNN/cuBLAS plans, SDPA backend selection, etc.) complete before
        # the graph capture region.  This is critical: CUDA Graph records a
        # fixed set of kernel launches; any first-time initialisation that
        # happens *inside* the capture will either fail or produce a broken graph.
        for _ in range(3):
            with torch.no_grad():
                _ = self.tokenizer.decode_fixed_length(static_input)
        torch.cuda.synchronize(device)

        pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        with torch.no_grad():
            with torch.cuda.graph(graph, pool=pool):
                static_output = self.tokenizer.decode_fixed_length(static_input)

        # Verify: replay immediately after capture with same zero input
        graph.replay()
        torch.cuda.synchronize(device)
        eager_ref = self.tokenizer.decode_fixed_length(static_input)
        torch.cuda.synchronize(device)
        cg_flat = static_output.float().reshape(-1)
        ea_flat = eager_ref.float().reshape(-1)
        min_l = min(cg_flat.shape[0], ea_flat.shape[0])
        diff = (cg_flat[:min_l] - ea_flat[:min_l]).abs()
        logger.info(
            "[CAPTURE VERIFY] size=%d max_diff=%.6f mean_diff=%.6f cg_max=%.4f ea_max=%.4f",
            size,
            diff.max().item(),
            diff.mean().item(),
            cg_flat.abs().max().item(),
            ea_flat.abs().max().item(),
        )

        self._graphs[size] = graph
        self._static_inputs[size] = static_input
        self._static_outputs[size] = static_output
        self._graph_pool_handles[size] = pool

    def _compute_output_token_to_samples(self, num_codes: int) -> int:
        """Compute the output waveform length for a given number of input code tokens."""
        cfg = self.tokenizer.config
        length = num_codes

        # dconv1: ConvTranspose1d(d_model, d_model, avg_pooler, avg_pooler)
        if cfg.avg_pooler != 1:
            k1 = cfg.avg_pooler
            s1 = cfg.avg_pooler
            length = (length - 1) * s1 + k1
            casual_pad_1 = max(0, k1 - s1)
            length -= casual_pad_1

        # dconv2: ConvTranspose1d(d_model, n_mels, decoder_kernel_size, decoder_stride_size)
        k2 = cfg.decoder_kernel_size
        s2 = cfg.decoder_stride_size
        length = (length - 1) * s2 + k2
        casual_pad_2 = max(0, k2 - s2)
        length -= casual_pad_2

        # vocoder ISTFTHead: each mel frame → hop_length samples
        length *= cfg.hop_length
        return length

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Decode audio codes to waveform, using CUDA Graph when possible.
        Input: codes of shape [n_q, seq_len] (or [seq_len] will be handled by the caller).
        Output: waveform tensor.
        """
        if not self.enabled or not self._warmed_up:
            return self.tokenizer.decode(codes)

        actual_size = codes.shape[-1]
        padded_size = self._get_padded_size(actual_size)

        if padded_size is None or padded_size not in self._graphs:
            return self.tokenizer.decode(codes)

        with self._lock:
            self._static_inputs[padded_size][:, :actual_size] = codes
            if actual_size < padded_size:
                self._static_inputs[padded_size][:, actual_size:] = codes[:, -1:]

            self._graphs[padded_size].replay()

            output = self._static_outputs[padded_size]
            actual_wav_len = self._compute_output_token_to_samples(actual_size)
            padded_wav_len = self._compute_output_token_to_samples(padded_size)

            # AIGC START
            cg_full = output.clone().float().reshape(-1)
            cg_wav = cg_full[:actual_wav_len] if actual_wav_len < cg_full.shape[0] else cg_full

            padded_input = self._static_inputs[padded_size].clone()
            eager_fl_padded = self.tokenizer.decode_fixed_length(padded_input)
            eager_fl_padded_full = eager_fl_padded.float().reshape(-1)
            eager_fl_padded_wav = (
                eager_fl_padded_full[:actual_wav_len]
                if actual_wav_len < eager_fl_padded_full.shape[0]
                else eager_fl_padded_full
            )

            eager_fl_unpadded = self.tokenizer.decode_fixed_length(codes)
            eager_fl_unpadded_wav = eager_fl_unpadded.float().reshape(-1)

            ref_output = self.tokenizer.decode(codes)
            ref_wav = ref_output.float().reshape(-1)

            min_len = min(
                ref_wav.shape[0], cg_wav.shape[0], eager_fl_padded_wav.shape[0], eager_fl_unpadded_wav.shape[0]
            )

            diff_cg_eager_padded = (cg_wav[:min_len] - eager_fl_padded_wav[:min_len]).abs()
            diff_eager_padded_ref = (eager_fl_padded_wav[:min_len] - ref_wav[:min_len]).abs()
            diff_eager_unpadded_ref = (eager_fl_unpadded_wav[:min_len] - ref_wav[:min_len]).abs()
            diff_cg_ref = (cg_wav[:min_len] - ref_wav[:min_len]).abs()

            logger.info(
                "[CG vs EAGER_PADDED] max=%.6f mean=%.6f | "
                "[EAGER_PADDED vs REF] max=%.6f mean=%.6f | "
                "[EAGER_UNPADDED vs REF] max=%.6f mean=%.6f | "
                "[CG vs REF] max=%.6f mean=%.6f | "
                "ref_max=%.4f unpadded_max=%.4f padded_max=%.4f cg_max=%.4f",
                diff_cg_eager_padded.max().item(),
                diff_cg_eager_padded.mean().item(),
                diff_eager_padded_ref.max().item(),
                diff_eager_padded_ref.mean().item(),
                diff_eager_unpadded_ref.max().item(),
                diff_eager_unpadded_ref.mean().item(),
                diff_cg_ref.max().item(),
                diff_cg_ref.mean().item(),
                ref_wav.abs().max().item(),
                eager_fl_unpadded_wav.abs().max().item(),
                eager_fl_padded_wav.abs().max().item(),
                cg_wav.abs().max().item(),
            )
            # AIGC END

            if actual_wav_len < padded_wav_len:
                return output[..., :actual_wav_len].clone()

            return output.clone()

    def streaming_decode(
        self,
        codes: torch.Tensor,
        chunk_size: int = 200,
        left_context_size: int = 25,
    ) -> torch.Tensor:
        """Chunked decode with CUDA Graph acceleration and left-context overlap."""
        wavs = []
        start_index = 0
        total_len = codes.shape[-1]
        samples_per_token = self._compute_output_token_to_samples(1)

        while start_index < total_len:
            end_index = min(start_index + chunk_size, total_len)
            context_size = left_context_size if start_index - left_context_size > 0 else start_index

            codes_chunk = codes[:, start_index - context_size : end_index]
            wav_chunk = self.decode(codes_chunk)

            if wav_chunk.dim() >= 2:
                wav_chunk = wav_chunk.reshape(-1)
            drop_samples = context_size * samples_per_token
            if drop_samples > 0:
                wav_chunk = wav_chunk[drop_samples:]
            wavs.append(wav_chunk)
            start_index = end_index

        return torch.cat(wavs, dim=-1)
