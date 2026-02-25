# Copyright 2025 Xiaomi Corporation.
"""
CUDA Graph wrapper for MiMo Audio Tokenizer decode.

This module provides CUDA Graph acceleration for the speech tokenizer decoder,
reducing kernel launch overhead during inference for both streaming and non-streaming decode.
"""

import torch
from torch.cuda import CUDAGraph
import traceback
from typing import Dict, List, Optional

from vllm.logger import init_logger

logger = init_logger(__name__)


class CUDAGraphDecoderWrapper:
    """
    CUDA Graph wrapper for MiMo Audio Tokenizer decode.

    This wrapper captures the tokenizer.decode forward pass for fixed input sizes
    and replays them during inference to reduce kernel launch overhead.
    """

    # Covers chunk_size=[1, 5, 10, 20]
    DEFAULT_CAPTURE_SIZES = [4, 20, 40, 80, 160] # chunk_size * group_size

    def __init__(
        self,
        tokenizer: torch.nn.Module,
        capture_sizes: Optional[List[int]] = None,
        audio_channels: int = 8,
        total_upsample: int = 960,
        enabled: bool = True,
    ):
        """
        Initialize the CUDA Graph wrapper.

        Args:
            tokenizer: The MiMoAudioTokenizer module (decode method)
            capture_sizes: List of T (sequence length) to capture graphs for
            audio_channels: Number of audio channels
            total_upsample: Samples per code frame for output trim
            enabled: Whether CUDA graph is enabled
        """
        self.tokenizer = tokenizer
        self.capture_sizes = capture_sizes or self.DEFAULT_CAPTURE_SIZES
        self.audio_channels = audio_channels
        self.total_upsample = total_upsample
        self.enabled = enabled

        # CUDA graph storage
        self.graphs: Dict[int, CUDAGraph] = {}
        self.static_inputs: Dict[int, torch.Tensor] = {}
        self.static_outputs: Dict[int, torch.Tensor] = {}
        self.static_input_lengths: Dict[int, torch.Tensor] = {}

        self._warmed_up = False
        self._device = None
        

    def _get_padded_size(self, actual_size: int) -> Optional[int]:
        """
        Get the smallest capture size that can accommodate the actual size.

        Args:
            actual_size: The actual code sequence length (T)

        Returns:
            The padded size, or None if no suitable size exists
        """
        for size in self.capture_sizes:
            if actual_size <= size:
                return size
        return None

    def _compute_graph_context(self, seq_len: int) -> dict:
        """Pre-compute ALL int values needed by the decoder to avoid any
        GPU-tensor-to-Python-int conversion (.item()) during graph capture.
        """
        cfg = self.tokenizer.config
        avg_pooler = getattr(cfg, "avg_pooler", 2)
        decoder_kernel_size = getattr(cfg, "decoder_kernel_size", 3)
        decoder_stride_size = getattr(cfg, "decoder_stride_size", 2)

        if avg_pooler != 1:
            dconv1_output_len = seq_len * avg_pooler
        else:
            dconv1_output_len = seq_len

        casual_padding_right = max(0, decoder_kernel_size - decoder_stride_size)
        dconv2_output_len = (
            (dconv1_output_len - 1) * decoder_stride_size
            + decoder_kernel_size
            - casual_padding_right
        )

        return {
            "vq_seq_len": seq_len,
            "dconv1_output_len": dconv1_output_len,
            "dconv2_output_len": dconv2_output_len,
            "decoder_max_seqlen_int": dconv1_output_len,
            "vocoder_max_seqlen_int": dconv2_output_len,
        }

    def warmup(self, device: torch.device, dtype: torch.dtype = torch.long):
        if not self.enabled:
            logger.info("[MiMo Tokenizer] CUDA Graph is disabled, skipping warmup")
            return

        if self._warmed_up:
            logger.warning("[MiMo Tokenizer] CUDA Graph already warmed up, skipping")
            return

        self._device = device
        self.tokenizer.encoder.quantizer.float()
        self.tokenizer.eval()

        logger.info(
            "[MiMo Tokenizer CUDA Graph] Starting warmup for %d sizes: %s",
            len(self.capture_sizes),
            self.capture_sizes,
        )

        # Capture graphs for each size
        for size in self.capture_sizes:
            try:
                self._capture_graph_for_size(size, device, dtype)
                logger.info("[MiMo Tokenizer CUDA Graph] Captured for size=%d", size)
            except Exception as e:
                logger.warning(
                    "[MiMo Tokenizer CUDA Graph] Failed to capture for size=%d: %s",
                    size,
                    e,
                )
                traceback.print_exc()

        self._warmed_up = True
        logger.info(
            "[MiMo Tokenizer CUDA Graph] Warmup complete. Captured %d graphs.",
            len(self.graphs),
        )

    def _capture_graph_for_size(
        self,
        size: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        """
        Capture a CUDA graph for a specific input size.

        Args:
            size: Code sequence length (T)
            device: Device to capture on
            dtype: Input dtype
        """

        # Create static input buffer: (audio_channels, T)
        static_input = torch.zeros(
            self.audio_channels,
            size,
            dtype=dtype,
            device=device,
        )

        hidden_states = self.tokenizer.encoder.decode_vq(static_input)
        seq_len = hidden_states.size(0)
        static_input_length = torch.full(
            (1,), seq_len, dtype=torch.long, device=device
        )

        graph_context = self._compute_graph_context(seq_len)

        with torch.no_grad():
            _ = self.tokenizer.decode(
                static_input,
                static_input_length=static_input_length,
                graph_context=graph_context,
            )

        torch.cuda.synchronize(device)

        # Capture the graph on the dedicated stream
        graph = CUDAGraph()
        with torch.cuda.graph(graph):
            static_output = self.tokenizer.decode(
                static_input,
                static_input_length=static_input_length,
                graph_context=graph_context,
            )

        self.graphs[size] = graph
        self.static_inputs[size] = static_input
        self.static_outputs[size] = static_output
        self.static_input_lengths[size] = static_input_length
        self._graph_contexts[size] = graph_context

    def decode(self, audio_codes: torch.Tensor) -> torch.Tensor:
        """
        Decode audio codes to waveform using CUDA graph if possible.

        Args:
            audio_codes: Input codes tensor of shape (audio_channels, T)

        Returns:
            Decoded audio tensor on GPU
        """
        if not self.enabled or not self._warmed_up:
            return self.tokenizer.decode(audio_codes)

        # Ensure input is on correct device and contiguous
        if audio_codes.device != self._device:
            audio_codes = audio_codes.to(self._device)

        actual_size = audio_codes.shape[-1]
        padded_size = self._get_padded_size(actual_size)

        if padded_size is None or padded_size not in self.graphs:
            logger.warning(
                "Size %d not captured, falling back to eager execution",
                actual_size,
            )
            return self.tokenizer.decode(audio_codes)

        # Copy input to static buffer (with padding if needed)
        self.static_inputs[padded_size].zero_()
        self.static_inputs[padded_size][:, :actual_size] = audio_codes

        self.graphs[padded_size].replay()

        # Get output and trim to actual size
        output = self.static_outputs[padded_size]

        # Calculate actual output length based on upsample ratio
        actual_output_len = actual_size * self.total_upsample

        # Handle output shape: decoder returns (1, 1, samples) or (batch, 1, samples)
        if output.dim() == 3:
            output_flat = output.squeeze(0).squeeze(0)
        else:
            output_flat = output.reshape(-1)

        # Trim to actual length; use min to avoid index out of bounds
        trim_len = min(actual_output_len, output_flat.numel())
        return output_flat[:trim_len].clone()
