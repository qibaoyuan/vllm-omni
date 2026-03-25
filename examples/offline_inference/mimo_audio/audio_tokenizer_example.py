#!/usr/bin/env python3
# Copyright 2025 Xiaomi Corporation.

"""
MiMoAudioTokenizer usage example
=================================

Demonstrates how to use MiMoAudioTokenizer to:
  1. Encode (waveform → mel → discrete codes)
  2. Decode (codes → waveform)
  3. Stream-decode (code chunks → waveform chunks)

Dependencies:
  pip install torch torchaudio soundfile transformers

Usage:
  python audio_tokenizer_example.py \
      --tokenizer-path /path/to/mimo_audio_tokenizer \
      --audio-path    input.wav \
      --output-path   reconstructed.wav
"""

import argparse
import os
import time

import librosa
import soundfile as sf
import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram

from vllm_omni.model_executor.models.mimo_audio.cuda_graph_decoder_wrapper import CUDAGraphMiMoDecoderWrapper
from vllm_omni.model_executor.models.mimo_audio.modeling_audio_tokenizer import (
    MiMoAudioTokenizer,
    StreamingCache,
    StreamingConfig,
)

# ──────────────────────── Utility functions ────────────────────────


def build_mel_transform(config, device="cpu"):
    """Build a MelSpectrogram transform from tokenizer config."""
    return (
        MelSpectrogram(
            sample_rate=config.sampling_rate,
            n_fft=config.nfft,
            hop_length=config.hop_length,
            win_length=config.window_size,
            f_min=config.fmin,
            f_max=config.fmax,
            n_mels=config.n_mels,
            power=1.0,
            center=True,
        )
        .to(device)
        .to(torch.float32)
    )


def load_audio(audio_path: str, target_sr: int) -> torch.Tensor:
    """Load an audio file and resample to target_sr if needed."""
    try:
        wav, sr = torchaudio.load(audio_path)
    except Exception:
        wav, sr = librosa.load(audio_path, sr=target_sr)
        wav = torch.tensor(wav).unsqueeze(0)

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.squeeze(0)  # [samples]
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav


def wav_to_mel(wav: torch.Tensor, mel_transform: MelSpectrogram) -> torch.Tensor:
    """waveform → log mel spectrogram, returns (seq_len, n_mels)."""
    spec = mel_transform(wav[None, :].to(torch.float32))
    return torch.log(torch.clip(spec, min=1e-7)).squeeze().transpose(0, 1)


# ──────────────────────── 1. Encode ────────────────────────


@torch.no_grad()
def encode_audio(
    tokenizer: MiMoAudioTokenizer,
    mel_transform: MelSpectrogram,
    wav: torch.Tensor,
    device: str = "cpu",
):
    """
    Encode a waveform segment into discrete codebook indices (codes).

    Args:
        tokenizer:     MiMoAudioTokenizer instance
        mel_transform: MelSpectrogram matching tokenizer config
        wav:             1-D waveform tensor [samples], float32
        device:          inference device

    Returns:
        codes:          (num_quantizers, total_tokens) discrete codes
        output_length:  length after encoding per sample
    """
    mel = wav_to_mel(wav, mel_transform).to(device)  # (seq_len, n_mels)
    input_lens = torch.tensor([mel.shape[0]], device=device)

    codes, output_length = tokenizer.encode(
        input_features=mel,
        input_lens=input_lens,
        return_codes_only=True,
    )
    return codes, output_length


# ──────────────────────── 2. Decode ────────────────────────


@torch.no_grad()
def decode_codes(tokenizer: MiMoAudioTokenizer, codes: torch.Tensor):
    """
    Decode discrete codes back to an audio waveform.

    Args:
        tokenizer: MiMoAudioTokenizer instance
        codes:     (num_quantizers, total_tokens)

    Returns:
        waveform: (1, 1, samples) reconstructed audio
    """
    return tokenizer.decode(codes)


# ──────────────────────── 3. Streaming decode ────────────────────────


@torch.no_grad()
def streaming_decode_codes(
    tokenizer: MiMoAudioTokenizer,
    codes: torch.Tensor,
    chunk_size: int = 50,
):
    """
    Stream-decode by chunking codes, simulating realtime synthesis.

    Args:
        tokenizer:  MiMoAudioTokenizer instance
        codes:      (num_quantizers, total_tokens)
        chunk_size: number of tokens per chunk

    Returns:
        wav_chunks: list of waveforms decoded from each chunk
    """
    total_tokens = codes.shape[-1]
    num_chunks = (total_tokens + chunk_size - 1) // chunk_size

    cache = StreamingCache()
    config = StreamingConfig()
    wav_chunks = []

    for i in range(num_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, total_tokens)
        chunk_codes = codes[:, start:end]
        is_last = i == num_chunks - 1

        chunk_lengths = [chunk_codes.shape[-1]]
        wavs, cache = tokenizer.streaming_decode(
            codes_chunks=chunk_codes,
            chunk_input_lengths=chunk_lengths,
            history_cache=cache,
            streaming_config=config,
            last_chunk=is_last,
        )
        if wavs[0] is not None:
            wav_chunks.append(wavs[0])
            print(f"  chunk {i + 1}/{num_chunks}: generated {wavs[0].shape[-1]} samples")
        else:
            print(f"  chunk {i + 1}/{num_chunks}: buffering (no output yet)")

    return wav_chunks


# ──────────────────────── Main entry ────────────────────────

"""
export CODE_DIR="/mnt/user/qibaoyuan/vllm-omni-qby/"
export MODEL_DIR="/mnt/user/qibaoyuan"
export CUDA_LAUNCH_BLOCKING=1
export MIMO_AUDIO_TOKENIZER_PATH="/mnt/user/qibaoyuan/MiMo-Audio-Tokenizer"

cd /mnt/user/qibaoyuan/vllm-omni-qby/examples/offline_inference/mimo_audio/
python3 -u audio_tokenizer_example.py  --tokenizer-path ${MIMO_AUDIO_TOKENIZER_PATH} \
--audio-path /mnt/user/qibaoyuan/vllm-omni-qby/examples/offline_inference/mimo_audio/beijing.mp3

"""


def main():
    parser = argparse.ArgumentParser(description="MiMoAudioTokenizer usage example")
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="/Users/qibaoyuan/Documents/llm/MiMo-Audio-Tokenizer",
    )
    parser.add_argument(
        "--audio-path",
        type=str,
        default="/Users/qibaoyuan/PycharmProjects/vllm-omni-qby/examples/offline_inference/mimo_audio/beijing.mp3",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="reconstructed.wav",
        help="Path to save reconstructed audio",
    )
    parser.add_argument(
        "--streaming-output-path",
        type=str,
        default="reconstructed_streaming.wav",
        help="Path to save streaming-reconstructed audio",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device (cpu / cuda)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50,
        help="Number of tokens per chunk in streaming decode",
    )
    args = parser.parse_args()

    device = args.device
    print(f"Using device: {device}")

    # ── Step 1: load tokenizer ──
    print("\n[1/5] Loading MiMoAudioTokenizer ...")
    t0 = time.time()
    tokenizer = MiMoAudioTokenizer.from_pretrained(
        args.tokenizer_path,
        device_map={"": device},
    )
    tokenizer.eval()

    # Convert dtype before CUDA Graph warmup/capture; otherwise capture
    # records kernels for float32 weight addresses, and replay after
    # .to(bfloat16) will read wrong values.
    if device != "cpu":
        tokenizer.to(dtype=torch.bfloat16)

    _cudagraph_wrapper = CUDAGraphMiMoDecoderWrapper(
        tokenizer=tokenizer,
        enabled=True,
    )
    _cudagraph_wrapper.warmup(
        device=torch.device(device),
        codec_chunk_frames=3,
        codec_left_context_frames=3,
    )
    print(f"      Load finished in {time.time() - t0:.2f}s")
    print(f"      Sample rate: {tokenizer.sampling_rate}")
    print(f"      Num quantizers: {tokenizer.config.num_quantizers}")
    print(f"      Codebook size: {tokenizer.config.codebook_size}")
    print(f"      Downsample rate: {tokenizer.downsample_rate}")

    # ── Step 2: load and preprocess audio ──
    print(f"\n[2/5] Loading audio: {args.audio_path}")
    wav = load_audio(args.audio_path, tokenizer.sampling_rate).to(device)
    duration = wav.shape[0] / tokenizer.sampling_rate
    print(f"      Duration: {duration:.2f}s, samples: {wav.shape[0]}")

    mel_transform = build_mel_transform(tokenizer.config, device)

    # ── Step 3: encode ──
    print("\n[3/5] Encoding audio → discrete codes ...")
    t0 = time.time()
    codes, output_length = encode_audio(tokenizer.encoder, mel_transform, wav, device)
    print(f"      codes shape: {codes.shape}  (num_quantizers × tokens)")
    print(f"      output_length: {output_length}")
    print(f"      Compression ratio: {wav.shape[0] / codes.shape[-1]:.1f}x")
    print(f"      Encode time: {time.time() - t0:.3f}s")

    # ── Step 3: encode (cg) ──
    # print("\n[3/5] Encoding audio → discrete codes ...")
    # t0 = time.time()
    # codes, output_length = encode_audio(_cudagraph_wrapper , mel_transform, wav, device)
    # print(f"      cg codes shape: {codes.shape}  (num_quantizers × tokens)")
    # print(f"      cg output_length: {output_length}")
    # print(f"      cg compression ratio: {wav.shape[0] / codes.shape[-1]:.1f}x")
    # print(f"      cg encode time: {time.time() - t0:.3f}s")

    # ── Step 4: non-streaming decode ──
    print("\n[4/5] Decoding codes → waveform (non-streaming)...")
    t0 = time.time()
    recon_wav = decode_codes(tokenizer, codes)
    recon_wav_np = recon_wav.float().detach().cpu().numpy().flatten()
    print(f"      Reconstructed length: {len(recon_wav_np)} samples")
    print(f"      Decode time: {time.time() - t0:.3f}s")

    sf.write(args.output_path, recon_wav_np, tokenizer.sampling_rate, format="WAV")
    print(f"      Saved: {args.output_path}")

    # ── Step 4: CUDA-graph non-streaming decode ──
    os.environ["MIMO_AUDIO_TOKENIZER_CUDA_GRAPH"] = "1"
    print("\n[5/7] CUDA-graph decode codes → waveform (non-streaming)...")
    t0 = time.time()
    recon_wav = decode_codes(_cudagraph_wrapper, codes)
    recon_wav_np = recon_wav.float().detach().cpu().numpy().flatten()
    print(f"      CUDA-graph reconstructed length: {len(recon_wav_np)} samples")
    print(f"      CUDA-graph decode time: {time.time() - t0:.3f}s")

    sf.write(args.output_path + ".cg.wav", recon_wav_np, tokenizer.sampling_rate, format="WAV")
    print(f"      CUDA-graph saved: {args.output_path + '.cg.wav'}")

    # ── Step 5: streaming decode ──
    print(f"\n[6/7] Streaming decode (chunk_size={args.chunk_size})...")
    t0 = time.time()
    wav_chunks = streaming_decode_codes(tokenizer, codes, chunk_size=args.chunk_size)
    if wav_chunks:
        full_wav = torch.cat(wav_chunks, dim=-1).float().detach().cpu().numpy().flatten()
        print(f"      Streaming total length: {len(full_wav)} samples")
        print(f"      Streaming decode time: {time.time() - t0:.3f}s")
        sf.write(args.streaming_output_path, full_wav, tokenizer.sampling_rate, format="WAV")
        print(f"      Saved: {args.streaming_output_path}")
    else:
        print("      No output (audio too short)")

    # ── Step 6: CUDA-graph streaming decode ──
    print(f"\n[7/7] CUDA-graph streaming decode (chunk_size={args.chunk_size})...")
    t0 = time.time()
    wav_chunks = streaming_decode_codes(_cudagraph_wrapper, codes, chunk_size=args.chunk_size)
    if wav_chunks:
        full_wav = torch.cat(wav_chunks, dim=-1).float().detach().cpu().numpy().flatten()
        print(f"      CUDA-graph streaming total length: {len(full_wav)} samples")
        print(f"      CUDA-graph streaming decode time: {time.time() - t0:.3f}s")
        sf.write(args.streaming_output_path + ".cg.wav", full_wav, tokenizer.sampling_rate, format="WAV")
        print(f"      CUDA-graph saved: {args.streaming_output_path + '.cg.wav'}")
    else:
        print("      No output (audio too short)")

    print("\nDone.")


if __name__ == "__main__":
    main()
