#!/usr/bin/env python3
# Copyright 2025 Xiaomi Corporation.
"""
MiMoAudioTokenizer 使用示例
============================

演示如何使用 MiMoAudioTokenizer 对音频进行：
  1. 编码（waveform → mel → discrete codes）
  2. 解码（codes → waveform）
  3. 流式解码（codes chunks → waveform chunks）

依赖:
  pip install torch torchaudio soundfile transformers

用法:
  python audio_tokenizer_example.py \
      --tokenizer-path /path/to/mimo_audio_tokenizer \
      --audio-path    input.wav \
      --output-path   reconstructed.wav
"""

import argparse
import time

import soundfile as sf
import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram

from vllm_omni.model_executor.models.mimo_audio.modeling_audio_tokenizer import (
    MiMoAudioTokenizer,
    StreamingCache,
    StreamingConfig,
)

# ──────────────────────── 工具函数 ────────────────────────


def build_mel_transform(config, device="cpu"):
    """根据 tokenizer config 构建 MelSpectrogram 变换."""
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
    """加载音频文件，必要时重采样到 target_sr."""
    wav, sr = torchaudio.load(audio_path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.squeeze(0)  # [samples]
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav


def wav_to_mel(wav: torch.Tensor, mel_transform: MelSpectrogram) -> torch.Tensor:
    """waveform → log mel spectrogram, 返回 (seq_len, n_mels)."""
    spec = mel_transform(wav[None, :].to(torch.float32))
    return torch.log(torch.clip(spec, min=1e-7)).squeeze().transpose(0, 1)


# ──────────────────────── 1. 编码 ────────────────────────


@torch.no_grad()
def encode_audio(
    tokenizer: MiMoAudioTokenizer,
    mel_transform: MelSpectrogram,
    wav: torch.Tensor,
    device: str = "cpu",
):
    """
    将一段音频波形编码为离散码本索引 (codes).

    参数:
        tokenizer:     MiMoAudioTokenizer 实例
        mel_transform: 与 tokenizer config 对应的 MelSpectrogram
        wav:           一维波形张量 [samples], float32
        device:        推理设备

    返回:
        codes:          (num_quantizers, total_tokens)  离散码
        output_length:  编码后每条样本的长度
    """
    mel = wav_to_mel(wav, mel_transform).to(device)  # (seq_len, n_mels)
    input_lens = torch.tensor([mel.shape[0]], device=device)

    codes, output_length = tokenizer.encoder.encode(
        input_features=mel,
        input_lens=input_lens,
        return_codes_only=True,
    )
    return codes, output_length


# ──────────────────────── 2. 解码 ────────────────────────


@torch.no_grad()
def decode_codes(tokenizer: MiMoAudioTokenizer, codes: torch.Tensor):
    """
    将离散码解码回音频波形.

    参数:
        tokenizer: MiMoAudioTokenizer 实例
        codes:     (num_quantizers, total_tokens)

    返回:
        waveform: (1, 1, samples) 重建后的音频
    """
    return tokenizer.decode(codes)


# ──────────────────────── 3. 流式解码 ────────────────────────


@torch.no_grad()
def streaming_decode_codes(
    tokenizer: MiMoAudioTokenizer,
    codes: torch.Tensor,
    chunk_size: int = 50,
):
    """
    将离散码分块后进行流式解码，模拟实时合成场景.

    参数:
        tokenizer:  MiMoAudioTokenizer 实例
        codes:      (num_quantizers, total_tokens)
        chunk_size: 每次送入的 token 数

    返回:
        wav_chunks: 每个 chunk 解码得到的波形列表
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
            print(f"  chunk {i + 1}/{num_chunks}: 生成 {wavs[0].shape[-1]} 采样点")
        else:
            print(f"  chunk {i + 1}/{num_chunks}: 缓冲中（尚未输出）")

    return wav_chunks


# ──────────────────────── 主流程 ────────────────────────

"""
export CODE_DIR="/mnt/user/qibaoyuan/vllm-omni-qby/"
export MODEL_DIR="/mnt/user/qibaoyuan"
export MIMO_AUDIO_TOKENIZER_PATH="/mnt/user/qibaoyuan/MiMo-Audio-Tokenizer"

cd /mnt/user/qibaoyuan/vllm-omni-qby/examples/offline_inference/mimo_audio/
python3 -u audio_tokenizer_example.py  --tokenizer-path ${MIMO_AUDIO_TOKENIZER_PATH} \
--audio-path /mnt/user/qibaoyuan/vllm-omni-qby/examples/offline_inference/mimo_audio/自然对话_闺蜜闲聊_剪.wav

"""


def main():
    parser = argparse.ArgumentParser(description="MiMoAudioTokenizer 使用示例")
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="/Users/qibaoyuan/Documents/llm/MiMo-Audio-Tokenizer",
    )
    parser.add_argument(
        "--audio-path",
        type=str,
        default="/Users/qibaoyuan/PycharmProjects/vllm-omni-qby/examples/offline_inference/mimo_audio/自然对话_闺蜜闲聊_剪.wav",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="reconstructed.wav",
        help="重建音频保存路径",
    )
    parser.add_argument(
        "--streaming-output-path",
        type=str,
        default="reconstructed_streaming.wav",
        help="流式重建音频保存路径",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="推理设备 (cpu / cuda)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50,
        help="流式解码每个 chunk 的 token 数",
    )
    args = parser.parse_args()

    device = args.device
    print(f"使用设备: {device}")

    # ── Step 1: 加载 tokenizer ──
    print("\n[1/5] 加载 MiMoAudioTokenizer ...")
    t0 = time.time()
    tokenizer = MiMoAudioTokenizer.from_pretrained(
        args.tokenizer_path,
        device_map={"": device},
    )
    tokenizer.eval()
    if device != "cpu":
        tokenizer.to(dtype=torch.bfloat16)
    print(f"      加载完成，耗时 {time.time() - t0:.2f}s")
    print(f"      采样率: {tokenizer.sampling_rate}")
    print(f"      量化器数: {tokenizer.config.num_quantizers}")
    print(f"      码本大小: {tokenizer.config.codebook_size}")
    print(f"      下采样倍率: {tokenizer.downsample_rate}")

    # ── Step 2: 加载并预处理音频 ──
    print(f"\n[2/5] 加载音频: {args.audio_path}")
    wav = load_audio(args.audio_path, tokenizer.sampling_rate).to(device)
    duration = wav.shape[0] / tokenizer.sampling_rate
    print(f"      时长: {duration:.2f}s, 采样点数: {wav.shape[0]}")

    mel_transform = build_mel_transform(tokenizer.config, device)

    # ── Step 3: 编码 ──
    print("\n[3/5] 编码音频 → 离散码 ...")
    t0 = time.time()
    codes, output_length = encode_audio(tokenizer, mel_transform, wav, device)
    print(f"      codes shape: {codes.shape}  (num_quantizers × tokens)")
    print(f"      output_length: {output_length}")
    print(f"      压缩比: {wav.shape[0] / codes.shape[-1]:.1f}x")
    print(f"      编码耗时: {time.time() - t0:.3f}s")

    # ── Step 4: 非流式解码 ──
    print("\n[4/5] 解码离散码 → 波形（非流式）...")
    t0 = time.time()
    recon_wav = decode_codes(tokenizer, codes)
    recon_wav_np = recon_wav.float().detach().cpu().numpy().flatten()
    print(f"      重建波形长度: {len(recon_wav_np)} 采样点")
    print(f"      解码耗时: {time.time() - t0:.3f}s")

    sf.write(args.output_path, recon_wav_np, tokenizer.sampling_rate, format="WAV")
    print(f"      已保存: {args.output_path}")

    # ── Step 5: 流式解码 ──
    print(f"\n[5/5] 流式解码（chunk_size={args.chunk_size}）...")
    t0 = time.time()
    wav_chunks = streaming_decode_codes(tokenizer, codes, chunk_size=args.chunk_size)
    if wav_chunks:
        full_wav = torch.cat(wav_chunks, dim=-1).float().detach().cpu().numpy().flatten()
        print(f"      流式重建总长度: {len(full_wav)} 采样点")
        print(f"      流式解码耗时: {time.time() - t0:.3f}s")
        sf.write(args.streaming_output_path, full_wav, tokenizer.sampling_rate, format="WAV")
        print(f"      已保存: {args.streaming_output_path}")
    else:
        print("      无输出（音频过短）")

    print("\n完成！")


if __name__ == "__main__":
    main()
