# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This example shows how to use vLLM for running offline inference
with the correct prompt format on Qwen3-Omni (thinker only).
"""

import os
import time
from typing import NamedTuple, List

import numpy as np
import json
import io
import base64
import librosa
import soundfile as sf
from vllm import SamplingParams
from vllm.assets.audio import AudioAsset
from vllm.assets.image import ImageAsset
from vllm.assets.video import VideoAsset
from vllm.multimodal.image import convert_image_mode

# from vllm.multimodal.image import convert_image_mode
from vllm.utils import FlexibleArgumentParser

from vllm_omni.entrypoints.omni import Omni

SEED = 42


class QueryResult(NamedTuple):
    inputs: dict
    limit_mm_per_prompt: dict[str, int]


# NOTE: The default `max_num_seqs` and `max_model_len` may result in OOM on
# lower-end GPUs.
# Unless specified, these settings have been tested to work on a single L4.

default_system = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba "
    "Group, capable of perceiving auditory and visual inputs, as well as "
    "generating text and speech."
)


def get_text_query(question: str = None) -> QueryResult:
    if question is None:
        # question = "Explain the system architecture for a scalable audio generation pipeline. Answer in 15 words."
        question = "请将这段文字转换为语音: 我是小米研发的mimo-audio机器人"

    """ Original TTS task prompt example """
    prompt = (
        "<|im_start|>user\n"
        f"{question}<|im_end|>\n"
        "<|im_start|>assistant\n<|sostm|>"
    )
    return QueryResult(
        inputs={
            "prompt": prompt,
        },
        limit_mm_per_prompt={},
    )


def get_multi_audios_query_from_json(
        message_path: str,
        sampling_rate: int = 24000
) -> QueryResult:
    """
    Get multi-audios query from json file. Supports base64 audio and file paths.
    
    Format rules:
    - Pure text: just text
    - Pure audio: <|empty|> (will be replaced by mimo_audio.py)
    - Text + audio (interleaved): <|sostm|>text<|eot|><|eostm|><|empty|>
    """
    with open(message_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    messages = data.get("messages", [])
    prompts: List[str] = []
    audio_list: List[tuple] = []

    for idx, msg in enumerate(messages):
        role = msg.get("role", "user")
        contents = msg.get("content", [])

        # Check if content has both text and audio
        has_text = any(c.get("type") == "text" for c in contents)
        has_audio = any(c.get("type") == "audio_url" for c in contents)

        prompts.append(f"<|im_start|>{role}\n")

        if role == "assistant" and has_text and has_audio:
            # Special handling for text + audio interleaved case
            text_content = ""
            audio_data = None

            # Extract text and audio separately
            for content in contents:
                ctype = content.get("type")

                if ctype == "text":
                    text_content = content.get("text", "")

                elif ctype == "audio_url":
                    audio_url = content.get("audio_url", {}).get("url", None)
                    if audio_url is None:
                        raise ValueError(f"audio_url is None for role: {role}")

                    # Handle both file paths and base64 data URLs
                    if audio_url.startswith("data:"):
                        header, b64_data = audio_url.split(",", 1)
                        audio_bytes = base64.b64decode(b64_data.strip())
                        audio_file = io.BytesIO(audio_bytes)
                    else:
                        # File path
                        audio_file = audio_url

                    audio_signal, sr = librosa.load(audio_file, sr=sampling_rate)
                    audio_data = (audio_signal.astype(np.float32), sr)

            # Build interleaved format
            prompts.append("<|sostm|>")
            prompts.append(text_content)
            prompts.append("<|eot|>")

            # Add audio with <|empty|> placeholder
            if audio_data is not None:
                prompts.append("<|empty|>")
                prompts.append("<|eostm|>")
                audio_list.append(audio_data)

        else:
            # Handle pure text or pure audio
            for content in contents:
                ctype = content.get("type")

                if ctype == "text":
                    text = content.get("text", "")
                    prompts.append(text)

                elif ctype == "audio_url":
                    audio_url = content.get("audio_url", {}).get("url", None)
                    if audio_url is None:
                        raise ValueError(f"audio_url is None for role: {role}")

                    # Handle both file paths and base64 data URLs
                    if audio_url.startswith("data:"):
                        header, b64_data = audio_url.split(",", 1)
                        audio_bytes = base64.b64decode(b64_data.strip())
                        audio_file = io.BytesIO(audio_bytes)
                    else:
                        # File path
                        audio_file = audio_url

                    audio_signal, sr = librosa.load(audio_file, sr=sampling_rate)
                    audio_data = (audio_signal.astype(np.float32), sr)
                    prompts.append("<|sosp|>")
                    prompts.append("<|empty|>")
                    prompts.append("<|eosp|>")
                    audio_list.append(audio_data)

                else:
                    prompts.append("")

        prompts.append("<|im_end|>\n")

    prompts.append("<|im_start|>assistant\n<|sostm|>")
    prompts = "".join(prompts)
    return QueryResult(
        inputs={
            "prompt": prompts,
            "multi_modal_data": {
                "audio": audio_list,
            },
        },
        limit_mm_per_prompt={
            "audio": len(audio_list) if len(audio_list) > 0 else 0,
        },
    )


query_map = {
    "text": get_text_query,
}


def main(args):
    model_name = args.model_name
    message_path = args.message_path
    sampling_rate = args.sampling_rate
    query_func = query_map[args.query_type]

    omni_llm = Omni(
        model=model_name,
        stage_configs_path=args.stage_configs_path,
        log_stats=args.enable_stats,
        log_file=("./logs/omni_llm_pipeline.log" if args.enable_stats else None),
        init_sleep_seconds=args.init_sleep_seconds,
        batch_timeout=args.batch_timeout,
        init_timeout=args.init_timeout,
        shm_threshold_bytes=args.shm_threshold_bytes,
    )

    thinker_sampling_params = SamplingParams(
        temperature=0.4,
        top_p=0.9,
        top_k=-1,
        max_tokens=1200,
        repetition_penalty=1.05,
        logit_bias={},
        seed=SEED,
    )

    talker_sampling_params = SamplingParams(
        temperature=0.9,
        top_k=50,
        max_tokens=4096,
        seed=SEED,
        detokenize=False,
        repetition_penalty=1.05,
        stop_token_ids=[2150],  # TALKER_CODEC_EOS_TOKEN_ID
    )

    # Sampling parameters for Code2Wav stage (audio generation)
    code2wav_sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=4096 * 16,
        seed=SEED,
        detokenize=True,
        repetition_penalty=1.1,
    )

    sampling_params_list = [
        thinker_sampling_params,
        code2wav_sampling_params,
    ]

    if args.query_type == "text":
        prompts_query_result = query_func()
    elif args.query_type == "multi_audios":
        prompts_query_result = query_func(message_path, sampling_rate)
    else:
        raise ValueError(f"Invalid query type: {args.query_type}, only support text and multi_audios")

    prompts = [prompts_query_result.inputs for _ in range(args.num_prompts)]


    t0 = time.time()
    omni_outputs = omni_llm.generate(prompts, sampling_params_list)

    output_dir = args.output_dir if getattr(args, "output_dir", None) else args.output_wav
    if args.query_type is not None:
        output_dir = os.path.join(output_dir, args.query_type)
    os.makedirs(output_dir, exist_ok=True)

    for stage_outputs in omni_outputs:
        if stage_outputs.final_output_type == "text":
            for output in stage_outputs.request_output:
                request_id = int(output.request_id)
                text_output = output.outputs[0].text
                # Save aligned text file per request
                prompt_text = prompts[request_id]["prompt"]
                out_txt = os.path.join(output_dir, f"{request_id:05d}.txt")
                lines = []
                lines.append("Prompt:\n")
                lines.append(str(prompt_text) + "\n")
                lines.append("vllm_text_output:\n")
                lines.append(str(text_output).strip() + "\n")
                try:
                    with open(out_txt, "w", encoding="utf-8") as f:
                        print("lines", lines)
                        f.writelines(lines)
                except Exception as e:
                    print(f"[Warn] Failed writing text file {out_txt}: {e}")
                print(f"Request ID: {request_id}, Text saved to {out_txt}")
        elif stage_outputs.final_output_type == "audio":
            for output in stage_outputs.request_output:
                request_id = int(output.request_id)
                audio_tensor = output.multimodal_output["audio"]
                from datetime import datetime

                time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                output_wav = os.path.join(output_dir, f"output_{output.request_id}_{time_str}.wav")

                # Convert to numpy array and ensure correct format
                audio_numpy = audio_tensor.float().detach().cpu().numpy()

                # Ensure audio is 1D (flatten if needed)
                if audio_numpy.ndim > 1:
                    audio_numpy = audio_numpy.flatten()

                # Save audio file with explicit WAV format
                sf.write(output_wav, audio_numpy, samplerate=24000, format="WAV")
                print(f"Request ID: {request_id}, Saved audio to {output_wav}")

    print('time cost', time.time() - t0, 'per', (time.time() - t0) / 10.0)


def parse_args():
    parser = FlexibleArgumentParser(description="Demo on using vLLM for offline inference with audio language models")
    parser.add_argument(
        "--model-name",
        "-m",
        type=str,
        default="/app/vllm-omni-dev/models/MiMo-Audio-7B-Instruct",
        help="Backbone LLM path.",
    )
    parser.add_argument(
        "--message-path",
        "-mp",
        type=str,
        default="/app/vllm-omni-dev/examples/message_base64.json",
        help="The path for query messages from users in frontend.",
    )
    parser.add_argument(
        "--query-type",
        "-q",
        type=str,
        default="multi_audios",
        choices=query_map.keys(),
        help="Query type.",
    )
    parser.add_argument(
        "--enable-stats",
        action="store_true",
        default=True,
        help="Enable writing detailed statistics (default: disabled)",
    )
    parser.add_argument(
        "--init-sleep-seconds",
        type=int,
        default=20,
        help="Sleep seconds after starting each stage process to allow initialization (default: 20)",
    )
    parser.add_argument(
        "--batch-timeout",
        type=int,
        default=5,
        help="Timeout for batching in seconds (default: 5)",
    )
    parser.add_argument(
        "--init-timeout",
        type=int,
        default=5000,
        help="Timeout for initializing stages in seconds (default: 300)",
    )
    parser.add_argument(
        "--shm-threshold-bytes",
        type=int,
        default=65536,
        help="Threshold for using shared memory in bytes (default: 65536)",
    )
    parser.add_argument(
        "--output-dir",
        default="./output_audio",
        help="[Deprecated] Output wav directory (use --output-dir).",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1,
        help="Number of prompts to generate.",
    )
    parser.add_argument(
        "--txt-prompts",
        type=str,
        default=None,
        help="Path to a .txt file with one prompt per line (preferred).",
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=24000,
        help="Sampling rate for audio.",
    )
    parser.add_argument(
        "--stage-configs-path",
        type=str,
        default="vllm_omni/model_executor/stage_configs/mimo_audio/mimo_audio_llm_code2wav.yaml",
        help="Path to a stage configs file.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
