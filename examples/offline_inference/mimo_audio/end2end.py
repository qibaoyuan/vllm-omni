# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This example shows how to use vLLM for running offline inference
with the correct prompt format on Qwen3-Omni (thinker only).
"""

import os

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
from vllm_omni.inputs.data import OmniTokensPrompt

from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_omni.entrypoints.omni import Omni

SEED = 42
MAX_CODE2WAV_TOKENS = 18192


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


def get_text_audio_query(question: str = None) -> QueryResult:
    if question is None:
        # question = "Explain the system architecture for a scalable audio generation pipeline. Answer in 15 words."
        # question = "请将这段文字转换为语音: 我是小米研发的mimo-audio机器人"
        question = "好的，我们来看看北京的历史天气。根据北京历史气象数据的平均值，可以知道：在北京，一月份的平均气温是零下四点六摄氏度，最低会到零下九点三摄氏度，最高是零点四摄氏度，这时候风力通常是二级，刮的是西北风。到了七月份，天气就热了，平均气温会达到二十六点二摄氏度，最低十三点四摄氏度，最高三十九点一摄氏度，风力会变大，有四级，主要刮的是东南风。而到了最热的八月份，平均气温是三十一点二摄氏度，最低二十五点五摄氏度，最高能达到四十二点一摄氏度，风力也最大，有五级，风向是东风。另外还有几点补充说明。第一，北京的冬天很冷，而且又干又燥，夏天呢，就又热又湿，偶尔还会出现极端高温。第二，春天和秋天的温差特别大，要特别注意防风防沙。第三，从十二月到第二年的三月，是北京的雾霾高发期，所以最好选择在不下雨、空气好的时候出门。如果你想了解更具体的数据，比如某一年的或者降水概率这些，可以再告诉我！"
    # prompt = (
    #     f"<|im_start|>system\n{default_system}<|im_end|>\n"
    #     "<|im_start|>user\n"
    #     f"{question}<|im_end|>\n"
    #     f"<|im_start|>assistant\n"
    # )
    """ Original TTS task prompt example """
    prompt = f"<|im_start|>user\n请将这段文字转换为语音: {question}<|im_end|>\n<|im_start|>assistant\n<|sostm|>"
    return QueryResult(
        inputs={
            "prompt": prompt,
        },
        limit_mm_per_prompt={},
    )


def get_video_query(question: str = None) -> QueryResult:
    if question is None:
        question = "Why is this video funny?"
    prompt = (
        f"<|im_start|>system\n{default_system}<|im_end|>\n"
        "<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>"
        f"{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    return QueryResult(
        inputs={
            "prompt": prompt,
            "multi_modal_data": {
                "video": VideoAsset(name="baby_reading", num_frames=16).np_ndarrays,
            },
        },
        limit_mm_per_prompt={"video": 1},
    )


def get_image_query(question: str = None) -> QueryResult:
    if question is None:
        question = "What is the content of this image?"
    prompt = (
        f"<|im_start|>system\n{default_system}<|im_end|>\n"
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
        f"{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return QueryResult(
        inputs={
            "prompt": prompt,
            "multi_modal_data": {
                "image": convert_image_mode(ImageAsset("cherry_blossom").pil_image, "RGB"),
            },
        },
        limit_mm_per_prompt={"image": 1},
    )


def get_audio_query(question: str = None) -> QueryResult:
    if question is None:
        question = "What is the content of this audio?"
    prompt = (
        f"<|im_start|>system\n{default_system}<|im_end|>\n"
        "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|>"
        f"{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return QueryResult(
        inputs={
            "prompt": prompt,
            "multi_modal_data": {
                "audio": AudioAsset("mary_had_lamb").audio_and_sample_rate,
            },
        },
        limit_mm_per_prompt={"audio": 1},
    )


def get_multi_audios_query_from_json(message_path: str, sampling_rate: int = 24000) -> QueryResult:
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

        prompts.append(f"<|im_start|>{role}\n")

        if role == "assistant":
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
                audio_list.append(audio_data)

            prompts.append("<|eostm|>")

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
                        audio_file = io.BytesIO(audio_bytes)  # wav: [channels, num_samples]
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


def get_codes_query_from_json(codes_path: str) -> QueryResult:
    with open(codes_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        code_final = data
    elif isinstance(data, dict) and "code_final" in data:
        code_final = data["code_final"]
    else:
        raise ValueError(
            f"Unsupported codes json format in {codes_path}.\n"
            "Expect a JSON list[int] or {{'code_final': list[int]}}."
        )

    if not isinstance(code_final, list) or not all(isinstance(x, int) for x in code_final):
        raise ValueError("code_final must be a list[int].")

    if len(code_final) > MAX_CODE2WAV_TOKENS:
        print(f"[Warn] code_final len={len(code_final)} > {MAX_CODE2WAV_TOKENS}, truncating.")
        code_final = code_final[:MAX_CODE2WAV_TOKENS]

    return QueryResult(
        inputs=OmniTokensPrompt(
            prompt_token_ids=code_final,
            multi_modal_data=None,
            mm_processor_kwargs=None,
        ),
        limit_mm_per_prompt={},
    )


def get_multi_text_dialogues_query(question: str = None) -> QueryResult:
    if question is None:
        question = "可以给我介绍一些中国的旅游景点吗？"
    prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    return QueryResult(
        inputs={
            "prompt": prompt,
        },
        limit_mm_per_prompt={},
    )


query_map = {
    "text_audio": get_text_audio_query,
    "use_audio": get_audio_query,
    "use_image": get_image_query,
    "use_video": get_video_query,
    "multi_audios": get_multi_audios_query_from_json,
    "multi_texts": get_multi_text_dialogues_query,
    "codes": get_codes_query_from_json,
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
        log_file=(
            "omni_llm_pipeline.log"
            if args.enable_stats
            else None
        ),
        init_sleep_seconds=args.init_sleep_seconds,
        batch_timeout=args.batch_timeout,
        init_timeout=args.init_timeout,
        shm_threshold_bytes=args.shm_threshold_bytes,
    )

    thinker_sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=3200,
        seed=SEED,
        logit_bias={},
        repetition_penalty=1.1,
    )

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

    ##spec
    if args.query_type == "codes":
        sampling_params_list = [code2wav_sampling_params]
        prompts_query_result = query_func(args.codes_path)
    elif args.query_type == "text_audio":
        prompts_query_result = query_func()
    elif args.query_type == "multi_texts":
        sampling_params_list = [thinker_sampling_params]
        prompts_query_result = query_func()
    elif args.query_type == "multi_audios":
        prompts_query_result = query_func(message_path, sampling_rate)
    else:
        raise ValueError(f"Invalid query type: {args.query_type}, only support text and multi_audios")

    prompts = [prompts_query_result.inputs for _ in range(args.num_prompts)]

    ##tts
    # prompts = [
    #     # '<|im_start|>user\n你是谁？<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n',
    #     "<|im_start|>user\n请将这段文字转换为语音: 我是小米研发的mimo-audio大语言模型<|im_end|>\n<|im_start|>assistant\n<|sostm|>",
    #     # "<|im_start|>user\n请将这段文字转换为语音: 今天天气真好<|im_end|>\n<|im_start|>assistant\n<|sostm|>",
    # ]
    # prompts = [{"prompt": prompt for prompt in prompts}]

    print("prompts", prompts)
    omni_outputs = omni_llm.generate(prompts, sampling_params_list)

    output_dir = args.output_dir if getattr(args, "output_dir", None) else args.output_wav
    if args.query_type is not None:
        output_dir = os.path.join(output_dir, args.query_type)
    os.makedirs(output_dir, exist_ok=True)

    for stage_outputs in omni_outputs:
        if stage_outputs.final_output_type == "text":
            for output in stage_outputs.request_output:
                request_id = output.request_id
                text_output = output.outputs[0].text
                # Save aligned text file per request
                prompt_text = output.prompt
                out_txt = os.path.join(output_dir, f"{request_id}.txt")
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
                request_id = output.request_id
                audio_tensor = output.multimodal_output["audio"]
                output_wav = os.path.join(output_dir, f"{request_id}.wav")

                # Convert to numpy array and ensure correct format
                audio_numpy = audio_tensor.float().detach().cpu().numpy()

                # Ensure audio is 1D (flatten if needed)
                if audio_numpy.ndim > 1:
                    audio_numpy = audio_numpy.flatten()

                # Save audio file with explicit WAV format
                sf.write(output_wav, audio_numpy, samplerate=24000, format="WAV")
                print(f"Request ID: {request_id}, Saved audio to {output_wav}")


def parse_args():
    parser = FlexibleArgumentParser(description="Demo on using vLLM for offline inference with audio language models")
    parser.add_argument(
        "--model-name",
        "-m",
        type=str,
        default="MiMo-Audio-7B-Instruct",
        help="Backbone LLM path.",
    )
    parser.add_argument(
        "--message-path",
        "-mp",
        type=str,
        default="",
        help="The path for query messages from users in frontend.",
    )
    parser.add_argument(
        "--query-type",
        "-q",
        type=str,
        default="text_audio",
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
    parser.add_argument(
        "--codes-path",
        type=str,
        default="./code_final.json",
        help="Path to a codes json file.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
