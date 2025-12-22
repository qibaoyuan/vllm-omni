import base64

from typing import Optional

import requests
from openai import OpenAI
from vllm.assets.audio import AudioAsset
from vllm.utils import FlexibleArgumentParser

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8091/v1"
import os

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
    base_url=openai_api_base,
)

SEED = 42


def encode_base64_content_from_url(content_url: str) -> str:
    """Encode a content retrieved from a remote url to base64 format."""

    with requests.get(content_url) as response:
        response.raise_for_status()
        result = base64.b64encode(response.content).decode("utf-8")

    return result


def encode_base64_content_from_file(file_path: str) -> str:
    """Encode a local file to base64 format."""
    with open(file_path, "rb") as f:
        content = f.read()
        result = base64.b64encode(content).decode("utf-8")
    return result


def get_audio_url_from_path(audio_path: Optional[str]) -> str:
    """Convert an audio path (local file or URL) to an audio URL format for the API.

    If audio_path is None or empty, returns the default URL.
    If audio_path is a local file path, encodes it to base64 data URL.
    If audio_path is a URL, returns it as-is.
    """
    if not audio_path:
        # Default audio URL
        return AudioAsset("mary_had_lamb").url

    # Check if it's a URL (starts with http:// or https://)
    if audio_path.startswith(("http://", "https://")):
        return audio_path

    # Otherwise, treat it as a local file path
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Detect audio MIME type from file extension
    audio_path_lower = audio_path.lower()
    if audio_path_lower.endswith((".mp3", ".mpeg")):
        mime_type = "audio/mpeg"
    elif audio_path_lower.endswith(".wav"):
        mime_type = "audio/wav"
    elif audio_path_lower.endswith(".ogg"):
        mime_type = "audio/ogg"
    elif audio_path_lower.endswith(".flac"):
        mime_type = "audio/flac"
    elif audio_path_lower.endswith(".m4a"):
        mime_type = "audio/mp4"
    else:
        # Default to wav if extension is unknown
        mime_type = "audio/wav"

    audio_base64 = encode_base64_content_from_file(audio_path)
    return f"data:{mime_type};base64,{audio_base64}"


def get_system_prompt():
    return {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": (
                    "You are Qwen, a virtual human developed by the Qwen Team, "
                    "Alibaba Group, capable of perceiving auditory and visual inputs, "
                    "as well as generating text and speech."
                ),
            }
        ],
    }


def get_text_query(custom_prompt: Optional[str] = None):
    prompt = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"请将这段文字转换为语音: {custom_prompt}",
            }
        ],
    }
    return prompt


def get_multi_audios_query(audio_path: Optional[str] = None, custom_prompt: Optional[str] = None):
    question = custom_prompt or "Are these two audio clips the same?"
    audio_url = get_audio_url_from_path(audio_path)
    prompt = {
        "role": "user",
        "content": [
            {
                "type": "audio_url",
                "audio_url": {"url": audio_url},
            },
            {
                "type": "audio_url",
                "audio_url": {"url": AudioAsset("winning_call").url},
            },
            {
                "type": "text",
                "text": f"{question}",
            },
        ],
    }
    return prompt


query_map = {
    # "multi_audios": get_multi_audios_query,
    "text": get_text_query,
}


def run_multimodal_generation(args) -> None:
    model_name = "MiMo-Audio-7B-Instruct"
    llm_sampling_params = {
        "temperature": 0.0,  # Deterministic - no randomness
        "top_p": 1.0,  # Disable nucleus sampling
        "top_k": -1,  # Disable top-k sampling
        "max_tokens": 2048,
        "seed": SEED,  # Fixed seed for sampling
        "detokenize": True,
        "repetition_penalty": 1.1,
    }

    code2wav_sampling_params = {
        "temperature": 0.0,  # Deterministic - no randomness
        "top_p": 1.0,  # Disable nucleus sampling
        "top_k": -1,  # Disable top-k sampling
        "max_tokens": 2048,
        "seed": SEED,  # Fixed seed for sampling
        "detokenize": True,
        "repetition_penalty": 1.1,
    }

    sampling_params_list = [
        llm_sampling_params,
        code2wav_sampling_params,
    ]

    # Get paths and custom prompt from args

    audio_path = getattr(args, "audio_path", None)
    custom_prompt = getattr(args, "prompt", None)

    # Get the query function and call it with appropriate parameters
    query_func = query_map[args.query_type]
    if args.query_type == "multi_audios":
        prompt = query_func(audio_path=audio_path, custom_prompt=custom_prompt)
    elif args.query_type == "text":
        prompt = query_func(custom_prompt=custom_prompt)
    else:
        prompt = query_func()

    extra_body = {
        "sampling_params_list": sampling_params_list,
    }

    chat_completion = client.chat.completions.create(
        messages=[
            get_system_prompt(),
            prompt,
        ],
        model=model_name,
        extra_body=extra_body,
    )

    count = 0
    for choice in chat_completion.choices:
        if choice.message.audio:
            audio_data = base64.b64decode(choice.message.audio.data)
            audio_file_path = f"audio_{count}.wav"
            with open(audio_file_path, "wb") as f:
                f.write(audio_data)
            print(f"Audio saved to {audio_file_path}")
            count += 1
        elif choice.message.content:
            print("Chat completion output from text:", choice.message.content)


def parse_args():
    parser = FlexibleArgumentParser(description="Demo on using vLLM for offline inference with audio language models")
    parser.add_argument(
        "--query-type",
        "-q",
        type=str,
        default="text",
        choices=query_map.keys(),
        help="Query type.",
    )
    parser.add_argument(
        "--audio-path",
        "-a",
        type=str,
        default=None,
        help="Path to local audio file or URL. If not provided and query-type uses audio, uses default audio URL.",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        default='我是小米研发的mimo-audio大模型',
        help="Custom text prompt/question to use instead of the default prompt for the selected query type.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_multimodal_generation(args)
