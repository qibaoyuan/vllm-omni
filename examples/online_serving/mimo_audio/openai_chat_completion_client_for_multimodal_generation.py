import base64
import json
import os
from typing import Any

import requests
from openai import OpenAI
from vllm.assets.audio import AudioAsset
from vllm.utils.argparse_utils import FlexibleArgumentParser

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8091/v1"

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


def get_video_url_from_path(video_path: str | None) -> str:
    """Convert a video path (local file or URL) to a video URL format for the API.

    If video_path is None or empty, returns the default URL.
    If video_path is a local file path, encodes it to base64 data URL.
    If video_path is a URL, returns it as-is.
    """
    if not video_path:
        # Default video URL
        return "https://huggingface.co/datasets/raushan-testing-hf/videos-test/resolve/main/sample_demo_1.mp4"

    # Check if it's a URL (starts with http:// or https://)
    if video_path.startswith(("http://", "https://")):
        return video_path

    # Otherwise, treat it as a local file path
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Detect video MIME type from file extension
    video_path_lower = video_path.lower()
    if video_path_lower.endswith(".mp4"):
        mime_type = "video/mp4"
    elif video_path_lower.endswith(".webm"):
        mime_type = "video/webm"
    elif video_path_lower.endswith(".mov"):
        mime_type = "video/quicktime"
    elif video_path_lower.endswith(".avi"):
        mime_type = "video/x-msvideo"
    elif video_path_lower.endswith(".mkv"):
        mime_type = "video/x-matroska"
    else:
        # Default to mp4 if extension is unknown
        mime_type = "video/mp4"

    video_base64 = encode_base64_content_from_file(video_path)
    return f"data:{mime_type};base64,{video_base64}"


def get_image_url_from_path(image_path: str | None) -> str:
    """Convert an image path (local file or URL) to an image URL format for the API.

    If image_path is None or empty, returns the default URL.
    If image_path is a local file path, encodes it to base64 data URL.
    If image_path is a URL, returns it as-is.
    """
    if not image_path:
        # Default image URL
        return "https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/cherry_blossom.jpg"

    # Check if it's a URL (starts with http:// or https://)
    if image_path.startswith(("http://", "https://")):
        return image_path

    # Otherwise, treat it as a local file path
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Detect image MIME type from file extension
    image_path_lower = image_path.lower()
    if image_path_lower.endswith((".jpg", ".jpeg")):
        mime_type = "image/jpeg"
    elif image_path_lower.endswith(".png"):
        mime_type = "image/png"
    elif image_path_lower.endswith(".gif"):
        mime_type = "image/gif"
    elif image_path_lower.endswith(".webp"):
        mime_type = "image/webp"
    else:
        # Default to jpeg if extension is unknown
        mime_type = "image/jpeg"

    image_base64 = encode_base64_content_from_file(image_path)
    return f"data:{mime_type};base64,{image_base64}"


def get_audio_url_from_path(audio_path: str | None) -> str:
    """Convert an audio path (local file or URL) to an audio URL format for the API.

    If audio_path is None or empty, returns the default URL.
    If audio_path is already a base64 data URL (starts with "data:"), returns it as-is.
    If audio_path is a URL (starts with http:// or https://), returns it as-is.
    If audio_path is a local file path, encodes it to base64 data URL.
    """
    if not audio_path:
        # Default audio URL
        return AudioAsset("mary_had_lamb").url

    # Check if it's already a base64 data URL
    if audio_path.startswith("data:"):
        return audio_path

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


def load_messages_from_json(message_json_path: str) -> dict[str, Any]:
    """Load messages from a JSON file."""
    with open(message_json_path, encoding="utf-8") as f:
        data = json.load(f)
    return data


def process_audio_url_in_content(content: dict[str, Any]) -> dict[str, Any]:
    """Process audio_url in content, handling both file paths and base64 URLs."""
    if content.get("type") == "audio_url":
        audio_url = content.get("audio_url", {}).get("url")
        if audio_url:
            # Process the audio URL (handles base64, file paths, and URLs)
            processed_url = get_audio_url_from_path(audio_url)
            content = content.copy()
            content["audio_url"] = {"url": processed_url}
    return content


def get_system_prompt(message_json_path: str | None = None):
    """Get system prompt, optionally from message.json file."""
    if message_json_path and os.path.exists(message_json_path):
        data = load_messages_from_json(message_json_path)
        messages = data.get("messages", [])
        # Find the first system message
        for msg in messages:
            if msg.get("role") == "system":
                # Process audio URLs in the content
                processed_content = []
                for content_item in msg.get("content", []):
                    processed_item = process_audio_url_in_content(content_item)
                    processed_content.append(processed_item)
                return {
                    "role": "system",
                    "content": processed_content,
                }


def get_text_query(custom_prompt: str | None = None):
    question = f"请将这段文字转换为语音: {custom_prompt}"
    prompt = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"{question}",
            }
        ],
    }
    return prompt


def get_multi_audios_query(
    audio_path: str | None = None,
    custom_prompt: str | None = None,
    message_json_path: str | None = None,
) -> list[dict[str, Any]] | dict[str, Any]:
    """Get multi-audios query, optionally from message.json file."""
    if message_json_path and os.path.exists(message_json_path):
        data = load_messages_from_json(message_json_path)
        messages = data.get("messages", [])

        # Skip the first system role and return all other messages
        rebuild_prompt_messages = []
        skipped_first_system = False

        for msg in messages:
            if msg.get("role") == "system" and not skipped_first_system:
                skipped_first_system = True
                continue

            # Process all content items in the message
            processed_content = []
            for content_item in msg.get("content", []):
                processed_item = process_audio_url_in_content(content_item)
                processed_content.append(processed_item)

            rebuild_prompt_messages.append(
                {
                    "role": msg.get("role"),
                    "content": processed_content,
                }
            )

        return rebuild_prompt_messages

    # Original behavior when message_json_path is not provided
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
    "multi_audios": get_multi_audios_query,
    "text": get_text_query,
}


def run_multimodal_generation(args) -> None:
    model_name = "MiMo-Audio-7B-Instruct"
    thinker_sampling_params = {
        "temperature": 0.0,  # Deterministic - no randomness
        "top_p": 1.0,  # Disable nucleus sampling
        "top_k": -1,  # Disable top-k sampling
        "max_tokens": 3200,
        "seed": SEED,  # Fixed seed for sampling
        "detokenize": True,
        "repetition_penalty": 1.1,
    }
    code2wav_sampling_params = {
        "temperature": 0.0,  # Deterministic - no randomness
        "top_p": 1.0,  # Disable nucleus sampling
        "top_k": -1,  # Disable top-k sampling
        "max_tokens": 4096 * 16,
        "seed": SEED,  # Fixed seed for sampling
        "detokenize": True,
        "repetition_penalty": 1.1,
    }

    sampling_params_list = [
        thinker_sampling_params,
        code2wav_sampling_params,
    ]

    # Get paths and custom prompt from args
    audio_path = getattr(args, "audio_path", None)
    custom_prompt = getattr(args, "prompt", None)
    message_json_path = getattr(args, "message_json", None)
    output_audio_path = getattr(args, "output_audio_path", None)

    # Get the query function and call it with appropriate parameters
    query_func = query_map[args.query_type]
    if args.query_type == "multi_audios":
        prompt = query_func(audio_path=audio_path, custom_prompt=custom_prompt, message_json_path=message_json_path)
    elif args.query_type == "text":
        prompt = query_func(custom_prompt=custom_prompt)
    else:
        prompt = query_func()

    extra_body = {
        "sampling_params_list": sampling_params_list
        # Optional, it has a default setting in stage_configs of the corresponding model.
    }

    # Build messages list
    if args.query_type == "multi_audios" and isinstance(prompt, list):
        messages = [get_system_prompt(message_json_path=message_json_path)] + prompt
    elif args.query_type == "text":
        messages = [prompt]

    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model_name,
        extra_body=extra_body,
    )

    count = 0
    for choice in chat_completion.choices:
        if choice.message.audio:
            audio_data = base64.b64decode(choice.message.audio.data)
            audio_file_path = f"{output_audio_path}/{args.query_type}/audio_{count}.wav"
            os.makedirs(os.path.dirname(audio_file_path), exist_ok=True)
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
        default="multi_audios",
        choices=query_map.keys(),
        help="Query type.",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        default="你在哪儿？",
        help="Custom text prompt/question to use instead of the default prompt for the selected query type.",
    )
    parser.add_argument(
        "--message-json",
        "-m",
        type=str,
        default="../../offline_inference/mimo_audio/message_base64_wav.json",
        help="Path to message.json file containing conversation history. When provided, "
        "system prompt and multi_audios query will be loaded from this file.",
    )
    parser.add_argument(
        "--output-audio-path",
        "-o",
        type=str,
        default="./",
        help="Path to save the generated audio files.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_multimodal_generation(args)
