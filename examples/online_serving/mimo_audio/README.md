# Online serving Example of vLLM-Omni for MiMo-Audio

## 🛠️ Installation

Please refer to [README.md](../../../README.md)

## Run examples (MiMo-Audio)

### Launch the Server
```bash
export MODEL_PATH= "./MiMo-Audio-7B-Instruct"
export STAGE_CONFIGS_PATH="./model_executor/stage_configs/mimo_audio/mimo_audio.yaml"
export CUDA_LAUNCH_BLOCKING=1
export MIMO_AUDIO_TOKENIZER_PATH="./MiMo-Audio-Tokenizer"
export MIMO_AUDIO_TOKENIZER_CONFIG_PATH="./MiMo-Audio-Tokenizer"
export MIMO_AUDIO_ECHO_CODES="false"
```


```bash
vllm-omni serve  ${MODEL_PATH} --omni \
--served-model-name "MiMo-Audio-7B-Instruct"  \
--port 8091 --stage-configs-path ${STAGE_CONFIGS_PATH}
```

If you have custom stage configs file, launch the server with command below
```bash
vllm serve Qwen/MiMo-Audio-7B --omni --port 8091 --stage-configs-path ${STAGE_CONFIGS_PATH}
```

### Send Multi-modal Request

Get into the example folder
```bash
cd examples/online_serving/mimo_audio
```

####  Send request via python

```bash
python openai_chat_completion_client_for_multimodal_generation.py --query-type mixed_modalities
```

The Python client supports the following command-line arguments:

- `--query-type` (or `-q`): Query type (default: `mixed_modalities`)
  - Options: `mixed_modalities`, `use_audio_in_video`, `multi_audios`, `text`
- `--video-path` (or `-v`): Path to local video file or URL
  - If not provided and query-type uses video, uses default video URL
  - Supports local file paths (automatically encoded to base64) or HTTP/HTTPS URLs
  - Example: `--video-path /path/to/video.mp4` or `--video-path https://example.com/video.mp4`
- `--image-path` (or `-i`): Path to local image file or URL
  - If not provided and query-type uses image, uses default image URL
  - Supports local file paths (automatically encoded to base64) or HTTP/HTTPS URLs
  - Supports common image formats: JPEG, PNG, GIF, WebP
  - Example: `--image-path /path/to/image.jpg` or `--image-path https://example.com/image.png`
- `--audio-path` (or `-a`): Path to local audio file or URL
  - If not provided and query-type uses audio, uses default audio URL
  - Supports local file paths (automatically encoded to base64) or HTTP/HTTPS URLs
  - Supports common audio formats: MP3, WAV, OGG, FLAC, M4A
  - Example: `--audio-path /path/to/audio.wav` or `--audio-path https://example.com/audio.mp3`
- `--prompt` (or `-p`): Custom text prompt/question
  - If not provided, uses default prompt for the selected query type
  - Example: `--prompt "What are the main activities shown in this video?"`


For example, to use mixed modalities with all local files:

```bash
python openai_chat_completion_client_for_multimodal_generation.py \
    --query-type mixed_modalities \
    --video-path /path/to/your/video.mp4 \
    --image-path /path/to/your/image.jpg \
    --audio-path /path/to/your/audio.wav \
    --prompt "Analyze all the media content and provide a comprehensive summary."
```

 