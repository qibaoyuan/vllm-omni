# mimo-audio demo, offline
```BASH

# It is mandatory due to its specialized architecture.
export MIMO_AUDIO_TOKENIZER_PATH="XiaomiMiMo/MiMo-Audio-Tokenizer"

python3 -u end2end.py \
--stage-configs-path vllm_omni/model_executor/stage_configs/mimo_audio.yaml \
--model XiaomiMiMo/MiMo-Audio-7B-Instruct  \
--query-type tts_sft
```
