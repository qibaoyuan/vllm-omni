# 2 cards 4090/48g, cuda:12.9

Wed Jan  7 17:01:31 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 575.51.03              Driver Version: 575.51.03      CUDA Version: 12.9     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4090        On  |   00000000:0C:00.0 Off |                    0 |
| 63%   31C    P8             32W /  450W |    5686MiB /  46068MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA GeForce RTX 4090        On  |   00000000:8A:00.0 Off |                    0 |
| 63%   37C    P2            121W /  450W |   23341MiB /  46068MiB |     35%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

# mimo-audio demo, offline
```BASH
export CODE_DIR="/TO/PATH1/vllm-omni"
export MODEL_DIR="/TO/PATH2"

export PYTHONPATH=${CODE_DIR}:${PYTHONPATH}

cd ${CODE_DIR}/examples/offline_inference/mimo_audio
export MIMO_AUDIO_TOKENIZER_DEVICE="cuda:0"

export CUDA_LAUNCH_BLOCKING=1
export MIMO_AUDIO_TOKENIZER_PATH="${MODEL_DIR}/MiMo-Audio-Tokenizer"
export MIMO_AUDIO_TOKENIZER_CONFIG_PATH=${MIMO_AUDIO_TOKENIZER_PATH}
export MODEL_PATH="${MODEL_DIR}/MiMo-Audio-7B-Instruct"

export config_file="${CODE_DIR}/vllm_omni/model_executor/stage_configs/mimo_audio/mimo_audio.yaml"

python3 -u end2end.py \
--stage-configs-path ${config_file} \
--model ${MODEL_PATH}  \
--query-type multi_audios --message-path  ./message_base64_wav.json
```
