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
export MIMO_AUDIO_TOKENIZER_PATH="${MODEL_DIR}/MiMo-Audio-Tokenizer"

python3 -u end2end.py \
--stage-configs-path vllm_omni/model_executor/stage_configs/mimo_audio.yaml \
--model MiMo-Audio-7B-Instruct  \
--query-type tts_sft
```
