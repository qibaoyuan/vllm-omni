# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This example shows how to use vLLM for running offline inference
with the correct prompt format on MiMo-Audio-Omni.
"""

import json
import os
from typing import NamedTuple

import soundfile as sf
from message_convert import (
    get_audio_data,
    get_audio_understanding_sft_prompt,
    get_s2t_dialogue_sft_multiturn_prompt,
    get_spoken_dialogue_sft_multiturn_prompt,
    get_text_dialogue_sft_multiturn_prompt,
    get_tts_sft_prompt,
    to_prompt,
)
from vllm import SamplingParams
from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniTokensPrompt

SEED = 42
MAX_CODE2WAV_TOKENS = 18192


class QueryResult(NamedTuple):
    inputs: dict
    limit_mm_per_prompt: dict[str, int]


def get_codes_query_from_json(codes_path: str) -> QueryResult:
    with open(codes_path, encoding="utf-8") as f:
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


def get_tts_sft(
    text="你好！请简单介绍一下你自己。",
    instruct=None,
    read_text_only=True,
    prompt_speech=None,
    audio_list=None,
):
    res = get_tts_sft_prompt(
        text,
        instruct=instruct,
        read_text_only=read_text_only,
        prompt_speech=prompt_speech,
    )

    prompt = to_prompt(res)
    final_prompt = {
        "prompt": prompt,
    }
    if audio_list is not None:
        final_prompt.update(
            {
                "multi_modal_data": {
                    "audio": audio_list,
                },
            }
        )
    return final_prompt


def get_audio_understanding_sft(audio_path, text="", thinking=False, use_sostm=False):
    audio_list = []
    audio_list.append(get_audio_data(audio_path))
    res = get_audio_understanding_sft_prompt(
        input_speech=audio_path, input_text=text, thinking=thinking, use_sostm=use_sostm
    )
    prompt = to_prompt(res)
    final_prompt = {
        "prompt": prompt,
        "multi_modal_data": {
            "audio": audio_list,
        },
    }
    return final_prompt


def get_spoken_dialogue_sft_multiturn(message_list, system_prompt=None, ref_audio_path=None, audio_list=None):
    res = get_spoken_dialogue_sft_multiturn_prompt(
        message_list, system_prompt=system_prompt, prompt_speech=ref_audio_path
    )
    prompt = to_prompt(res)
    final_prompt = {
        "prompt": prompt,
        "multi_modal_data": {
            "audio": audio_list,
        },
    }
    return final_prompt


def get_speech2text_dialogue_sft_multiturn(message_list, thinking=False, audio_list=None):
    res = get_s2t_dialogue_sft_multiturn_prompt(
        message_list,
        thinking=thinking,
    )
    prompt = to_prompt(res)
    final_prompt = {
        "prompt": prompt,
        "multi_modal_data": {
            "audio": audio_list,
        },
    }
    return final_prompt


def get_text_dialogue_sft_multiturn(
    message_list,
):
    res = get_text_dialogue_sft_multiturn_prompt(
        message_list,
    )
    prompt = to_prompt(res)
    final_prompt = {
        "prompt": prompt,
    }
    return final_prompt


query_map = {
    "tts_sft": get_tts_sft,
    "tts_sft_with_instruct": get_tts_sft,
    "tts_sft_with_audio": get_tts_sft,
    "tts_sft_with_natural_instruction": get_tts_sft,
    "audio_trancribing_sft": get_audio_understanding_sft,
    "audio_understanding_sft": get_audio_understanding_sft,
    "audio_understanding_sft_with_thinking": get_audio_understanding_sft,
    "spoken_dialogue_sft_multiturn": get_spoken_dialogue_sft_multiturn,
    "speech2text_dialogue_sft_multiturn": get_speech2text_dialogue_sft_multiturn,
    "text_dialogue_sft_multiturn": get_text_dialogue_sft_multiturn,
}


def main(args):
    model_name = args.model_name

    # Get paths from args
    text = getattr(args, "text", None)
    audio_path = getattr(args, "audio_path", None)

    instruct = getattr(args, "instruct", None)

    # Get the query function and call it with appropriate parameters
    query_func = query_map[args.query_type]

    omni_llm = Omni(
        model=model_name,
        stage_configs_path=args.stage_configs_path,
        log_stats=args.enable_stats,
        log_file=("omni_llm_pipeline.log" if args.enable_stats else None),
        init_sleep_seconds=args.init_sleep_seconds,
        batch_timeout=args.batch_timeout,
        init_timeout=args.init_timeout,
        shm_threshold_bytes=args.shm_threshold_bytes,
    )

    thinker_sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=50,
        max_tokens=1024,
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

    # Build query result based on query type
    # Notice: The audio files used in this example are available at: https://github.com/XiaomiMiMo/MiMo-Audio/tree/main/examples
    if args.query_type == "tts_sft":
        # python3 -u end2end.py --stage-configs-path ${config_file} --model ${MODEL_PATH}  --query-type tts_sft
        """"
        lines ['Prompt:\n', '<|im_start|>user\n请将这段文字转换为语音: 今天天气真好<|im_end|>\n<|im_start|>assistant\n<|sostm|>\n', 'vllm_text_output:\n', '今天天气真好\n']
        Request ID: 0_f96f7bcd-a861-4fa0-a1f4-f804d8202be, Text saved to ./output_audio/tts_sft/0_f96f7bcd-a861-4fa0-a1f4-f804d8202be.txt
        Request ID: 0_f96f7bcd-a861-4fa0-a1f4-f804d8202be, Saved audio to ./output_audio/tts_sft/0_f96f7bcd-a861-4fa0-a1f4-f804d8202be.wav
        """
        query_result = query_func(text=text, read_text_only=True)
    elif args.query_type == "tts_sft_with_instruct":
        # python3 -u end2end.py --stage-configs-path ${config_file} --model ${MODEL_PATH}  --query-type tts_sft_with_instruct --instruct "用小孩子的声音开心的说"
        """
        lines ['Prompt:\n', '<|im_start|>system\nYou need to generate speech based on the specified style instructions and text content.<|im_end|>\n<|im_start|>user\n请将这段文字转换为语音: 今天天气真好(用小孩子的声音开心的说)<|im_end|>\n<|im_start|>assistant\n<think>\n\n', 'vllm_text_output:\n', '好的，这次是要模仿一个小孩子说话。指令很明确，“小孩子”、“开心”。那我的声音就要提得高一点，音色要亮一些，听起来天真无邪。语速嘛，不能太快，得有点慢悠悠、一字一顿的感觉，就像小朋友在认真地表达自己的发现一样。“今天天气真好”，这句话本身就挺阳光的，所以我要带着那种发自内心的喜悦感去说，句尾可以稍微上扬一点点，显得更活泼可爱。\n</think>\n今天天气真好\n']
        Request ID: 0_f6885005-c769-47ef-93fb-f22093fb42a6, Text saved to ./output_audio/tts_sft_with_instruct/0_f6885005-c769-47ef-93fb-f22093fb42a6.txt
        Request ID: 0_f6885005-c769-47ef-93fb-f22093fb42a6, Saved audio to ./output_audio/tts_sft_with_instruct/0_f6885005-c769-47ef-93fb-f22093fb42a6.wav
        """
        query_result = query_func(text=text, instruct=instruct, read_text_only=True)
    elif args.query_type == "tts_sft_with_audio":
        # python3 -u end2end.py --stage-configs-path ${config_file} --model ${MODEL_PATH}  --query-type tts_sft_with_audio --audio_path "./spoken_dialogue_assistant_turn_1.wav"
        audio_list = [get_audio_data(audio_path)]
        query_result = query_func(text=text, read_text_only=True, prompt_speech=audio_path, audio_list=audio_list)
    elif args.query_type == "tts_sft_with_natural_instruction":
        # python3 -u end2end.py --stage-configs-path ${config_file} --model ${MODEL_PATH}  --query-type tts_sft_with_natural_instruction --text "用气喘吁吁的年轻男性声音说：我跑不动了，你等等我！"
        """
        lines ['Prompt:\n', '<|im_start|>system\nYou need to generate speech based on the specified style instructions and text content.<|im_end|>\n<|im_start|>user\n用气喘吁吁的年轻男性声音说：我跑不动了，你等等我！<|im_end|>\n<|im_start|>assistant\n<think>\n\n', 'vllm_text_output:\n', '好的，这个要求很明确。首先是个年轻男性的声音，然后关键是“气喘吁吁”。这说明他刚经过剧烈运动，体力不支。所以我的声音里得带上明显的喘息声，尤其是在句子的开头和结尾。语速要放慢，断断续续的，好像每说一个字都很费劲。“我跑不动了”这里可以表现出一种无力感，音调稍微有点上扬但又很快落下去。到了“你等等我！”的时候，情绪要更急切一点，因为是在求人，但身体状态还是跟不上，所以这种急切是虚弱中的急切。嗯，重点就是把那种上气不接下气的感觉给做出来。\n</think>\n我跑不动了，你等等我！\n']
        Request ID: 0_7c161be3-96d3-46b1-9981-a59fa1ae81e5, Text saved to ./output_audio/tts_sft_with_natural_instruction/0_7c161be3-96d3-46b1-9981-a59fa1ae81e5.txt
        Request ID: 0_7c161be3-96d3-46b1-9981-a59fa1ae81e5, Saved audio to ./output_audio/tts_sft_with_natural_instruction/0_7c161be3-96d3-46b1-9981-a59fa1ae81e5.wav        """
        query_result = query_func(text=text, read_text_only=False)
    elif args.query_type == "audio_trancribing_sft":
        # python3 -u end2end.py --stage-configs-path ${config_file} --model ${MODEL_PATH}  --query-type audio_trancribing_sft --audio_path "./spoken_dialogue_assistant_turn_1.wav"
        """
        lines ['Prompt:\n', '<|im_start|>user\n<|sosp|><|empty|><|eosp|>Please transcribe this audio and repeat it once.<|im_end|>\n<|im_start|>assistant\n<|sostm|>\n', 'vllm_text_output:\n', '今天天气如何？\n']
        Request ID: 0_a9c107ec-7a4e-44fe-a304-d3ee6e1dcca6, Text saved to ./output_audio/audio_trancribe_sft/0_a9c107ec-7a4e-44fe-a304-d3ee6e1dcca6.txt
        Request ID: 0_a9c107ec-7a4e-44fe-a304-d3ee6e1dcca6, Audio saved to ./output_audio/audio_trancribe_sft/0_a9c107ec-7a4e-44fe-a304-d3ee6e1dcca6.wav
        """
        audio_path = "spoken_dialogue_assistant_turn_1.wav"
        text = "Please transcribe this audio and repeat it once."
        query_result = query_func(text=text, audio_path=audio_path, use_sostm=True)
    elif args.query_type == "audio_understanding_sft":
        # python3 -u end2end.py --stage-configs-path ${config_file} --model ${MODEL_PATH}  --query-type audio_understanding_sft --text "Summarize the audio." --audio_path "./spoken_dialogue_assistant_turn_1.wav"
        """
        lines ['Prompt:\n', '<|im_start|>user\n<|sosp|><|empty|><|eosp|>Summarize the audio.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n', 'vllm_text_output:\n', "The speaker provides several ways to check today's weather, including using built-in phone features (like Apple Weather), professional services (such as AccuWeather or China Meteoweb), and search engines (Google or Baidu). They also mention that while they can analyze historical weather trends for a specific city, real-time data must be obtained through official sources. The speaker invites the listener to share their location for further assistance.\n"]
        Request ID: 0_0e3dd143-99fd-4f37-8d0c-f78859e76665, Text saved to ./output_audio/audio_understanding_sft/0_0e3dd143-99fd-4f37-8d0c-f78859e76665.txt
        """
        query_result = query_func(text=text, audio_path=audio_path)
    elif args.query_type == "audio_understanding_sft_with_thinking":
        # python3 -u end2end.py --stage-configs-path ${config_file} --model ${MODEL_PATH}  --query-type audio_understanding_sft_with_thinking --text "Summarize the audio." --audio_path "./spoken_dialogue_assistant_turn_1.wav"
        """
        lines ['Prompt:\n', '<|im_start|>user\n<|sosp|><|empty|><|eosp|>Summarize the audio.<|im_end|>\n<|im_start|>assistant\n<think>\n\n', 'vllm_text_output:\n', 'The user wants a summary of the provided audio transcript.\n\n1.  **Identify the core topic:** The main subject is how to check today\'s weather forecast.\n2.  **Recognize the key constraint:** The speaker explicitly states they cannot access real-time data themselves ("我没办法获取实时的天气信息").\n3.  **List the methods suggested:** The speaker provides several alternative ways for the listener to find the weather information:\n    *   Using built-in phone features (specifically mentioning Apple\'s Weather app and checking in "系统设置" - system settings).\n    *   Using professional weather services, giving examples like AccuWeather, Weather.com, and Chinese services like 中最天气网 (zhongzuiweather.com) and 梅花天气 (mehua weather).\n    *   Using search engines (Google or Baidu) by searching for "[city name] + 天气" ([城市名] + weather).\n4.  **Note any additional offers or conditions:** The speaker offers to help analyze historical weather trends if the listener provides their city name, but reiterates that current data must be obtained from official sources.\n5.  **Synthesize into a concise summary:** Combine these points into a clear and brief paragraph. Start with the main point (the inability to get live data), then list the recommended methods, and finally include the offer for historical analysis. This structure accurately reflects the content and flow of the original audio.\n</think>\nThe speaker explains that they cannot provide real-time weather information directly. Instead, they suggest several methods for the listener to check the current weather:\n\n*   Use the built-in weather application on your smartphone (like Apple\'s Weather app).\n*   Visit professional weather websites such as AccuWeather, Weather.com, 中最天气网, or 梅花天气.\n*   Search for your city followed by the word "天气" (weather) using Google or Baidu.\n\nThe speaker also offers to help analyze historical weather trends for a specific city if the listener provides its name, but emphasizes that all current data should be obtained through official channels.\n']
        Request ID: 0_7899d15a-1d5c-439a-9888-dd8c807b8165, Text saved to ./output_audio/audio_understanding_sft_with_thinking/0_7899d15a-1d5c-439a-9888-dd8c807b8165.txt
        """
        query_result = query_func(text=text, audio_path=audio_path, thinking=True)
    elif args.query_type == "spoken_dialogue_sft_multiturn":
        # python3 -u end2end.py --stage-configs-path ${config_file} --model ${MODEL_PATH}  --query-type spoken_dialogue_sft_multiturn  --audio_path "./prompt_speech_zh_m.wav"
        """
        lines ['Prompt:\n', '<|im_start|>system\nYour Voice Should be:<|sosp|><|empty|><|eosp|><|im_end|>\n<|im_start|>user\n<|sosp|><|empty|><|eosp|><|im_end|>\n<|im_start|>assistant\n我没办法获取实时的天气信息。不过呢，你可以试试几个方法来查看今天的天气。首先，你可以用手机自带的天气功能，比如苹果手机的天气应用，或者直接在系统设置里查看。其次，你也可以用一些专业的天气服务，像是国外的AccuWeather、Weather.com，或者国内的中国天气网、墨迹天气等等。再有就是，你还可以在谷歌或者百度里直接搜索你所在的城市加上天气这两个字。如果你能告诉我你所在的城市，我也可以帮你分析一下历史天气趋势，不过最新的数据还是需要你通过官方渠道去获取哦。<|sosp|><|empty|><|eosp|><|im_end|>\n<|im_start|>user\n<|sosp|><|empty|><|eosp|><|im_end|>\n<|im_start|>assistant\n<|sostm|>\n', 'vllm_text_output:\n', '好的，为您查询到北京当前的天气情况是这样的：首先是温度，现在是零下3摄氏度，体感非常寒冷。天气状况是晴天，湿度百分之四十五，空气质量指数是120，属于轻度污染，主要污染物是PM2.5。风向是西北风，风力不大，在每秒2到4米之间。气压是一千零二十二百帕。今天白天的最高气温是零上2摄氏度，夜间最低气温会降到零下6摄氏度。另外还有两个小贴士给您：第一，因为温差大，请注意防寒保暖，特别是要保护好耳朵和手指这些露在外面的皮肤。第二，目前空气质量不太好，建议您减少户外活动的时间，如果需要用口罩的话，最好选择N95级别的。如果您想查其他城市或者更详细的信息，可以告诉我具体的城市名或者日期，我会帮您调整的！\n']
        Request ID: 0_a2b4a232-2b86-442f-8fbb-d9b8fd198b00, Text saved to ./output_audio/spoken_dialogue_sft_multiturn/0_a2b4a232-2b86-442f-8fbb-d9b8fd198b00.txt
        Request ID: 0_a2b4a232-2b86-442f-8fbb-d9b8fd198b00, Saved audio to ./output_audio/spoken_dialogue_sft_multiturn/0_a2b4a232-2b86-442f-8fbb-d9b8fd198b00.wav
        """
        first_turn_text_response = "我没办法获取实时的天气信息。不过呢，你可以试试几个方法来查看今天的天气。首先，你可以用手机自带的天气功能，比如苹果手机的天气应用，或者直接在系统设置里查看。其次，你也可以用一些专业的天气服务，像是国外的AccuWeather、Weather.com，或者国内的中国天气网、墨迹天气等等。再有就是，你还可以在谷歌或者百度里直接搜索你所在的城市加上天气这两个字。如果你能告诉我你所在的城市，我也可以帮你分析一下历史天气趋势，不过最新的数据还是需要你通过官方渠道去获取哦。"
        audio_list = []
        s1_audio_path = "今天天气如何.mp3"
        s2_audio_path = "spoken_dialogue_assistant_turn_1.wav"
        s3_audio_path = "北京.mp3"
        audio_list.append(get_audio_data(audio_path))
        audio_list.append(get_audio_data(s1_audio_path))
        audio_list.append(get_audio_data(s2_audio_path))
        audio_list.append(get_audio_data(s3_audio_path))

        message_list = [
            {"role": "user", "content": s1_audio_path},
            {"role": "assistant", "content": {"text": first_turn_text_response, "audio": s2_audio_path}},
            {"role": "user", "content": s3_audio_path},
        ]
        query_result = query_func(message_list, system_prompt=None, ref_audio_path=audio_path, audio_list=audio_list)
    elif args.query_type == "speech2text_dialogue_sft_multiturn":
        # python3 -u end2end.py --stage-configs-path ${config_file_only_llm} --model ${MODEL_PATH}  --query-type speech2text_dialogue_sft_multiturn
        """
        lines ['Prompt:\n', '<|im_start|>user\n<|sosp|><|empty|><|eosp|><|im_end|>\n<|im_start|>assistant\n你好，我没办法获取实时的天气信息。如果你能告诉我你所在的城市，我也可以帮你分析一下历史天气趋势，不过最新的数据还是需要你通过官方渠道去获取哦。<|im_end|>\n<|im_start|>user\n<|sosp|><|empty|><|eosp|><|im_end|>\n<|im_start|>assistant\n<think>\n\n', 'vllm_text_output:\n', '好的，用户问的是北京的天气情况。虽然我无法提供实时数据，但我可以告诉用户查询天气的可靠途径，并根据历史规律给出一些参考建议。\n\n首先，最准确的方法是使用专业的天气应用或网站，比如中国气象局官网、中央电视台天气预报或者像墨迹天气这样的手机应用。这些平台的数据都来自权威机构，更新及时，还能看到具体的温度、湿度、风速和空气质量指数等详细信息。\n\n其次，如果用户想了解长期趋势，我可以分享一些北京的历史气候特点。北京属于温带季风气候，四季分明。春季通常在3月到5月，特点是干燥多风，偶尔有沙尘天气，平均气温从10摄氏度左右逐渐升到25摄氏度以上。夏季是6月到8月，炎热多雨，平均气温在25到30摄氏度之间，7月份最热的时候可能达到35摄氏度以上，而且经常有雷阵雨。秋季是从9月到11月，天气凉爽宜人，平均气温在15到25摄氏度，是旅游的好季节。冬季则是在12月到次年2月，寒冷干燥，平均气温在零下5摄氏度到5摄氏度之间，1月份最冷时可能低至零下15摄氏度，降雪不多但风寒效应明显。\n\n另外，我还得提醒用户注意空气质量。北京的PM2.5指数有时会比较高，尤其是在冬天供暖期间，建议关注AQI指数，必要时佩戴口罩。穿衣方面也要根据实时天气调整，夏天防晒防雨，冬天保暖防风。\n\n最后，如果用户需要更具体的信息，比如未来一周的预报或者某个特定日期的天气，最好还是通过上述专业渠道查询，这样得到的结果才最准确可靠。\n</think>\n关于北京当前的天气，由于我无法访问实时数据，以下是一些实用建议：\n\n### 1️⃣ **推荐查询方式**\n- **官方渠道**：  \n  - 中国气象局官网（[www.nmc.cn](http://www.nmc.cn)）  \n  - 央视《新闻联播》后的天气预报（约晚上7点）\n- **常用APP**：  \n  墨迹天气、彩云天气、AccuWeather（可查看每小时降水概率）\n\n### 2️⃣ **近期典型天气特征**\n- **春秋季**（3-5月/9-11月）：  \n  昼夜温差大（±10℃），需备外套  \n- **夏季**（6-8月）：  \n  高温常达30℃+，午后局部降雨  \n- **冬季**（12-2月）：  \n  平均低温-5℃，雾霾高发期\n\n### 3️⃣ **出行小贴士**\n- 查看实时交通路况（百度地图/高德）  \n- 提前关注空气质量指数（AQI＞150建议减少户外活动）  \n- 若计划爬山（如香山），请确认当日是否封路\n\n建议您通过上述任一渠道快速获取最新信息。如需其他帮助，欢迎随时告知！ 🌦️\n']
        """
        sampling_params_list = [
            thinker_sampling_params,
        ]
        s1_audio_path = "今天天气如何.mp3"
        s2_audio_path = "北京.mp3"
        audio_list = []
        audio_list.append(get_audio_data(s1_audio_path))
        audio_list.append(get_audio_data(s2_audio_path))
        message_list = [
            {"role": "user", "content": s1_audio_path},
            {
                "role": "assistant",
                "content": "你好，我没办法获取实时的天气信息。如果你能告诉我你所在的城市，我也可以帮你分析一下历史天气趋势，不过最新的数据还是需要你通过官方渠道去获取哦。",
            },
            {"role": "user", "content": s2_audio_path},
        ]
        query_result = query_func(message_list, thinking=True, audio_list=audio_list)
    elif args.query_type == "text_dialogue_sft_multiturn":
        # python3 -u end2end.py --stage-configs-path ${config_file_only_llm} --model ${MODEL_PATH}  --query-type text_dialogue_sft_multiturn
        """
        lines ['Prompt:\n', '<|im_start|>user\n可以给我介绍一些中国的旅游景点吗？<|im_end|>\n<|im_start|>assistant\n你好，您想去哪个城市旅游呢？<|im_end|>\n<|im_start|>user\n北京<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n', 'vllm_text_output:\n', '当然！北京作为中国首都，拥有丰富的历史文化和现代景观。以下是一些值得一游的景点推荐：\n\n---\n\n### **1. 故宫（紫禁城）**\n- **特色**：明清两代皇家宫殿，世界最大、保存最完整的木质结构古建筑群。\n- **亮点**：太和殿、珍宝馆、钟表馆；冬季可体验“故宫雪景”。\n- **门票**：旺季60元/人，需提前预约。\n\n### **2. 长城（八达岭/慕田峪）**\n- **八达岭长城**：最经典段落，交通便利，适合初次游览。\n- **慕田峪长城**：风景秀丽，人相对较少，适合拍照。\n- **建议**：清晨或傍晚游览避开人流，穿舒适运动鞋。\n\n### **3. 天安门广场 & 国家博物馆**\n- **天安门广场**：世界上最大的城市广场，可看升旗仪式（需查时间表）。\n- **国家博物馆**：免费开放，展示中华五千年文明。\n\n### **4. 颐和园**\n- **特色**：清代皇家园林，以昆明湖、万寿山为基址，融合江南园林风格。\n- **必看**：长廊彩绘、佛香阁、十七孔桥。\n\n### **5. 北京胡同与四合院**\n- **推荐区域**：\n  - **南锣鼓巷**：文艺小店聚集地，适合年轻人打卡。\n  - **什刹海**：后海酒吧街夜生活，划船赏秋叶。\n  - **杨梅竹斜街**：小众胡同，咖啡馆与文创店。\n\n### **6. 景山公园**\n- **登顶俯瞰**：故宫全景最佳观景点，日落时分尤其美。\n\n### **7. 奥林匹克公园（鸟巢、水立方）**\n- **现代地标**：2008年奥运会场馆，夜晚灯光秀很震撼。\n\n### **8. 西红门野生动物园**\n- **亲子游首选**：可自驾或乘小火车近距离接触动物。\n\n### **9. 玉渊潭公园**\n- **春季樱花**：3月底至4月初樱花盛开，是热门赏樱地。\n\n### **10. 地铁里的文化站**\n- **推荐站点**：东华门（古代皇城）、鼓楼大街（老北京风情）、西直门（交通枢纽）。\n\n---\n\n### **旅行小贴士**\n- **交通**：地铁覆盖广，下载“亿通行”APP扫码乘车；共享单车方便短途。\n- **美食**：烤鸭（四季民福、大董）、炸酱面、豆汁儿（尝试前做好心理准备）。\n- **季节**：春秋最佳（3-5月、9-11月），夏季炎热，冬季寒冷但可滑雪（如南山滑雪场）。\n\n如果需要更具体的路线规划或深度体验建议，可以告诉我你的兴趣偏好哦！ 😊\n']
        Request ID: 0_32f2ec15-accc-4d78-bfe0-c61788e56299, Text saved to ./output_audio/text_dialogue_sft_multiturn/0_32f2ec15-accc-4d78-bfe0-c61788e56299.txt
        """
        sampling_params_list = [
            thinker_sampling_params,
        ]
        message_list = [
            {"role": "user", "content": "可以给我介绍一些中国的旅游景点吗？"},
            {"role": "assistant", "content": "你好，您想去哪个城市旅游呢？"},
            {"role": "user", "content": "北京"},
        ]
        query_result = query_func(message_list=message_list)
    else:
        raise ValueError(f"Invalid query type: {args.query_type}")

    prompts = [query_result for _ in range(args.num_prompts)]

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
                print(f"Request ID: {request_id}, Text saved to {out_txt}\n")
        elif stage_outputs.final_output_type == "audio":
            for output in stage_outputs.request_output:
                request_id = output.request_id
                audio_tensor = output.multimodal_output.get("audio")

                if audio_tensor is None:
                    continue

                output_wav = os.path.join(output_dir, f"{request_id}.wav")

                # Convert to numpy array and ensure correct format
                audio_numpy = audio_tensor.float().detach().cpu().numpy()

                # Ensure audio is 1D (flatten if needed)
                if audio_numpy.ndim > 1:
                    audio_numpy = audio_numpy.flatten()

                # Save audio file with explicit WAV format
                sf.write(output_wav, audio_numpy, samplerate=24000, format="WAV")
                print(f"Request ID: {request_id}, Audio saved to {output_wav}")


def parse_args():
    parser = FlexibleArgumentParser(description="Demo on using vLLM for offline inference with audio language models")
    parser.add_argument(
        "--model-name",
        "-m",
        type=str,
        default="XiaomiMiMo/MiMo-Audio-7B-Instruct",
        help="Backbone LLM path.",
    )
    parser.add_argument(
        "--text",
        "-t",
        type=str,
        default="今天天气真好",
        help="input text",
    )
    parser.add_argument(
        "--query-type",
        "-q",
        type=str,
        default="tts_sft",
        choices=query_map.keys(),
        help="Query type.",
    )
    parser.add_argument(
        "--audio-path",
        "-a",
        type=str,
        default=None,
        help="Path to local audio file. If not provided, uses default audio asset.",
    )
    parser.add_argument(
        "--instruct",
        type=str,
        default=None,
        help="instruct",
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
        help="Output audio wav directory.",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1,
        help="Number of prompts to generate.",
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
        default="../../../model_executor/stage_configs/mimo_audio.yaml",
        help="Path to a stage configs file.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
