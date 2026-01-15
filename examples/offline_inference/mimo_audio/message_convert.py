import base64
import io
import os
import random
import re
from collections.abc import Callable

import librosa
import numpy as np
import torch
import torchaudio
from process_speechdata import InputSegment, StreamingInputSegment
from torchaudio.transforms import MelSpectrogram

speech_zeroemb_idx = 151667
empty_token = "<|empty|>"
mimo_audio_tokenizer = None
device = "cpu"

# Copyright 2025 Xiaomi Corporation.
asr_zh_templates = [
    "请将这段语音转换为文字",
    "帮我识别这个音频文件中的内容",
    "把这段录音转成文本",
    "请转录这段语音",
    "将音频内容转换成文字格式",
    "识别并转写这段语音",
    "把语音内容写成文字",
    "转录这个音频片段",
    "将这段对话转换为文本",
    "麻烦帮我把这段录音整理成详细的文字记录",
]

asr_en_templates = [
    "Please transcribe this audio file",
    "Convert this speech recording to text",
    "Transcribe the following voice message",
    "Turn this audio into readable text",
    "Please convert the recording to written format",
    "Transcribe what you hear in this audio",
    "Convert this spoken content to text",
    "Please write down what is said in this recording",
    "Transcribe this voice recording",
    "Could you please help me transcribe this important recording?",
    "Would you mind converting this voice message into a readable text format?",
    "I'd really appreciate it if you could turn this audio file into a written document",
]

tts_zh_templates = [
    "请将这段文字转换为语音",
    # "帮我把这个文本读出来",
    # "将这些文字生成音频",
    # "请朗读这段内容",
    # "把这段话转换成语音文件",
    # "生成这段文字的语音版本",
    # "请用语音播报这些内容",
    # "将文本转换为可听的音频",
    # "帮我朗读这段文字",
    # "把这些内容念出来",
]

tts_en_templates = [
    "Please convert this text to speech",
    "Turn this writing into audio",
    "Generate speech from this text",
    "Read this content out loud",
    "Convert these words to voice",
    "Create an audio version of this text",
    "Please vocalize this content",
    "Turn this text into audible format",
    "Help me convert this writing to speech",
    "Make this text into spoken audio",
]


def detect_language(text):
    if re.search(r"[\u4e00-\u9fff]", text):
        return "zh"
    else:
        return "en"


# ============================================
# 公共辅助函数 - InputSegment 创建
# ============================================


def create_segment(text: str = "", audio=None) -> InputSegment:
    """创建一个标准的 InputSegment，使用默认的 zeroemb 参数"""
    return InputSegment(
        text=text,
        audio=audio,
        speech_zeroemb_idx=speech_zeroemb_idx,
        text_zeroemb_idx=empty_token,
    )


def create_streaming_segment(
    text: str, audio, tokenizer, group_size: int, audio_channels: int
) -> StreamingInputSegment:
    """创建一个 StreamingInputSegment"""
    return StreamingInputSegment(
        text=text,
        audio=audio,
        tokenizer=tokenizer,
        group_size=group_size,
        audio_channels=audio_channels,
        speech_zeroemb_idx=speech_zeroemb_idx,
        text_zeroemb_idx=empty_token,
    )


# ============================================
# 公共辅助函数 - 常用标记 Segment 创建
# ============================================


def create_user_start() -> InputSegment:
    """创建 user 角色开始标记"""
    return create_segment(text="<|im_start|>user\n")


def create_user_end() -> InputSegment:
    """创建角色结束标记"""
    return create_segment(text="<|im_end|>\n")


def create_assistant_start() -> InputSegment:
    """创建 assistant 角色开始标记"""
    return create_segment(text="<|im_start|>assistant\n")


def create_system_start() -> InputSegment:
    """创建 system 角色开始标记"""
    return create_segment(text="<|im_start|>system\n")


def create_thinking_segment(thinking: bool = False) -> InputSegment:
    """创建 thinking 标记，根据 thinking 参数决定是否闭合"""
    if thinking:
        return create_segment(text="<think>\n")
    else:
        return create_segment(text="<think>\n\n</think>\n")


def create_sostm_segment() -> InputSegment:
    """创建流式输出开始标记"""
    return create_segment(text="<|sostm|>")


def create_assistant_start_with_sostm() -> InputSegment:
    """创建带 sostm 的 assistant 开始标记"""
    return create_segment(text="<|im_start|>assistant\n<|sostm|>")


def create_assistant_start_with_think() -> InputSegment:
    """创建带 think 开始的 assistant 标记"""
    return create_segment(text="<|im_start|>assistant\n<think>\n")


# ============================================
# 公共辅助函数 - 复合 Segment 创建
# ============================================


def create_user_turn_with_audio(audio_tokenized, extra_text: str = None) -> list[InputSegment]:
    """创建包含音频的 user turn"""
    segments = [
        create_user_start(),
        create_segment(audio=audio_tokenized),
    ]
    if extra_text:
        segments.append(create_segment(text=extra_text))
    segments.append(create_user_end())
    return segments


def create_user_turn_with_text(text: str) -> list[InputSegment]:
    """创建纯文本的 user turn"""
    return [
        create_user_start(),
        create_segment(text=text),
        create_user_end(),
    ]


def create_system_turn_with_voice_prompt(prompt_text: str, audio_token) -> list[InputSegment]:
    """创建带音色提示的 system turn"""
    return [
        create_system_start(),
        create_segment(text=prompt_text),
        create_segment(text="", audio=audio_token),
        create_user_end(),
    ]


def create_system_turn_text_only(system_text: str) -> list[InputSegment]:
    """创建纯文本的 system turn"""
    return [
        create_system_start(),
        create_segment(text=system_text),
        create_user_end(),
    ]


# ============================================
# 公共辅助函数 - 多轮对话处理
# ============================================


def process_multiturn_messages(
    message_list: list[dict],
    user_processor: Callable[[dict], list[InputSegment]],
    assistant_processor: Callable[[dict], list[InputSegment]],
) -> list[InputSegment]:
    """
    通用的多轮对话消息处理函数

    Args:
        message_list: 消息列表，每个消息包含 'role' 和 'content'
        user_processor: 处理 user 消息的函数
        assistant_processor: 处理 assistant 消息的函数

    Returns:
        处理后的 InputSegment 列表
    """
    lm_prompt = []
    for message in message_list:
        role = message["role"]
        if role == "user":
            lm_prompt.extend(user_processor(message))
        elif role == "assistant":
            lm_prompt.extend(assistant_processor(message))
        else:
            raise ValueError(f"Invalid role: {role}")
    return lm_prompt


def create_text_user_message(message: dict) -> list[InputSegment]:
    """处理纯文本的 user 消息"""
    return [
        create_user_start(),
        create_segment(text=message["content"]),
        create_user_end(),
    ]


def create_text_assistant_message(message: dict) -> list[InputSegment]:
    """处理纯文本的 assistant 消息"""
    return [
        create_assistant_start(),
        create_segment(text=message["content"]),
        create_user_end(),
    ]


def create_audio_user_message(message: dict) -> list[InputSegment]:
    """处理音频的 user 消息"""
    return [
        create_user_start(),
        create_segment(audio=preprocess_input(message["content"])),
        create_user_end(),
    ]


def append_assistant_ending(
    lm_prompt: list[InputSegment], thinking: bool = False, use_sostm: bool = False
) -> list[InputSegment]:
    """
    为 prompt 添加 assistant 结尾

    Args:
        lm_prompt: 现有的 prompt 列表
        thinking: 是否使用开放的 thinking 标记
        use_sostm: 是否使用 sostm 标记（用于语音输出）
    """
    if use_sostm:
        lm_prompt.append(create_assistant_start_with_sostm())
    else:
        lm_prompt.append(create_assistant_start())
        lm_prompt.append(create_thinking_segment(thinking))
    return lm_prompt


def get_asr_sft_prompt(
    input: None | str = None,
):
    """ASR (语音识别) 任务的 prompt 构建"""
    audio_tokenized = preprocess_input(input)
    template = random.choice(asr_zh_templates + asr_en_templates)

    lm_prompt = create_user_turn_with_audio(audio_tokenized, extra_text=template)
    lm_prompt = append_assistant_ending(lm_prompt, thinking=False)
    return lm_prompt


def resample_audio_if_needed(wav_tensor: torch.Tensor, original_sr: int):
    target_sr = 24000
    if original_sr != target_sr:
        wav_tensor = torchaudio.functional.resample(wav_tensor, original_sr, target_sr)
    return wav_tensor


def wav2mel(wav, device="cpu"):
    mel_transform = MelSpectrogram(
        sample_rate=mimo_audio_tokenizer.config.sampling_rate,
        n_fft=mimo_audio_tokenizer.config.nfft,
        hop_length=mimo_audio_tokenizer.config.hop_length,
        win_length=mimo_audio_tokenizer.config.window_size,
        f_min=mimo_audio_tokenizer.config.fmin,
        f_max=mimo_audio_tokenizer.config.fmax,
        n_mels=mimo_audio_tokenizer.config.n_mels,
        power=1.0,
        center=True,
    ).to(device)
    spec = mel_transform(wav[None, :])
    return torch.log(torch.clip(spec, min=1e-7)).squeeze()


def group_by_length(features: torch.Tensor, lengths: torch.Tensor, max_length: int):
    if features.size(0) != lengths.sum().item():
        raise ValueError(f"Feature size mismatch: {features.size(0)} vs {lengths.sum().item()}")

    split_points = []
    current_sum = 0

    for i, seq_len in enumerate(lengths):
        if current_sum + seq_len > max_length and current_sum > 0:
            split_points.append(i)
            current_sum = seq_len.item()
        else:
            current_sum += seq_len.item()

    # Convert split points to group sizes
    group_sizes = []
    prev = 0
    for point in split_points:
        group_sizes.append(point - prev)
        prev = point
    if prev < len(lengths):
        group_sizes.append(len(lengths) - prev)

    len_groups = torch.split(lengths, group_sizes)
    feature_sizes = [group.sum().item() for group in len_groups]
    feature_groups = torch.split(features, feature_sizes)

    return feature_groups, len_groups


def encode_batch(input_features: torch.Tensor, input_lens: torch.Tensor, max_length: int = 256000):
    feature_groups, len_groups = group_by_length(input_features, input_lens, max_length)

    encoded_parts = []
    for features, lengths in zip(feature_groups, len_groups):
        with torch.no_grad():
            codes, _ = mimo_audio_tokenizer.encoder.encode(
                input_features=features.to(device), input_lens=lengths.to(device), return_codes_only=True
            )
            encoded_parts.append(codes)

    return torch.cat(encoded_parts, dim=-1)


def preprocess_input(input: None | str | torch.Tensor = None, device="cpu", audio_channels=4, group_size=8):
    if isinstance(input, torch.Tensor) or (isinstance(input, str) and os.path.isfile(input)):
        return "<|sosp|><|empty|><|eosp|>"
        if isinstance(input, torch.Tensor):
            wav = input
        else:
            wav, sr = torchaudio.load(input)
            if wav.ndim == 2:
                wav = wav.mean(dim=0)
            wav = resample_audio_if_needed(wav, sr)
        wav = wav.to(device)

        if wav.shape[0] < sr:
            wav = torch.cat(
                [
                    wav,
                    torch.zeros(sr - wav.shape[0], device=wav.device, dtype=wav.dtype),
                ],
            )

        mel = wav2mel(wav).transpose(0, 1)  # (seq_len, n_mels)

        input_len = mel.size(0)
        segment_size = 6000
        input_len_seg = [segment_size] * (input_len // segment_size)
        if input_len % segment_size > 0:
            input_len_seg.append(input_len % segment_size)

        codes_packed = encode_batch(
            input_features=mel,
            input_lens=torch.tensor(input_len_seg),
        )

        codes = codes_packed.transpose(0, 1).detach().cpu()
        audio_codes = codes[:, :audio_channels]

        # Pad the sequence to be a multiple of group_size by repeating the last frame
        num_timesteps = audio_codes.shape[0]
        if num_timesteps % group_size != 0:
            padding_needed = group_size - (num_timesteps % group_size)
            last_tokens = audio_codes[-1:, :]  # Keep dim for repeat
            padding_tokens = last_tokens.repeat(padding_needed, 1)
            audio_codes = torch.cat([audio_codes, padding_tokens], dim=0)

        audio_tokenized = audio_codes.reshape(-1)

        return audio_tokenized
    else:
        text = input
        if (
            text.isupper() or text.islower()
        ):  # If the text only contains upper-case or lower-case letters, capitalize it.
            text = text.capitalize()
        return text


def _build_tts_system_prompt(has_voice_prompt: bool, voice_audio_token=None) -> list[InputSegment]:
    """构建 TTS 任务的 system prompt"""
    if has_voice_prompt and voice_audio_token is not None:
        return [
            create_system_start(),
            create_segment(
                text="你需要根据指定的风格指令和文本内容来生成和语音prompt具有相同音色的语音。你的音色应该是："
            ),
            create_segment(text="", audio=voice_audio_token),
            create_user_end(),
        ]
    else:
        return create_system_turn_text_only("你需要根据指定的风格指令和文本内容来生成语音。")


def get_tts_sft_prompt(
    input: None | str = None,
    instruct=None,
    read_text_only=True,
    prompt_speech=None,
):
    """
    TTS (文本转语音) 任务的 prompt 构建

    Args:
        input: 输入文本
        instruct: 风格指令（如：用小孩子的声音开心的说）
        read_text_only: 是否只读取纯文本（False 表示文本里包含 template）
        prompt_speech: 参考音频（用于音色克隆）
    """
    assistant_prompt_audio_token = preprocess_input(prompt_speech) if prompt_speech is not None else None

    if not read_text_only:
        # 不止读取文本，文本里有 template（template:text 性质）
        text = preprocess_input(input)
        lm_prompt = _build_tts_system_prompt(
            has_voice_prompt=assistant_prompt_audio_token is not None, voice_audio_token=assistant_prompt_audio_token
        )
        lm_prompt.append(create_segment(text=f"<|im_start|>user\n{text}<|im_end|>\n"))
        lm_prompt.append(create_assistant_start_with_think())
    else:
        # 纯文本（没有指令在里面）
        language = detect_language(input)
        template = random.choice(tts_zh_templates if language == "zh" else tts_en_templates)
        text = preprocess_input(input)

        if instruct is None:
            # 没有 instruct 指令
            lm_prompt = [
                create_segment(text=f"<|im_start|>user\n{template}: {text}<|im_end|>\n"),
                create_assistant_start_with_sostm(),
            ]
        else:
            # 有 instruct 指令
            lm_prompt = _build_tts_system_prompt(
                has_voice_prompt=assistant_prompt_audio_token is not None,
                voice_audio_token=assistant_prompt_audio_token,
            )
            lm_prompt.append(create_segment(text=f"<|im_start|>user\n{template}: {text}({instruct})<|im_end|>\n"))
            lm_prompt.append(create_assistant_start_with_think())

    return lm_prompt


def get_audio_understanding_sft_prompt(
    input_speech,
    input_text,
    thinking=False,
):
    """音频理解任务的 prompt 构建"""
    audio_tokenized = preprocess_input(input_speech)

    lm_prompt = create_user_turn_with_audio(audio_tokenized, extra_text=input_text)
    lm_prompt = append_assistant_ending(lm_prompt, thinking=thinking)
    return lm_prompt


def _build_voice_prompt_system(prompt_speech) -> list[InputSegment]:
    """构建带音色提示的 system prompt"""
    return create_system_turn_with_voice_prompt(
        prompt_text="Your voice should be：", audio_token=preprocess_input(prompt_speech)
    )


def get_spoken_dialogue_sft_prompt(
    input_speech,
    system_prompt=None,
    prompt_speech=None,
    add_history=False,
):
    """
    语音对话任务的 prompt 构建

    Args:
        input_speech: 输入语音
        system_prompt: 系统提示文本
        prompt_speech: 参考音频（用于音色）
        add_history: 是否添加历史（注：原代码中 history 变量未定义）
    """
    audio_tokenized = preprocess_input(input_speech)
    lm_prompt = []

    # 注意：原代码中 history 变量未定义，此分支可能永远不会执行
    # 如需使用历史功能，应将 history 作为参数传入
    if add_history:
        # 添加历史的简化形式
        lm_prompt = create_user_turn_with_audio(audio_tokenized)
        lm_prompt.append(create_assistant_start_with_sostm())
    else:
        # 添加音色提示（如果有）
        if prompt_speech:
            lm_prompt.extend(_build_voice_prompt_system(prompt_speech))

        # 添加 user turn
        lm_prompt.append(create_user_start())
        if system_prompt:
            lm_prompt.append(create_segment(text=system_prompt))
        lm_prompt.append(create_segment(audio=audio_tokenized))
        lm_prompt.append(create_user_end())
        lm_prompt.append(create_assistant_start_with_sostm())

    return lm_prompt


def get_spoken_dialogue_sft_multiturn_prompt(
    message_list,
    system_prompt=None,
    prompt_speech=None,
    tokenizer=None,
    group_size=8,
    audio_channels=4,
):
    """
    多轮语音对话任务的 prompt 构建

    Args:
        message_list: 消息列表，包含 role 和 content
        system_prompt: 系统提示文本
        prompt_speech: 参考音频（用于音色）
        tokenizer: 分词器
        group_size: 分组大小
        audio_channels: 音频通道数
    """
    lm_prompt = []

    # 添加音色提示（如果有）
    if prompt_speech:
        lm_prompt.extend(
            create_system_turn_with_voice_prompt(
                prompt_text="Your Voice Should be:", audio_token=preprocess_input(prompt_speech)
            )
        )

    # 定义消息处理器
    def user_processor(msg):
        segments = [create_user_start()]
        if system_prompt:
            segments.append(create_segment(text=system_prompt))
        segments.append(create_segment(audio=preprocess_input(msg["content"])))
        segments.append(create_user_end())
        return segments

    def assistant_processor(msg):
        return [
            create_assistant_start(),
            create_streaming_segment(
                text=msg["content"]["text"],
                audio=preprocess_input(msg["content"]["audio"]),
                tokenizer=tokenizer,
                group_size=group_size,
                audio_channels=audio_channels,
            ),
            create_user_end(),
        ]

    # 处理消息列表
    lm_prompt.extend(process_multiturn_messages(message_list, user_processor, assistant_processor))
    lm_prompt.append(create_assistant_start_with_sostm())

    return lm_prompt


def get_s2t_dialogue_sft_prompt(
    input_speech,
    thinking=False,
):
    """语音到文本对话任务的 prompt 构建"""
    audio_tokenized = preprocess_input(input_speech)

    lm_prompt = create_user_turn_with_audio(audio_tokenized)
    lm_prompt = append_assistant_ending(lm_prompt, thinking=thinking)
    return lm_prompt


def get_s2t_dialogue_sft_multiturn_prompt(message_list, thinking=False):
    """多轮语音到文本对话任务的 prompt 构建"""
    lm_prompt = process_multiturn_messages(
        message_list, user_processor=create_audio_user_message, assistant_processor=create_text_assistant_message
    )
    lm_prompt = append_assistant_ending(lm_prompt, thinking=thinking)
    return lm_prompt


def get_text_dialogue_sft_prompt(
    input_text,
    thinking=False,
):
    """纯文本对话任务的 prompt 构建"""
    lm_prompt = create_user_turn_with_text(input_text)
    lm_prompt = append_assistant_ending(lm_prompt, thinking=thinking)
    return lm_prompt


def get_text_dialogue_sft_multiturn_prompt(
    message_list,
    thinking=False,
):
    """多轮纯文本对话任务的 prompt 构建"""
    lm_prompt = process_multiturn_messages(
        message_list, user_processor=create_text_user_message, assistant_processor=create_text_assistant_message
    )
    lm_prompt = append_assistant_ending(lm_prompt, thinking=thinking)
    return lm_prompt


def get_in_context_learning_s2s_prompt(
    instruction,
    prompt_examples,
    audio,
    tokenizer=None,
    group_size=8,
    audio_channels=4,
):
    """
    上下文学习 (In-Context Learning) 语音到语音任务的 prompt 构建

    Args:
        instruction: 指令文本
        prompt_examples: 示例列表，每个示例包含 input_audio, output_transcription, output_audio
        audio: 待处理的输入音频
        tokenizer: 分词器
        group_size: 分组大小
        audio_channels: 音频通道数
    """
    prompt = [create_segment(text=f"[Int]:{instruction}\n")]

    # 添加示例
    for example in prompt_examples:
        prompt.extend(
            [
                create_segment(audio=preprocess_input(example["input_audio"])),
                create_segment(text="\n"),
                create_streaming_segment(
                    text=example["output_transcription"],
                    audio=preprocess_input(example["output_audio"]),
                    tokenizer=tokenizer,
                    group_size=group_size,
                    audio_channels=audio_channels,
                ),
                create_segment(text=" \n\n"),
            ]
        )

    # 添加待处理的音频
    prompt.extend(
        [
            create_segment(audio=preprocess_input(audio)),
            create_segment(text="\n"),
            create_sostm_segment(),
        ]
    )

    return prompt


def get_audio_data(audio_url):
    if audio_url.startswith("data:"):
        header, b64_data = audio_url.split(",", 1)
        audio_bytes = base64.b64decode(b64_data.strip())
        audio_file = io.BytesIO(audio_bytes)
    else:
        # File path
        audio_file = audio_url

    audio_signal, sr = librosa.load(audio_file, sr=24000)
    audio_data = (audio_signal.astype(np.float32), sr)
    return audio_data


def to_prompt(input_segs):
    out_put = []

    for input_seg in input_segs:
        out_put.append(input_seg.text)
        if input_seg.audio is not None:
            out_put.append(input_seg.audio)

    prompt = "".join(out_put)
    print("to_prompt,prompt->", prompt)
    return prompt


if __name__ == "__main__":
    # res = get_tts_sft_prompt(
    #     "用气喘吁吁的年轻男性声音说：我跑不动了，你等等我！",
    #      # instruct='用小孩子的声音开心的说',
    #     read_text_only=False,
    #     prompt_speech=None,
    # )
    #
    # audio_list = []
    # input_speech = "./spoken_dialogue_assistant_turn_1.wav"
    # audio_list.append(get_audio_data(input_speech))
    #
    # res = get_audio_understanding_sft_prompt(
    #     input_speech="./spoken_dialogue_assistant_turn_1.wav",
    #     input_text="Summarize the audio."
    # )
    # prompt = to_prompt(res)
    #
    # final_prompt = {
    #     "prompt": prompt,
    #     "multi_modal_data": {
    #         "audio": audio_list,
    #     },
    # }

    # audio_list = []
    # first_turn_text_response = "我没办法获取实时的天气信息。不过呢，你可以试试几个方法来查看今天的天气。首先，你可以用手机自带的天气功能，比如苹果手机的天气应用，或者直接在系统设置里查看。其次，你也可以用一些专业的天气服务，像是国外的AccuWeather、Weather.com，或者国内的中国天气网、墨迹天气等等。再有就是，你还可以在谷歌或者百度里直接搜索你所在的城市加上天气这两个字。如果你能告诉我你所在的城市，我也可以帮你分析一下历史天气趋势，不过最新的数据还是需要你通过官方渠道去获取哦。"
    # s1_audio_path="今天天气如何.mp3"
    # s2_audio_path="spoken_dialogue_assistant_turn_1.wav"
    # s3_audio_path="北京.mp3"
    # audio_list.append(get_audio_data(s1_audio_path))
    # audio_list.append(get_audio_data(s2_audio_path))
    # audio_list.append(get_audio_data(s3_audio_path))
    # message_list = [
    #     {"role": "user", "content": s1_audio_path},
    #     {"role": "assistant",
    #      "content": {"text": first_turn_text_response, "audio": s2_audio_path}},
    #     {"role": "user", "content": s3_audio_path},
    # ]
    # res = get_spoken_dialogue_sft_multiturn_prompt(message_list,system_prompt=None, prompt_speech="prompt_speech_zh_m.wav")
    # prompt = to_prompt(res)
    #
    # final_prompt = {
    #     "prompt": prompt,
    #     "multi_modal_data": {
    #         "audio": audio_list,
    #     },
    # }

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
    res = get_s2t_dialogue_sft_multiturn_prompt(
        message_list,
        thinking=True,
    )
    prompt = to_prompt(res)
    final_prompt = {
        "prompt": prompt,
        "multi_modal_data": {
            "audio": audio_list,
        },
    }
    print(res)
    print(final_prompt)
