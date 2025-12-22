from typing import Any, Union

import torch
from vllm.inputs import TextPrompt

from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.model_executor.models.mimo_audio.myutils import print_shape

TALKER_CODEC_PAD_TOKEN_ID = 8292
TALKER_CODEC_START_TOKEN_ID = 8293
TALKER_CODEC_END_TOKEN_ID = 8294
MAX_LENGTH = 8192

# MiMo-Audio special tokens
EMPTY_TOKEN_ID = 151667  # <|empty|>
SOSTM_TOKEN_ID = 151670  # <|sostm|>
EOSTM_TOKEN_ID = 151671  # <|eostm|>


def thinker2talker(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: Union[OmniTokensPrompt, TextPrompt, None] = None,
    requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    """
    Process thinker outputs to create talker inputs for MiMo-Audio.

    关键：Talker 需要接收 Thinker 的 hidden states，
    并在生成 audio codes 后将其转换为 embeddings 返回给 Thinker。

    Args:
        stage_list: List of stage objects
        engine_input_source: Source stage IDs (typically [0] for thinker)
        prompt: Original prompt data
        requires_multimodal_data: Whether multimodal data is required

    Returns:
        List of OmniTokensPrompt for talker stage
    """
    print(
        "thinker2talker",
        "stage_list",
        stage_list,
        "prompt",
        prompt,
        "requires_multimodal_data",
        requires_multimodal_data,
    )

    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")

    source_stage_id = engine_input_source[0]
    if source_stage_id >= len(stage_list):
        raise IndexError(f"Invalid stage_id: {source_stage_id}")

    if stage_list[source_stage_id].engine_outputs is None:
        raise RuntimeError(f"Stage {source_stage_id} has no outputs yet")

    thinker_outputs = stage_list[source_stage_id].engine_outputs
    talker_inputs = []

    if not isinstance(prompt, list):
        prompt = [prompt]

    multi_modal_data = {
        thinker_output.request_id: p.get("multi_modal_data", None) for thinker_output, p in zip(thinker_outputs, prompt)
    }

    for i, thinker_output in enumerate(thinker_outputs):
        output = thinker_output.outputs[0]
        prompt_token_ids = thinker_output.prompt_token_ids
        thinker_output_ids = output.token_ids
        prompt_token_ids_len = len(prompt_token_ids)

        # Extract hidden states from thinker output
        latent = output.multimodal_output["latent"]
        thinker_hidden_states = latent.clone().detach().to(latent.device)

        additional_information = {
            "thinker_result": thinker_hidden_states[prompt_token_ids_len:].to(torch.float32),
            "prompt_embeds": thinker_hidden_states[:prompt_token_ids_len].to(torch.float32),
            "prompt_token_ids": prompt_token_ids,
            "thinker_output_token_ids": thinker_output_ids,
            "thinker_result_shape": list(thinker_hidden_states[prompt_token_ids_len:].shape),
            "prompt_embeds_shape": list(thinker_hidden_states[:prompt_token_ids_len].shape),
        }

        talker_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[EMPTY_TOKEN_ID],  # Talker 处理 <|empty|> token
                additional_information=additional_information,
                multi_modal_data=(
                    multi_modal_data[thinker_output.request_id] if multi_modal_data is not None else None
                ),
                mm_processor_kwargs=None,
            )
        )

    return talker_inputs


# TypeError: llm2lf() takes from 2 to 3 positional arguments but 4 were given
def llm2lf(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: Union[OmniTokensPrompt, TextPrompt, None] = None,
    requires_multimodal_data: bool = False,
):
    print("llm2lf", "stage_list", stage_list, "prompt", prompt, "requires_multimodal_data", requires_multimodal_data)
    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")
    source_stage_id = engine_input_source[0]
    if source_stage_id >= len(stage_list):
        raise IndexError(f"Invalid stage_id: {source_stage_id}")
    if stage_list[source_stage_id].engine_outputs is None:
        raise RuntimeError(f"Stage {source_stage_id} has no outputs yet")
    llm_outputs = stage_list[source_stage_id].engine_outputs
    talker_inputs = []
    if not isinstance(prompt, list):
        prompt = [prompt]
    multi_modal_data = {
        thinker_output.request_id: p.get("multi_modal_data", None) for thinker_output, p in zip(llm_outputs, prompt)
    }

    for i, thinker_output in enumerate(llm_outputs):
        output = thinker_output.outputs[0]
        prompt_token_ids = thinker_output.prompt_token_ids
        thinker_output_ids = output.token_ids
        prompt_token_ids_len = len(prompt_token_ids)
        latent = output.multimodal_output["latent"]
        llm_hidden_states = latent.clone().detach().to(latent.device)
        additional_information = {
            "llm_result": llm_hidden_states[prompt_token_ids_len:].to(torch.float32),
            "prompt_embeds": llm_hidden_states[:prompt_token_ids_len].to(torch.float32),
            "prompt_token_ids": prompt_token_ids,
            "llm_output_token_ids": thinker_output_ids,
            "llm_result_shape": list(llm_hidden_states[prompt_token_ids_len:].shape),
            "prompt_embeds_shape": list(llm_hidden_states[:prompt_token_ids_len].shape),
        }
        talker_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[TALKER_CODEC_START_TOKEN_ID]
                + [TALKER_CODEC_PAD_TOKEN_ID] * (len(prompt_token_ids))
                + [TALKER_CODEC_END_TOKEN_ID],
                additional_information=additional_information,
                multi_modal_data=(
                    multi_modal_data[thinker_output.request_id] if multi_modal_data is not None else None
                ),
                mm_processor_kwargs=None,
            )
        )
    return talker_inputs


def lf2llm(stage_list, engine_input_source, prompt: Union[OmniTokensPrompt, TextPrompt] = None):
    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")
    source_stage_id = engine_input_source[0]
    if source_stage_id >= len(stage_list):
        raise IndexError(f"Invalid stage_id: {source_stage_id}")
    if stage_list[source_stage_id].engine_outputs is None:
        raise RuntimeError(f"Stage {source_stage_id} has no outputs yet")
    llm_outputs = stage_list[source_stage_id].engine_outputs
    talker_inputs = []
    if not isinstance(prompt, list):
        prompt = [prompt]
    multi_modal_data = {
        thinker_output.request_id: p.get("multi_modal_data", None) for thinker_output, p in zip(llm_outputs, prompt)
    }

    for i, thinker_output in enumerate(llm_outputs):
        output = thinker_output.outputs[0]
        prompt_token_ids = thinker_output.prompt_token_ids
        thinker_output_ids = output.token_ids
        prompt_token_ids_len = len(prompt_token_ids)
        latent = output.multimodal_output["latent"]
        llm_hidden_states = latent.clone().detach().to(latent.device)
        additional_information = {
            "llm_result": llm_hidden_states[prompt_token_ids_len:].to(torch.float32),
            "prompt_embeds": llm_hidden_states[:prompt_token_ids_len].to(torch.float32),
            "prompt_token_ids": prompt_token_ids,
            "llm_output_token_ids": thinker_output_ids,
            "llm_result_shape": list(llm_hidden_states[prompt_token_ids_len:].shape),
            "prompt_embeds_shape": list(llm_hidden_states[:prompt_token_ids_len].shape),
        }
        talker_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[TALKER_CODEC_START_TOKEN_ID]
                + [TALKER_CODEC_PAD_TOKEN_ID] * (len(prompt_token_ids))
                + [TALKER_CODEC_END_TOKEN_ID],
                additional_information=additional_information,
                multi_modal_data=(
                    multi_modal_data[thinker_output.request_id] if multi_modal_data is not None else None
                ),
                mm_processor_kwargs=None,
            )
        )
    return talker_inputs


def prepend_and_flatten_colmajor(x: torch.Tensor, pad_vec: torch.Tensor):
    """
    x: (..., R, C)  最后两维是 (行, 列)
    pad_vec: (C,)   要在最前面添加的一行，列数必须等于 C

    返回：按列优先（col-major）展平的一维向量
    """
    # 保证 pad_vec 的形状为 (1, C)
    pad_row = pad_vec.view(1, -1)

    # 扩展 pad_row 到 x 的前面，保持其他批次维度一致
    # 例：x shape = (B,1,R,C) → pad shape = (B,1,1,C)
    pad_expand = pad_row.view(*([1] * (x.dim() - 2)), 1, x.size(-1)).expand(*x.shape[:-2], 1, x.size(-1))

    # prepend 到行维度
    y = torch.cat([pad_expand, x], dim=-2)  # (..., R+1, C)

    # 按列优先展开：
    # 先把 (..., R+1, C) → (..., C, R+1)
    # 再 flatten
    y_col_major = y.permute(*range(y.dim() - 2), -1, -2).reshape(-1)

    return y_col_major


def llm2code2wav(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: Union[OmniTokensPrompt, TextPrompt, None] = None,
    requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    """
    Process talker outputs to create code2wav inputs.

    Workflow:
    1. Extract talker's codec code outputs (8-layer RVQ codes)
    2. Flatten codes for code2wav input
    3. Package for code2wav stage

    Args:
        stage_list: List of stage objects
        engine_input_source: Source stage IDs (typically [1] for talker)
        prompt: Original prompt data
        requires_multimodal_data: Whether multimodal data is required

    Returns:
        List of OmniTokensPrompt for code2wav stage
    """
    print_shape(
        stage_list=stage_list,
        engine_input_source=engine_input_source,
        prompt=prompt,
        requires_multimodal_data=requires_multimodal_data,
    )
    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")

    source_stage_id = engine_input_source[0]
    if source_stage_id >= len(stage_list):
        raise IndexError(f"Invalid stage_id: {source_stage_id}")

    if stage_list[source_stage_id].engine_outputs is None:
        raise RuntimeError(f"Stage {source_stage_id} has no outputs yet")

    talker_outputs = stage_list[source_stage_id].engine_outputs
    code2wav_inputs = []

    # Process each talker output
    for i, talker_output in enumerate(talker_outputs):
        output = talker_output.outputs[0]
        print("output loop talker_outputs", output, "code", output.multimodal_output["code"])

        # Extract codec codes from talker output
        # Expected shape: [8, seq_len] (8-layer RVQ codes)
        codec_codes = output.multimodal_output["code"].to(torch.long)
        # codec_codes = (
        #     output.multimodal_output["code"].to(torch.long).transpose(0, 1).cpu().to(torch.long).reshape(-1).tolist()
        # )  # 16, seq_len
        print("codec_codes", codec_codes)

        pad_vec = torch.tensor([151667, 151667, 151667, 151667])

        # 在每个最内层张量前加 pad_vec
        # x shape: (2, 1, 8, 4)
        pad = pad_vec.view(1, 1, 1, 4).expand(codec_codes.size(0), 1, 1, 4)  # (2, 1, 1, 4)

        code_final = prepend_and_flatten_colmajor(codec_codes, pad_vec)
        code_final = code_final.tolist()

        if len(code_final) > MAX_LENGTH:
            print(f"Warning: code_final length {len(code_final)} exceeds {MAX_LENGTH}, truncating to {MAX_LENGTH}")
            code_final = code_final[:MAX_LENGTH]
        print("original code_final", code_final)
        # code_final = [
        #     151665,
        #     1024,
        #     1024,
        #     128,
        #     128,
        #     128,
        #     128,
        #     128,
        #     128,
        #     151665,
        #     1024,
        #     1024,
        #     128,
        #     128,
        #     128,
        #     128,
        #     128,
        #     128,
        #     151665,
        #     1024,
        #     1024,
        #     128,
        #     128,
        #     128,
        #     128,
        #     128,
        #     128,
        #     151665,
        #     1024,
        #     1024,
        #     128,
        #     128,
        #     128,
        #     128,
        #     128,
        #     128,
        #     151667,
        #     285,
        #     341,
        #     39,
        #     25,
        #     104,
        #     69,
        #     57,
        #     47,
        #     151667,
        #     285,
        #     100,
        #     119,
        #     18,
        #     65,
        #     89,
        #     110,
        #     37,
        #     151667,
        #     43,
        #     1021,
        #     5,
        #     9,
        #     44,
        #     99,
        #     41,
        #     10,
        #     151667,
        #     947,
        #     940,
        #     75,
        #     14,
        #     44,
        #     56,
        #     31,
        #     102,
        #     151667,
        #     965,
        #     309,
        #     37,
        #     5,
        #     61,
        #     37,
        #     121,
        #     89,
        #     151667,
        #     602,
        #     905,
        #     35,
        #     20,
        #     78,
        #     92,
        #     72,
        #     41,
        #     151667,
        #     981,
        #     727,
        #     20,
        #     64,
        #     17,
        #     115,
        #     12,
        #     116,
        #     151667,
        #     981,
        #     963,
        #     14,
        #     13,
        #     78,
        #     73,
        #     120,
        #     19,
        #     151667,
        #     390,
        #     573,
        #     19,
        #     62,
        #     108,
        #     46,
        #     40,
        #     18,
        #     151667,
        #     171,
        #     8,
        #     50,
        #     0,
        #     17,
        #     99,
        #     26,
        #     57,
        #     151667,
        #     557,
        #     8,
        #     20,
        #     74,
        #     59,
        #     74,
        #     25,
        #     11,
        #     151667,
        #     391,
        #     573,
        #     35,
        #     3,
        #     59,
        #     94,
        #     50,
        #     29,
        #     151667,
        #     862,
        #     227,
        #     76,
        #     44,
        #     38,
        #     66,
        #     80,
        #     105,
        #     151667,
        #     358,
        #     946,
        #     43,
        #     64,
        #     15,
        #     34,
        #     28,
        #     27,
        #     151667,
        #     38,
        #     392,
        #     76,
        #     37,
        #     36,
        #     118,
        #     12,
        #     34,
        #     151667,
        #     323,
        #     30,
        #     76,
        #     118,
        #     76,
        #     68,
        #     32,
        #     56,
        #     151667,
        #     358,
        #     944,
        #     26,
        #     118,
        #     101,
        #     68,
        #     123,
        #     10,
        #     151667,
        #     38,
        #     857,
        #     97,
        #     111,
        #     59,
        #     28,
        #     121,
        #     127,
        #     151667,
        #     323,
        #     621,
        #     26,
        #     37,
        #     127,
        #     14,
        #     12,
        #     40,
        #     151667,
        #     839,
        #     72,
        #     66,
        #     48,
        #     31,
        #     46,
        #     81,
        #     116,
        #     151667,
        #     38,
        #     392,
        #     65,
        #     89,
        #     36,
        #     122,
        #     68,
        #     108,
        #     151667,
        #     862,
        #     621,
        #     26,
        #     122,
        #     76,
        #     100,
        #     95,
        #     123,
        #     151667,
        #     1001,
        #     448,
        #     107,
        #     52,
        #     75,
        #     19,
        #     119,
        #     54,
        #     151667,
        #     421,
        #     431,
        #     81,
        #     17,
        #     109,
        #     96,
        #     92,
        #     77,
        #     151667,
        #     80,
        #     935,
        #     112,
        #     38,
        #     59,
        #     126,
        #     11,
        #     45,
        #     151667,
        #     862,
        #     581,
        #     81,
        #     88,
        #     24,
        #     19,
        #     29,
        #     9,
        #     151667,
        #     1001,
        #     72,
        #     124,
        #     123,
        #     83,
        #     42,
        #     85,
        #     22,
        #     151667,
        #     508,
        #     640,
        #     26,
        #     37,
        #     76,
        #     5,
        #     126,
        #     114,
        #     151667,
        #     839,
        #     402,
        #     101,
        #     11,
        #     1,
        #     19,
        #     57,
        #     0,
        #     151667,
        #     557,
        #     728,
        #     76,
        #     60,
        #     32,
        #     102,
        #     84,
        #     77,
        #     151667,
        #     323,
        #     621,
        #     102,
        #     125,
        #     99,
        #     5,
        #     117,
        #     127,
        #     151667,
        #     839,
        #     504,
        #     84,
        #     112,
        #     76,
        #     26,
        #     116,
        #     92,
        #     151667,
        #     557,
        #     469,
        #     27,
        #     38,
        #     121,
        #     63,
        #     9,
        #     124,
        #     151667,
        #     358,
        #     180,
        #     26,
        #     19,
        #     28,
        #     58,
        #     119,
        #     89,
        #     151667,
        #     862,
        #     504,
        #     26,
        #     37,
        #     106,
        #     46,
        #     43,
        #     114,
        #     151667,
        #     708,
        #     177,
        #     88,
        #     5,
        #     76,
        #     79,
        #     101,
        #     118,
        #     151667,
        #     358,
        #     686,
        #     97,
        #     0,
        #     113,
        #     113,
        #     93,
        #     77,
        #     151667,
        #     61,
        #     298,
        #     40,
        #     88,
        #     87,
        #     74,
        #     77,
        #     121,
        #     151667,
        #     557,
        #     769,
        #     26,
        #     89,
        #     76,
        #     49,
        #     43,
        #     113,
        #     151667,
        #     38,
        #     68,
        #     43,
        #     121,
        #     51,
        #     62,
        #     121,
        #     28,
        #     151667,
        #     862,
        #     402,
        #     26,
        #     37,
        #     57,
        #     82,
        #     44,
        #     114,
        #     151667,
        #     358,
        #     137,
        #     76,
        #     37,
        #     72,
        #     19,
        #     110,
        #     38,
        #     151667,
        #     557,
        #     180,
        #     26,
        #     13,
        #     106,
        #     46,
        #     75,
        #     35,
        #     151667,
        #     323,
        #     431,
        #     17,
        #     37,
        #     8,
        #     19,
        #     1,
        #     5,
        #     151667,
        #     862,
        #     504,
        #     122,
        #     37,
        #     76,
        #     22,
        #     2,
        #     79,
        #     151667,
        #     839,
        #     488,
        #     76,
        #     37,
        #     48,
        #     79,
        #     26,
        #     63,
        #     151667,
        #     862,
        #     160,
        #     26,
        #     88,
        #     76,
        #     5,
        #     11,
        #     127,
        #     151667,
        #     839,
        #     198,
        #     1,
        #     73,
        #     57,
        #     2,
        #     98,
        #     56,
        #     151667,
        #     557,
        #     177,
        #     108,
        #     78,
        #     76,
        #     35,
        #     45,
        #     52,
        #     151667,
        #     323,
        #     118,
        #     3,
        #     57,
        #     111,
        #     40,
        #     93,
        #     87,
        #     151667,
        #     351,
        #     646,
        #     108,
        #     78,
        #     87,
        #     90,
        #     116,
        #     0,
        #     151667,
        #     3,
        #     99,
        #     125,
        #     38,
        #     21,
        #     4,
        #     123,
        #     8,
        #     151667,
        #     287,
        #     935,
        #     18,
        #     3,
        #     106,
        #     89,
        #     117,
        #     23,
        #     151667,
        #     805,
        #     655,
        #     37,
        #     116,
        #     2,
        #     59,
        #     1,
        #     25,
        #     151667,
        #     781,
        #     89,
        #     79,
        #     71,
        #     66,
        #     4,
        #     116,
        #     63,
        #     151667,
        #     127,
        #     80,
        #     112,
        #     121,
        #     56,
        #     29,
        #     38,
        #     63,
        #     151667,
        #     960,
        #     960,
        #     60,
        #     122,
        #     77,
        #     43,
        #     47,
        #     22,
        #     151667,
        #     960,
        #     463,
        #     60,
        #     45,
        #     122,
        #     35,
        #     12,
        #     88,
        #     151667,
        #     960,
        #     745,
        #     60,
        #     58,
        #     10,
        #     58,
        #     27,
        #     102,
        #     151667,
        #     955,
        #     55,
        #     89,
        #     9,
        #     122,
        #     115,
        #     72,
        #     102,
        #     151667,
        #     377,
        #     421,
        #     21,
        #     125,
        #     4,
        #     94,
        #     96,
        #     108,
        #     151667,
        #     981,
        #     610,
        #     27,
        #     99,
        #     4,
        #     69,
        #     102,
        #     9,
        #     151667,
        #     176,
        #     527,
        #     125,
        #     43,
        #     126,
        #     35,
        #     81,
        #     12,
        #     151667,
        #     868,
        #     88,
        #     21,
        #     73,
        #     125,
        #     35,
        #     91,
        #     96,
        #     151667,
        #     51,
        #     601,
        #     85,
        #     69,
        #     41,
        #     122,
        #     80,
        #     102,
        #     151667,
        #     881,
        #     580,
        #     60,
        #     51,
        #     89,
        #     85,
        #     42,
        #     121,
        #     151667,
        #     999,
        #     519,
        #     54,
        #     81,
        #     22,
        #     41,
        #     64,
        #     114,
        #     151667,
        #     569,
        #     519,
        #     71,
        #     81,
        #     35,
        #     53,
        #     98,
        #     4,
        #     151667,
        #     930,
        #     519,
        #     45,
        #     93,
        #     126,
        #     58,
        #     78,
        #     105,
        #     151667,
        #     122,
        #     1,
        #     29,
        #     61,
        #     55,
        #     77,
        #     124,
        #     37,
        #     151667,
        #     718,
        #     892,
        #     29,
        #     87,
        #     88,
        #     43,
        #     62,
        #     16,
        #     151667,
        #     400,
        #     401,
        #     75,
        #     99,
        #     88,
        #     71,
        #     23,
        #     110,
        #     151667,
        #     938,
        #     820,
        #     21,
        #     51,
        #     94,
        #     1,
        #     86,
        #     1,
        #     151667,
        #     584,
        #     669,
        #     43,
        #     52,
        #     95,
        #     1,
        #     111,
        #     127,
        #     151667,
        #     122,
        #     250,
        #     97,
        #     69,
        #     45,
        #     112,
        #     54,
        #     4,
        #     151667,
        #     171,
        #     824,
        #     30,
        #     113,
        #     88,
        #     61,
        #     54,
        #     1,
        #     151667,
        #     43,
        #     96,
        #     114,
        #     48,
        #     60,
        #     112,
        #     121,
        #     96,
        #     151667,
        #     862,
        #     212,
        #     66,
        #     48,
        #     28,
        #     101,
        #     64,
        #     14,
        #     151667,
        #     351,
        #     212,
        #     88,
        #     80,
        #     51,
        #     19,
        #     32,
        #     38,
        #     151667,
        #     862,
        #     997,
        #     26,
        #     17,
        #     12,
        #     123,
        #     10,
        #     89,
        #     151667,
        #     557,
        #     180,
        #     102,
        #     89,
        #     45,
        #     47,
        #     117,
        #     108,
        #     151667,
        #     314,
        #     728,
        #     26,
        #     119,
        #     77,
        #     86,
        #     75,
        #     84,
        #     151667,
        #     80,
        #     212,
        #     43,
        #     4,
        #     90,
        #     3,
        #     83,
        #     126,
        #     151667,
        #     557,
        #     325,
        #     43,
        #     69,
        #     88,
        #     103,
        #     81,
        #     68,
        #     151667,
        #     323,
        #     744,
        #     122,
        #     120,
        #     122,
        #     66,
        #     77,
        #     23,
        #     151667,
        #     323,
        #     514,
        #     97,
        #     38,
        #     15,
        #     45,
        #     126,
        #     53,
        #     151667,
        #     839,
        #     1021,
        #     121,
        #     78,
        #     87,
        #     13,
        #     36,
        #     123,
        #     151667,
        #     358,
        #     91,
        #     65,
        #     37,
        #     45,
        #     47,
        #     75,
        #     101,
        #     151667,
        #     32,
        #     588,
        #     51,
        #     13,
        #     25,
        #     103,
        #     36,
        #     74,
        #     151667,
        #     38,
        #     91,
        #     64,
        #     105,
        #     36,
        #     107,
        #     53,
        #     74,
        #     151667,
        #     414,
        #     337,
        #     70,
        #     19,
        #     63,
        #     67,
        #     19,
        #     74,
        #     151667,
        #     414,
        #     337,
        #     70,
        #     19,
        #     63,
        #     67,
        #     19,
        #     74,
        #     151666,
        #     1024,
        #     1024,
        #     128,
        #     128,
        #     128,
        #     128,
        #     128,
        #     128,
        #     151666,
        #     1024,
        #     1024,
        #     128,
        #     128,
        #     128,
        #     128,
        #     128,
        #     128,
        #     151666,
        #     1024,
        #     1024,
        #     128,
        #     128,
        #     128,
        #     128,
        #     128,
        #     128,
        #     151666,
        #     1024,
        #     1024,
        #     128,
        #     128,
        #     128,
        #     128,
        #     128,
        #     128
        # ]
        # code_final = (
        #     [
        #         151665,
        #         1024,
        #         1024,
        #         128,
        #         128,
        #         128,
        #         128,
        #         128,
        #         128,
        #         151665,
        #         1024,
        #         1024,
        #         128,
        #         128,
        #         128,
        #         128,
        #         128,
        #         128,
        #         151665,
        #         1024,
        #         1024,
        #         128,
        #         128,
        #         128,
        #         128,
        #         128,
        #         128,
        #         151665,
        #         1024,
        #         1024,
        #         128,
        #         128,
        #         128,
        #         128,
        #         128,
        #         128,
        #     ]
        #     + code_final
        #     + [
        #         151666,
        #         1024,
        #         1024,
        #         128,
        #         128,
        #         128,
        #         128,
        #         128,
        #         128,
        #         151666,
        #         1024,
        #         1024,
        #         128,
        #         128,
        #         128,
        #         128,
        #         128,
        #         128,
        #         151666,
        #         1024,
        #         1024,
        #         128,
        #         128,
        #         128,
        #         128,
        #         128,
        #         128,
        #         151666,
        #         1024,
        #         1024,
        #         128,
        #         128,
        #         128,
        #         128,
        #         128,
        #         128,
        #     ]
        # )
        print("code_final", code_final)
        code2wav_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=code_final,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return code2wav_inputs
