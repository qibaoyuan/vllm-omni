from typing import Any, Union

import torch
from vllm.inputs import TextPrompt

from vllm_omni.inputs.data import OmniTokensPrompt

TALKER_CODEC_PAD_TOKEN_ID = 8292
TALKER_CODEC_START_TOKEN_ID = 8293
TALKER_CODEC_END_TOKEN_ID = 8294


def prepend_and_flatten_colmajor(x: torch.Tensor, pad_vec: torch.Tensor) -> torch.Tensor:
    """
    Prepend a padding vector to the input tensor and flatten in column-major order.

    This function expands the padding vector to match the batch dimensions of the input
    tensor, prepends it to the row dimension, and then flattens the result in column-major
    order (transposing before flattening).

    Args:
        x: Input tensor with shape (..., R, C) where R is the row dimension and C is
            the column dimension. Example: (B, 1, 8, 4) where B is batch size.
        pad_vec: Padding vector with shape (C,) to be prepended to x. The vector will
            be expanded to match the batch dimensions of x.

    Returns:
        A flattened 1D tensor in column-major order with shape (-1,). The result
        contains the padded row followed by all rows of x, flattened column by column.
    """
    pad_row = pad_vec.view(1, -1)

    # Expand pad_row to the front of x, keeping other batch dimensions consistent
    # Example: x shape = (B,1,R,C) → pad shape = (B,1,1,C)
    pad_expand = pad_row.view(*([1] * (x.dim() - 2)), 1, x.size(-1)).expand(*x.shape[:-2], 1, x.size(-1))

    # Prepend to the row dimension
    y = torch.cat([pad_expand, x], dim=-2)  # (..., R+1, C)

    # Flatten in column-major order:
    # First transpose (..., R+1, C) → (..., C, R+1)
    # Then flatten
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

        # Extract codec codes from talker output
        # Expected shape: [8, seq_len] (8-layer RVQ codes)
        codec_codes = output.multimodal_output["code"].to(torch.long)

        pad_vec = torch.tensor([151667, 151667, 151667, 151667])

        code_final = prepend_and_flatten_colmajor(codec_codes, pad_vec)
        code_final = code_final.tolist()

        code2wav_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=code_final,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return code2wav_inputs
