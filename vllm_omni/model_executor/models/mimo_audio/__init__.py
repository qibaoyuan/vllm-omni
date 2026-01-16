__all__ = [
    "interleave_5_and_5_in_span"
]


def interleave_5_and_5_in_span(
    input_ids: list[int],
    *,
    span_start_token: int = 151670,
    span_end_token: int = 151672,
    pad_token_id: int = 151667,
    text_group_size: int = 5,
    pad_group_size: int = 5,
    no_interleave_next_token: int = 151671,
) -> list[int]:
    if not input_ids:
        return input_ids

    original_len = len(input_ids)
    output_ids: list[int] = []
    cursor = 0

    while cursor < original_len:
        # Non-span starting point: Copy as is
        if input_ids[cursor] != span_start_token:
            output_ids.append(input_ids[cursor])
            cursor += 1
            continue

        # ===== Enter span processing =====
        output_ids.append(input_ids[cursor])
        cursor += 1

        span_end_pos = cursor
        while span_end_pos < original_len and input_ids[span_end_pos] != span_end_token:
            span_end_pos += 1

        # If span_end_token is not found: Output the remaining as is and end
        if span_end_pos >= original_len:
            output_ids.extend(input_ids[cursor:])
            break

        next_pos = span_end_pos + 1
        next_token = input_ids[next_pos] if next_pos < original_len else None

        # Rule:
        # After "-end" is 151667(PAD) -> for interlacing
        # After "-end" is 151671 -> Do not interlace
        do_interleave = next_token == pad_token_id
        if next_token == no_interleave_next_token:
            do_interleave = False

        if not do_interleave:
            output_ids.extend(input_ids[cursor:span_end_pos])
            output_ids.append(span_end_token)
            cursor = span_end_pos + 1
            continue

        span_content = input_ids[cursor:span_end_pos]
        span_original_len = len(span_content)
        span_text_tokens = [t for t in span_content if t != pad_token_id]

        rebuilt_span: list[int] = []
        text_cursor = 0
        total_text = len(span_text_tokens)

        while text_cursor < total_text:
            take_text = min(text_group_size, total_text - text_cursor)
            rebuilt_span.extend(span_text_tokens[text_cursor : text_cursor + take_text])
            text_cursor += take_text

            if text_cursor < total_text:
                rebuilt_span.extend([pad_token_id] * pad_group_size)

        output_ids.extend(rebuilt_span)
        output_ids.append(span_end_token)

        extra_tokens_added = len(rebuilt_span) - span_original_len
        cursor = span_end_pos + 1

        while extra_tokens_added > 0 and cursor < original_len:
            if input_ids[cursor] == pad_token_id:
                extra_tokens_added -= 1
                cursor += 1
            else:
                output_ids.append(input_ids[cursor])
                cursor += 1

        if extra_tokens_added > 0:
            return output_ids[:original_len]

    if len(output_ids) > original_len:
        output_ids = output_ids[:original_len]
    elif len(output_ids) < original_len:
        output_ids.extend([pad_token_id] * (original_len - len(output_ids)))

    return output_ids