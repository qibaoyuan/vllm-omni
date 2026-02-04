# // AIGC START
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from vllm.logger import init_logger

logger = init_logger(__name__)


def _unwrap_model(model: Any) -> Any:
    """Best-effort unwrap common wrapper modules (e.g. torch.compile OptimizedModule)."""
    cur = model
    seen: list[str] = []
    # Avoid infinite loops in case of self-referential attributes.
    for _ in range(8):
        if cur is None:
            return None
        seen.append(cur.__class__.__name__)
        # Common wrapper attributes (best-effort / duck-typing).
        for attr in ("_orig_mod", "module", "model", "_model", "inner_model", "wrapped_model", "_wrapped_model"):
            nxt = getattr(cur, attr, None)
            if nxt is not None and nxt is not cur:
                cur = nxt
                break
        else:
            break
    return cur


def _unwrap_chain(model: Any) -> list[Any]:
    """Return the unwrap chain objects (including original) without logging."""
    chain: list[Any] = []
    cur = model
    for _ in range(8):
        chain.append(cur)
        if cur is None:
            break
        for attr in ("_orig_mod", "module", "model", "_model", "inner_model", "wrapped_model", "_wrapped_model"):
            nxt = getattr(cur, attr, None)
            if nxt is not None and nxt is not cur:
                cur = nxt
                break
        else:
            break
    return chain


def is_mimo_audio_model(model: Any) -> bool:
    """Robust MiMoAudio model check without importing model classes.

    - Prefer checking the unwrapped model class name.
    - Fall back to feature detection for wrapped/proxied models.
    """
    chain = _unwrap_chain(model)

    def _matches(obj: Any) -> bool:
        name = getattr(obj, "__class__", type("X", (), {})).__name__
        mod = getattr(getattr(obj, "__class__", None), "__module__", "") or ""
        if name == "MiMoAudioForConditionalGeneration":
            return True
        if "MiMoAudio" in name:
            return True
        if "mimo_audio" in mod:
            return True
        return False

    # Check every hop in the unwrap chain (original + wrappers + inner).
    for obj in chain:
        if _matches(obj):
            return True

    m = chain[-1] if chain else None

    # Feature detection fallback: MiMoAudio typically has fused_thinker_talker and postprocess hooks.
    has_ftt = hasattr(m, "fused_thinker_talker") or hasattr(m, "token2wav") or hasattr(m, "model_stage")
    has_pp = hasattr(m, "postprocess_batch") or hasattr(m, "postprocess")
    has_flag = bool(getattr(m, "has_postprocess", False) or getattr(m, "has_preprocess", False))
    return bool(has_ftt and has_pp and has_flag)


def inject_sampled_token_ids(
    runner: Any,
    valid_sampled_token_ids: torch.Tensor | list | tuple | None,
    req_ids_output_copy: list[str],
    req_id_to_index_output_copy: dict[str, int],
    sampler_output: Any,
) -> bool:
    """Inject sampled_token_id / request_id into per-request additional_information_cpu for MiMo only.

    This enables MiMo postprocess(_batch) to decide whether to run local decoding based on sampled_token_id.
    Returns True if injection happened (MiMo), otherwise False.
    """
    model = getattr(runner, "model", None)
    is_mimo = is_mimo_audio_model(model)
    if not is_mimo:
        return False

    sampled_token_ids_list: list[int] = []

    # Parse valid_sampled_token_ids (can be Tensor/list/tuple/None).
    if valid_sampled_token_ids is None:
        pass
    elif isinstance(valid_sampled_token_ids, torch.Tensor):
        sampled_token_ids_list = [int(t) for t in valid_sampled_token_ids.tolist()]
    elif isinstance(valid_sampled_token_ids, (list, tuple)):
        for t in valid_sampled_token_ids:
            if isinstance(t, (list, tuple)) and t:
                sampled_token_ids_list.append(int(t[0]))
            elif isinstance(t, (list, tuple)):
                continue
            else:
                sampled_token_ids_list.append(int(t))

    # Fallback: sampler_output.sampled_token_ids (GPU tensor).
    if len(sampled_token_ids_list) == 0 and len(req_ids_output_copy) > 0:
        st = getattr(sampler_output, "sampled_token_ids", None)
        if isinstance(st, torch.Tensor):
            for rid in req_ids_output_copy:
                idx = req_id_to_index_output_copy.get(rid)
                if idx is not None and idx < len(st):
                    sampled_token_ids_list.append(int(st[idx].item()))
                else:
                    sampled_token_ids_list.append(None)
        else:
            sampled_token_ids_list = [None] * len(req_ids_output_copy)

    # Inject to the requests that will be returned to engine output.
    for rid, token_id in zip(req_ids_output_copy, sampled_token_ids_list):
        req_state = runner.requests.get(rid)
        if req_state is None:
            continue
        info = getattr(req_state, "additional_information_cpu", None)
        if not isinstance(info, dict):
            info = {}
        info["sampled_token_id"] = (int(token_id) if token_id is not None else None)
        info["request_id"] = rid
        # Stable key name used by MiMo code paths.
        info["req_id"] = rid
        setattr(req_state, "additional_information_cpu", info)

    # Ensure all requests in input_batch have sampled_token_id placeholder
    # (req_ids_output_copy and input_batch.req_ids can differ after filtering).
    for req_id in runner.input_batch.req_ids:
        if req_id in req_ids_output_copy:
            continue
        req_state = runner.requests.get(req_id)
        if req_state is None:
            continue
        info = getattr(req_state, "additional_information_cpu", None)
        if not isinstance(info, dict):
            info = {}
        info.setdefault("sampled_token_id", None)
        info["request_id"] = req_id
        info["req_id"] = req_id
        setattr(req_state, "additional_information_cpu", info)

    return True


def run_postprocess(
    runner: Any,
    hidden_states: torch.Tensor,
    num_scheduled_tokens_np: np.ndarray,
    scheduler_output: Any,
) -> None:
    """Run MiMo postprocess (prefer batch) and merge updates into request state."""
    model = getattr(runner, "model", None)
    is_mimo = is_mimo_audio_model(model)
    if not is_mimo:
        return

    model = runner.model

    # Prefer batched postprocess if available.
    if hasattr(model, "postprocess_batch"):
        req_infos_list: list[dict] = []

        # Keep the same req_id order as sampled_token_id injection if provided by AR runner.
        req_ids_to_use = getattr(runner, "_current_req_ids_output_copy", None)
        if req_ids_to_use is None:
            req_ids_to_use = runner.input_batch.req_ids

        for req_id in req_ids_to_use:
            if runner.model_config.async_chunk:
                req_infos = runner._get_additional_information(scheduler_output, req_id)
            else:
                req_state = runner.requests.get(req_id)
                req_infos = getattr(req_state, "additional_information_cpu", None) if req_state is not None else None
            if not isinstance(req_infos, dict):
                req_infos = {}
            req_infos_list.append(req_infos)

        updates = model.postprocess_batch(
            hidden_states=hidden_states,
            req_infos_list=req_infos_list,
            query_start_loc=runner.query_start_loc.cpu,
            num_scheduled_tokens_np=num_scheduled_tokens_np,
        )

        # updates can be:
        # - dict[req_id, dict]
        # - list[dict] aligned with req_infos_list (thus aligned with req_ids_to_use)
        if isinstance(updates, dict):
            for req_id, upd in updates.items():
                if isinstance(upd, dict):
                    runner._merge_additional_information_update(req_id, upd)
        elif isinstance(updates, list):
            for req_id, upd in zip(req_ids_to_use, updates):
                if isinstance(upd, dict):
                    runner._merge_additional_information_update(req_id, upd)
        return

    # Compat path: per-request postprocess().
    for req_index, req_id in enumerate(runner.input_batch.req_ids):
        if runner.model_config.async_chunk:
            req_infos = runner._get_additional_information(scheduler_output, req_id)
        else:
            req_state = runner.requests.get(req_id)
            req_infos = getattr(req_state, "additional_information_cpu", None) if req_state is not None else None
        if not isinstance(req_infos, dict):
            req_infos = {}

        start_offset = int(runner.query_start_loc.cpu[req_index])
        sched_tokens = int(num_scheduled_tokens_np[req_index])
        s, e = start_offset, start_offset + sched_tokens
        hidden_states_slice = hidden_states[s:e]
        update_dict = model.postprocess(hidden_states_slice, **req_infos)
        runner._merge_additional_information_update(req_id, update_dict)
# // AIGC END


