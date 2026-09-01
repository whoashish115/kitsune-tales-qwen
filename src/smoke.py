"""GPU smoke tests for the Qwen3.5 hybrid architecture (Phase 2 gate; runs on one cheap GPU).
Each check is independent and returns a dict, so one failure does not hide the others:
1. ``load``        – AutoModelForCausalLM loads the text model; parameter counts; LoRA target leaves;
                     whether flash-linear-attention kernels are active.
2. ``packing``     – are packed (padding-free, position_ids reset) sequences numerically equivalent to
                     separate ones? Gated DeltaNet layers carry recurrent state, so this decides D-012.
3. ``train``       – a few LoRA steps on fixture examples: finite, decreasing loss; tokens/s; memory.
4. ``merge``       – merged vs adapter-on-base equivalence (the same check used for the release).
5. ``vllm``        – vLLM loads the base text-only, generates Japanese, and (if supported) serves a LoRA.
"""
from kitsune.fixtures import GENRES, STORY, SYNOPSIS, TITLE
from kitsune.prompts import StoryRequest, build_user_prompt, render_prompt, tokenize_example

def _guard(name: str, fn: Any, *a: Any, **k: Any) -> dict[str, Any]:
    t0 = time.time()
    try:
        out = fn(*a, **k)
        _free()
        return {"ok": True, "seconds": round(time.time() - t0, 1), **out}
    except Exception as e:  # report and continue with the other checks
        _free()
        return {
            "ok": False,
            "seconds": round(time.time() - t0, 1),
            "error": f"{type(e).__name__}: {e}",
            "trace": traceback.format_exc()[-3000:],
        }
