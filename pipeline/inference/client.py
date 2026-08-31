"""pipeline/inference/client.py

Native-response inference clients: given a rendered chat-message list and a
JSON schema, return the model's parsed {target_label: value} dict directly.
No token probabilities, no logprobs, no answer labels anywhere in this
module -- the model is asked for (and, on the vLLM backend, CONSTRAINED to
produce) its native integer answers directly.

Environment note: see the (archived) former vllm_elicit.py's provenance --
this machine's CUDA/torch/vLLM/transformers pin triangle is unchanged
(torch==2.5.1+cu121, vllm==0.7.3, transformers==4.48.2); see
pipeline/requirements.txt.
"""

from __future__ import annotations

import json

from inference.model_config import inference_parameters, selected_model

DEFAULT_VLLM_MODEL = None
DEFAULT_HF_MODEL = None


class VLLMNativeClient:
    """Loads the model once; __call__ returns one parsed {label: value}
    dict per (messages, schema) request, using vLLM's guided JSON decoding
    (xgrammar backend, already a vLLM 0.7.3 dependency -- verified with a
    real model call: constrained output was valid, in-range JSON every
    time) so the model is constrained to produce a schema-conforming
    answer, not just asked nicely for one.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_VLLM_MODEL,
        max_model_len: int = 6144,
        gpu_memory_utilization: float = 0.85,
        max_new_tokens: int = 1024,
        temperature: float | None = None,
        top_p: float | None = None,
        **llm_kwargs,
    ):
        # A full block prompt (all 44 items + one condition's stimulus) measured
        # ~2900 tokens for the longest real stimulus sampled -- 6144 leaves
        # solid headroom over that plus max_new_tokens, without the earlier
        # single-answer client's tighter 4096 (fine there since prompts were
        # per-item, much shorter; this client sends the whole block at once).
        from vllm import LLM

        self.llm = LLM(model=model_name, dtype="bfloat16", gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len, **llm_kwargs)
        self.tokenizer = self.llm.get_tokenizer()
        self.max_new_tokens = max_new_tokens
        params = inference_parameters()
        self.temperature = float(params["temperature"] if temperature is None else temperature)
        self.top_p = float(params["top_p"] if top_p is None else top_p)

    def _sampling_params(self, schema: dict):
        from vllm import SamplingParams
        from vllm.sampling_params import GuidedDecodingParams

        return SamplingParams(max_tokens=self.max_new_tokens, temperature=self.temperature, top_p=self.top_p, guided_decoding=GuidedDecodingParams(json=schema))

    def __call__(self, messages: list[dict[str, str]], schema: dict) -> dict[str, int]:
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        [output] = self.llm.generate([prompt], self._sampling_params(schema), use_tqdm=False)
        return json.loads(output.outputs[0].text)

    def call_many(self, requests: list[tuple[list[dict[str, str]], dict]]) -> list[dict[str, int]]:
        """Batch many independent (messages, schema) requests into ONE
        vLLM generate() call -- vLLM's continuous batching handles a
        heterogeneous batch of different guided schemas correctly (each
        request carries its own SamplingParams). Not part of the minimal
        simulate_response() interface; a caller that's collected many
        requests up front can use this for real throughput.
        """
        prompts = [self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m, _ in requests]
        sampling_params = [self._sampling_params(schema) for _, schema in requests]
        outputs = self.llm.generate(prompts, sampling_params, use_tqdm=False)
        return [json.loads(o.outputs[0].text) for o in outputs]


class HFNativeClient:
    """CPU/transformers fallback: no grammar-constrained decoding available
    here, so this prompts for JSON and parses + validates the result,
    retrying with a corrective reprompt a bounded number of times on
    invalid/incomplete/out-of-range JSON -- documented as less robust than
    the vLLM backend's guided decoding, not silently assumed equivalent.

    Confirmed by a real failure, not hypothetical: even with each item's
    numeric range stated explicitly in the prompt text (inference.prompts),
    the small default HF model (Qwen2.5-1.5B-Instruct) still occasionally
    puts an out-of-range value on one field (e.g. a donation item answered
    on the surrounding sliders' 0-100 scale instead of its own 0-10) after
    several retries -- a small-model instruction-following limit, not
    something a bigger retry budget reliably fixes. Since this backend has
    no generation-time constraint to fall back on, after exhausting
    retries this CLAMPS any remaining out-of-range values to their nearest
    bound and logs exactly which fields were clamped, rather than crashing
    a whole run over one stubborn field -- the vLLM backend never needs
    this path (verified: real constrained-decoding output was valid and
    in-range on every call tested).
    """

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        max_new_tokens: int = 1024,
        max_retries: int = 6,
        temperature: float | None = None,
        top_p: float | None = None,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.max_retries = max_retries
        params = inference_parameters()
        self.temperature = float(params["temperature"] if temperature is None else temperature)
        self.top_p = float(params["top_p"] if top_p is None else top_p)

    def _generate_once(self, messages: list[dict[str, str]]) -> str:
        import torch

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)

    def __call__(self, messages: list[dict[str, str]], schema: dict) -> dict[str, int]:
        required = schema["required"]
        attempt_messages = list(messages)
        text, parsed = "", None
        for _ in range(self.max_retries + 1):
            text = self._generate_once(attempt_messages)
            parsed = _try_parse_json(text)
            if parsed is not None and set(required) <= set(parsed.keys()) and _values_in_bounds(parsed, schema):
                return {k: parsed[k] for k in required}
            violations = _describe_violations(parsed, schema) if parsed is not None else "not valid JSON at all"
            attempt_messages = messages + [
                {"role": "assistant", "content": text},
                {"role": "user", "content": f"That response was invalid: {violations}. Respond with ONLY the corrected JSON object, one integer key per item, all in range."},
            ]

        if parsed is None or not (set(required) <= set(parsed.keys())):
            raise ValueError(f"model failed to produce valid JSON with all required keys after {self.max_retries + 1} attempts; last output: {text!r}")
        clamped, fixed_fields = _clamp_to_bounds(parsed, schema)
        print(f"[HFNativeClient] clamped out-of-range field(s) after exhausting retries: {fixed_fields}")
        return {k: clamped[k] for k in required}


def _try_parse_json(text: str) -> dict | None:
    text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def _values_in_bounds(parsed: dict, schema: dict) -> bool:
    for key, spec in schema["properties"].items():
        if key not in parsed:
            continue
        v = parsed[key]
        if isinstance(v, bool) or not isinstance(v, int):
            return False
        if "enum" in spec and v not in spec["enum"]:
            return False
        if "minimum" in spec and not (spec["minimum"] <= v <= spec["maximum"]):
            return False
    return True


def _describe_violations(parsed: dict, schema: dict) -> str:
    problems = []
    for key, spec in schema["properties"].items():
        if key not in parsed:
            problems.append(f"{key} is missing")
            continue
        v = parsed[key]
        if isinstance(v, bool) or not isinstance(v, int):
            problems.append(f"{key}={v!r} is not an integer")
        elif "enum" in spec and v not in spec["enum"]:
            problems.append(f"{key}={v} must be one of {spec['enum']}")
        elif "minimum" in spec and not (spec["minimum"] <= v <= spec["maximum"]):
            problems.append(f"{key}={v} must be between {spec['minimum']} and {spec['maximum']}")
    return "; ".join(problems) if problems else "unknown validation failure"


def _clamp_to_bounds(parsed: dict, schema: dict) -> tuple[dict, list[str]]:
    """Last-resort fallback after exhausting retries (HFNativeClient only --
    see its docstring): push any remaining out-of-range integer to its
    nearest bound. Returns (clamped_dict, list of "field: old->new" for
    every field actually changed, so the caller can log it -- never silent.
    """
    clamped = dict(parsed)
    fixed = []
    for key, spec in schema["properties"].items():
        if key not in clamped:
            continue
        v = clamped[key]
        if isinstance(v, bool) or not isinstance(v, int):
            continue  # not a boundable numeric violation -- still missing/wrong-typed, caller already raises for that
        if "enum" in spec:
            new_v = min(spec["enum"], key=lambda allowed: (abs(allowed - v), allowed))
        else:
            new_v = max(spec["minimum"], min(spec["maximum"], v))
        if new_v != v:
            fixed.append(f"{key}: {v}->{new_v}")
            clamped[key] = new_v
    return clamped, fixed


def make_native_client(backend: str, model_name: str | None = None, **kwargs):
    resolved_model = model_name or selected_model("g", require_frozen=True)
    if backend == "vllm":
        return VLLMNativeClient(model_name=resolved_model, **kwargs)
    if backend == "hf":
        return HFNativeClient(model_name=resolved_model, **kwargs)
    raise ValueError(f"unknown backend: {backend!r}")
