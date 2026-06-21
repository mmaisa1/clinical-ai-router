# app/pipeline/llm.py
"""
LLM fallback for low-confidence classifications.

Called when Stage 1 + Stage 2 routing score falls below CONFIDENCE_THRESHOLD.
Backend is selected via LLM_BACKEND in .env — local_qwen (production default),
groq (RAM-constrained / fast iteration), or anthropic (max accuracy, paid).

Output is always validated against the Stage 1 Top-K candidates passed in.
Invalid output triggers one retry; if still invalid, falls back to
candidates[0] rather than returning an unvalidated label.
"""

from app.core.config import (
    LLM_BACKEND,
    QWEN_MODEL_NAME,
    QWEN_USE_4BIT,
    GROQ_API_KEY,
    GROQ_MODEL,
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    LLM_MAX_RETRIES,
    LLM_MAX_NEW_TOKENS,
)
from app.core.labels import LABEL_MAP


def _build_prompt(report_text: str, candidates: list[str]) -> str:
    """Shared prompt template across all backends."""
    label_list = "\n".join(f"- {c}: {LABEL_MAP[c]}" for c in candidates)
    return (
        "You are a clinical pathology expert.\n\n"
        "Read the following pathology report excerpt and classify it "
        "into exactly one cancer type from the list below.\n\n"
        f"REPORT:\n{report_text}\n\n"
        f"CANCER TYPE OPTIONS:\n{label_list}\n\n"
        "Respond with ONLY the cancer type code (e.g. BRCA, READ, LUAD). "
        "No explanation. No punctuation. Just the code."
    )


def _validate_output(raw: str, candidates: list[str]) -> str | None:
    """Return the cleaned code if valid, else None."""
    if not raw:
        return None
    code = raw.strip().upper().split()[0] if raw.strip() else None
    if code in candidates:
        return code
    return None


# Qwen2.5-7B (local, production default)

class QwenFallback:
    def __init__(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        self.torch = torch
        kwargs = {"device_map": "auto"}
        if QWEN_USE_4BIT and torch.cuda.is_available():
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else:
            kwargs["torch_dtype"] = torch.float32  # CPU fallback, no quantization

        self.tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(QWEN_MODEL_NAME, **kwargs)
        self.model.eval()

    def _call(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        text_input = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text_input, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=LLM_MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def classify(self, report_text: str, candidates: list[str]) -> tuple[str, bool]:
        """Returns (predicted_code, was_valid)."""
        prompt = _build_prompt(report_text, candidates)
        for attempt in range(LLM_MAX_RETRIES + 1):
            raw = self._call(prompt)
            code = _validate_output(raw, candidates)
            if code:
                return code, True
        return candidates[0], False


# Groq Llama 3.1 8B (fast iteration / RAM-constrained deployments)

class GroqFallback:
    def __init__(self):
        from groq import Groq

        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY not set in .env")
        self.client = Groq(api_key=GROQ_API_KEY)

    def classify(self, report_text: str, candidates: list[str]) -> tuple[str, bool]:
        prompt = _build_prompt(report_text, candidates)
        for attempt in range(LLM_MAX_RETRIES + 1):
            response = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=LLM_MAX_NEW_TOKENS,
                temperature=0,
            )
            raw = response.choices[0].message.content
            code = _validate_output(raw, candidates)
            if code:
                return code, True
        return candidates[0], False


# Anthropic Claude Haiku (max accuracy, paid)

class AnthropicFallback:
    def __init__(self):
        import anthropic

        if not ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY not set in .env")
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    def classify(self, report_text: str, candidates: list[str]) -> tuple[str, bool]:
        prompt = _build_prompt(report_text, candidates)
        for attempt in range(LLM_MAX_RETRIES + 1):
            response = self.client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=LLM_MAX_NEW_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text
            code = _validate_output(raw, candidates)
            if code:
                return code, True
        return candidates[0], False


# Factory 

def get_llm_fallback():
    """Returns the configured fallback instance based on LLM_BACKEND."""
    if LLM_BACKEND == "local_qwen":
        return QwenFallback()
    elif LLM_BACKEND == "groq":
        return GroqFallback()
    elif LLM_BACKEND == "anthropic":
        return AnthropicFallback()
    else:
        raise ValueError(f"Unknown LLM_BACKEND: {LLM_BACKEND}")