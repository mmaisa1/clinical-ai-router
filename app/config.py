"""
Central configuration for the Clinical AI Inference Router.

All values are loaded from environment variables via .env, with defaults
matching the values validated during offline evaluation
(see research/01_train_and_evaluate.ipynb).

Do not hardcode secrets here — they belong in .env, which is gitignored.
"""

import os
import secrets

from dotenv import load_dotenv

load_dotenv()


def _get_bool(key: str, default: bool) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes")


# Pipeline configuration
# These must match the values used in research/01_train_and_evaluate.ipynb.
# Changing them without re-running offline evaluation invalidates the
# validated 0.6 threshold and reported accuracy numbers.

TOP_K = int(os.getenv("TOP_K", 5))
CHUNK_TOKENS = int(os.getenv("CHUNK_TOKENS", 256))
CHUNK_STRIDE = int(os.getenv("CHUNK_STRIDE", 64))
MAX_CHUNKS = int(os.getenv("MAX_CHUNKS", 12))

# Validated empirically: accuracy drops from 90.5% to 53.0% below this point.
# Can be updated live via POST /admin/threshold without restarting the service.
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.6))

# Model paths
MODELS_DIR = os.getenv("MODELS_DIR", "./models")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "vectorizer.joblib")
SVM_MODEL_PATH = os.path.join(MODELS_DIR, "svm_model.joblib")

NLI_MODEL_NAME = os.getenv("NLI_MODEL_NAME", "facebook/bart-large-mnli")

# LLM fallback configuration
# LLM_BACKEND selects which fallback model handles low-confidence cases.
# Options: "local_qwen" (default, zero cost) | "groq" | "anthropic"
LLM_BACKEND = os.getenv("LLM_BACKEND", "local_qwen")

QWEN_MODEL_NAME = os.getenv("QWEN_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
QWEN_USE_4BIT = _get_bool("QWEN_USE_4BIT", True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", 1))
LLM_MAX_NEW_TOKENS = int(os.getenv("LLM_MAX_NEW_TOKENS", 10))

# Admin authentication
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "")

if not ADMIN_API_KEY:
    raise RuntimeError(
        "ADMIN_API_KEY is not set. Generate one with:\n"
        "  python -c \"import secrets; print(secrets.token_hex(32))\"\n"
        "and add it to your .env file."
    )

# Database
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/clinical_router"
)

# App metadata
APP_NAME = "Clinical AI Inference Router"
APP_VERSION = "0.1.0"