import os

MODEL_CONTEXT_WINDOW = int(os.getenv("MODEL_CONTEXT_WINDOW", "128000"))
CONTEXT_WINDOW_USAGE_LIMIT = float(
    os.getenv("CONTEXT_WINDOW_USAGE_LIMIT", "0.75"))

CONTEXT_TOKEN_BUDGETS = {
    "system": int(os.getenv("CONTEXT_SYSTEM_TOKENS", "1000")),
    "structured_state": int(os.getenv("CONTEXT_STATE_TOKENS", "500")),
    "summary": int(os.getenv("CONTEXT_SUMMARY_TOKENS", "600")),
    "history": int(os.getenv("CONTEXT_HISTORY_TOKENS", "2000")),
    "rag": int(os.getenv("CONTEXT_RAG_TOKENS", "5000")),
    "output": int(os.getenv("CONTEXT_OUTPUT_TOKENS", "2000")),
}

MAX_CONTEXT_TOKENS = int(MODEL_CONTEXT_WINDOW * CONTEXT_WINDOW_USAGE_LIMIT)

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").strip().lower()
GROQ_GENERATION_MODEL = os.getenv("GENERATION_MODEL", "openai/gpt-oss-20b")
OPENAI_GENERATION_MODEL = os.getenv("OPENAI_GENERATION_MODEL", "gpt-4o-mini")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local").strip().lower()
OPENAI_EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
