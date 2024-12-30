import os

MIN_CHARS_PER_CHUNK = 5
EMBEDDER_MODEL = "danielheinz/e5-base-sts-en-de"
NORMALIZE_EMBEDDINGS = True

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_NAME = "gpt-4o"
TEMPERATURE = 0.0
MAX_TOKENS = 4096

PATH_SEGMENTS = "data/segments/segments_with_embeddings.pkl"
DOCUMENT_LIMIT = 5
EXTRA_CONTEXT = True