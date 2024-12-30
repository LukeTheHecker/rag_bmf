import sys
from pathlib import Path

# Add the project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from settings import EMBEDDER_MODEL, NORMALIZE_EMBEDDINGS
from src.embedder import Embedder
import json
import pickle as pkl

# Load the embedding model
embedder = Embedder(EMBEDDER_MODEL, normalize=NORMALIZE_EMBEDDINGS)

# Load the segments
path_segments = str(project_root / "data" / "segments" / "segments.json")
with open(path_segments, "r", encoding="utf-8") as f:
    segments = json.load(f)

# Compute the embeddings in batches for speed
texts = [segment["text"] for segment in segments]
embeddings = embedder.embed_documents(texts, batch_size=16)

# Save the embeddings to the segments
for i, embedding in enumerate(embeddings):
    segments[i]["embedding"] = embedding

# Save the segments with embeddings
path_segments_with_embeddings = str(project_root / "data" / "segments" / "segments_with_embeddings.pkl")
with open(path_segments_with_embeddings, "wb") as f:
    pkl.dump(segments, f)

print(f"Script finished.")