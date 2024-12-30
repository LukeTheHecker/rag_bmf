import sys
from pathlib import Path

# Add the project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from settings import EMBEDDER_MODEL, NORMALIZE_EMBEDDINGS
from src.embedder import Embedder
import numpy as np

# Prepare test data
intact_text = "Wie errechnet sich die Höhe des Solidaritätszuschlags?"
# the opposite of the intact is 
corrupted_texts = [
    "Wie e rrechnet sichdie H öhe des Solidaritätszusc  hlags  ?",  # corrputed spaces
    """Wie errec,,,hnet sich die "Höhe" des Solidaritäts-zuschlags?""",  # added special characters
    "Wie wird die Höhe des Solidaritätszuschlags eigentlich berechnet?",  # paraphrasing
]

descriptions = [
    "corrupted spaces",
    "added special characters",
    "paraphrasing",
]
 
# Load the embedding model
embedder = Embedder(EMBEDDER_MODEL, normalize=NORMALIZE_EMBEDDINGS)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

intact_embedding = embedder.embed([intact_text])[0]

similarity_matrix = np.zeros((len(corrupted_texts)))
for j, corrputed_text in enumerate(corrupted_texts):
    corrputed_embedding = embedder.embed([corrputed_text])[0]
    similarity_matrix[j] = cosine_similarity(intact_embedding, corrputed_embedding)
    print(f"Similarity to {descriptions[j]}: {similarity_matrix[j]:.2f}")