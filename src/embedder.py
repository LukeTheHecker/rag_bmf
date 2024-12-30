from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np


class Embedder:
    def __init__(self, model_name, normalize=True):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.normalize = normalize
        self.embed_dim = self.model.config.hidden_size

    def embed(self, text):
        # Tokenize the text
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)

        # Get the embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)

        # To numpy array
        embeddings = embeddings.numpy()

        # Normalize the embeddings
        if self.normalize:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        return embeddings
    
    def embed_documents(self, documents, batch_size=16):
        embeddings = np.zeros((len(documents), self.embed_dim))
        for i in range(0, len(documents), batch_size):
            print(f"Embedding batch {i//batch_size+1} of {len(documents)//batch_size}")
            batch = documents[i:i+batch_size]
            # Batch tokenize
            inputs = self.tokenizer(batch, return_tensors="pt", truncation=True, padding=True)

            # Get the embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings_new = outputs.last_hidden_state.mean(dim=1)
            # To numpy array
            embeddings_new = embeddings_new.numpy()

            # Normalize the embeddings
            if self.normalize:
                embeddings_new = embeddings_new / np.linalg.norm(embeddings_new, axis=1, keepdims=True)

            
            embeddings[i:i+batch_size] = embeddings_new

        return embeddings