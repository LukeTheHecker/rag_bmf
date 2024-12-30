import chromadb
from chromadb.config import Settings
import numpy as np
from typing import List, Optional
from settings import PATH_SEGMENTS
from .data_models import Document
import pickle as pkl

class DocumentDatabase:
    def __init__(self, persist: bool = True):
        """Initialize the database with ChromaDB backend.
        
        Args:
            persist (bool): If True, stores data on disk. If False, runs in-memory.
        """
        # Configure ChromaDB
        settings = Settings(
            is_persistent=persist,
            persist_directory="./chroma_db" if persist else None,
            anonymized_telemetry=False
        )
        
        self.client = chromadb.Client(settings)
        # Reset collection if it exists (helps during development)
        try:
            self.client.delete_collection("documents")
        except:
            pass
        
        self.collection = self.client.create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Initialize the embedding cache
        self._embedding_cache = {}
        
        self.load_segments()
        self.index_documents()
        
        # Keep a mapping of document IDs to their embeddings for quick lookup
        # self._embedding_cache = {}  # <-- Removed from here

    def load_segments(self):
        """Load document segments from pickle file."""
        with open(PATH_SEGMENTS, "rb") as f:
            self.segments = pkl.load(f)

    def index_documents(self):
        """Index all documents into ChromaDB."""
        embeddings = []
        texts = []
        ids = []
        metadatas = []
        
        # Prepare data in batches
        for segment in self.segments:
            doc = Document(**segment)
            embeddings.append(doc.embedding.tolist())
            texts.append(doc.text)
            ids.append(doc.id)
            
            # Cache embedding for later use
            self._embedding_cache[doc.id] = doc.embedding
            
            metadatas.append({
                "full_path": doc.full_path,
                "filename": doc.filename,
                "page": doc.page,
                "previous_id": doc.previous_id if doc.previous_id else "",
                "next_id": doc.next_id if doc.next_id else "",
                "document_date": doc.document_date
            })
        
        # Add documents to collection
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            ids=ids,
            metadatas=metadatas
        )

    def find(self, query_embedding: np.ndarray, limit: int = 5, extra_context: bool = True) -> List[Document]:
        """Find similar documents using vector similarity search.
        
        Args:
            query_embedding: Query vector
            limit: Number of results to return
            extra_context: Whether to include previous and next segments
        
        Returns:
            List of Document objects
        """
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=limit,
            include=["documents", "metadatas", "distances"]
        )
        
        retrieved_docs = []
        
        # Convert results to Document objects
        for i in range(len(results['ids'][0])):
            doc_id = results['ids'][0][i]
            metadata = results['metadatas'][0][i]
            
            doc = Document(
                text=results['documents'][0][i],
                embedding=self._embedding_cache[doc_id],  # Use cached embedding
                id=doc_id,
                full_path=metadata['full_path'],
                filename=metadata['filename'],
                page=metadata['page'],
                previous_id=metadata['previous_id'] or None,
                next_id=metadata['next_id'] or None,
                document_date=metadata['document_date']
            )
            
            if extra_context:
                doc = self._add_context(doc)
            
            retrieved_docs.append(doc)
        
        return retrieved_docs

    def _add_context(self, doc: Document) -> Document:
        """Add previous and next segment context to a document."""
        from copy import deepcopy
        doc_extended = deepcopy(doc)
        
        # Add previous context if exists
        if doc.previous_id:
            prev_result = self.collection.get(
                ids=[doc.previous_id],
                include=["documents"]
            )
            if prev_result and prev_result['documents']:
                doc_extended.text = prev_result['documents'][0] + "\n\n" + doc_extended.text
        
        # Add next context if exists
        if doc.next_id:
            next_result = self.collection.get(
                ids=[doc.next_id],
                include=["documents"]
            )
            if next_result and next_result['documents']:
                doc_extended.text = doc_extended.text + "\n\n" + next_result['documents'][0]
        
        return doc_extended