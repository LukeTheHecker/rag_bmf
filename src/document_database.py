
import numpy as np
import pickle as pkl
from docarray import DocList
from docarray.index import InMemoryExactNNIndex
from typing import List
from constants import PATH_SEGMENTS
from .data_models import Document


class DocumentDatabase:
    def __init__(self):
        self.load_segments()
        self.doc_index = InMemoryExactNNIndex[Document]()
        self.index_documents()

    def load_segments(self):
        with open(PATH_SEGMENTS, "rb") as f:
            segments = pkl.load(f)
        self.segments = segments

    def index_documents(self):
        doc_list = DocList[Document]([Document(**segment) for segment in self.segments])
        self.doc_index.index(doc_list)

    def find(self, query_embedding: np.ndarray, limit: int = 5, extra_context: bool = True) -> List[Document]:
        '''
        Find the most relevant documents for a given query embedding.

        Parameters:
        ----------
        query_embedding: np.ndarray
            The embedding of the query.
        limit: int
            The number of documents to return.
        extra_context: bool
            Whether to return the most relevant documents with extra context.

        Returns:
        -------
        retrieved_docs: List[Document]
            The most relevant documents.
        '''
        from copy import deepcopy
        retrieved_docs, _ = self.doc_index.find(query_embedding, search_field='embedding', limit=limit)
        if extra_context:
            retrieved_docs_extended = []
            for doc in retrieved_docs:
                doc_extended = deepcopy(doc)
                next_id = doc.next_id
                previous_id = doc.previous_id
                
                if previous_id:
                    doc_previous = self.doc_index[previous_id]
                    doc_extended.text = doc_previous.text + "\n\n" + doc_extended.text
                
                if next_id:
                    doc_next = self.doc_index[next_id]
                    doc_extended.text = doc_extended.text + "\n\n" + doc_next.text
                
                retrieved_docs_extended.append(doc_extended)

            return retrieved_docs_extended
        else:
            return list(retrieved_docs)
