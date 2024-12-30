from .embedder import Embedder
from .prompt_constructor import PromptConstructor
from .chatgpt_client import ChatGPTClient
from settings import (EMBEDDER_MODEL, NORMALIZE_EMBEDDINGS, 
                        DOCUMENT_LIMIT, EXTRA_CONTEXT)
from .document_database_2 import DocumentDatabase

class RAGPipeline:
    def __init__(self, verbose: bool = False):
        import time
        self.verbose = verbose

        if self.verbose:
            print("\nInitializing RAG Pipeline components...")
        
        start = time.time()
        self.embedder = Embedder(EMBEDDER_MODEL, normalize=NORMALIZE_EMBEDDINGS)
        if self.verbose:
            print(f"Embedder initialization took {time.time() - start:.2f}s")
        
        start = time.time()
        self.document_database = DocumentDatabase()
        if self.verbose:
            print(f"Document Database initialization took {time.time() - start:.2f}s")
        
        start = time.time()
        self.prompt_constructor = PromptConstructor()
        if self.verbose:
            print(f"Prompt Constructor initialization took {time.time() - start:.2f}s")
        
        start = time.time()
        self.chatgpt_client = ChatGPTClient()
        if self.verbose:
            print(f"ChatGPT Client initialization took {time.time() - start:.2f}s")

    def run(self, query: str, doc_limit: int = DOCUMENT_LIMIT, extra_context: bool = EXTRA_CONTEXT) -> str:
        '''
        Run the RAG pipeline.

        Parameters:
        ----------
        query: str
            The query to answer.
        doc_limit: int
            Number of documents to retrieve
        extra_context: bool
            Whether to include extra context

        Returns:
        -------
        response: str
            The response to the query.
        '''
        import time
        
        start = time.time()
        query_embedding = self.embedder.embed(query.strip())[0]
        if self.verbose:
            print(f"Query embedding took {time.time() - start:.2f}s")
        
        start = time.time()
        retrieved_docs = self.document_database.find(query_embedding, limit=doc_limit, extra_context=extra_context)
        if self.verbose:
            print(f"Document retrieval took {time.time() - start:.2f}s")
        
        start = time.time()
        prompt = self.prompt_constructor.construct_prompt(query, retrieved_docs)
        if self.verbose:
            print(f"Prompt construction took {time.time() - start:.2f}s")
        
        start = time.time()
        response = self.chatgpt_client.generate_response(prompt)
        if self.verbose:
            print(f"OpenAI call took {time.time() - start:.2f}s")
        
        return response

    def run_with_retry(self, query: str, max_retries: int = 2, doc_limit_increment: int = 2) -> str:
        '''
        Wrapper for run() with automatic retry on "Hoppla" responses.

        Parameters:
        ----------
        query: str
            The query to answer
        max_retries: int
            Maximum number of retry attempts
        doc_limit_increment: int
            How much to increase the document limit on each retry

        Returns:
        -------
        response: str
            The final response to the query
        '''
        current_limit = DOCUMENT_LIMIT
        current_try = 0
        extra_context = EXTRA_CONTEXT
        
        while current_try <= max_retries:
            response = self.run(query, doc_limit=current_limit, extra_context=extra_context)
            
            if not response.startswith("Hoppla"):
                if current_try > 0:
                    print(f"\tneeded {current_try+1} try/tries")
                return response
            
            # print(f"(Re)Try {current_try} failed with response: {response[:100]}...\n\nAdding {doc_limit_increment} more documents to the context.")
                
            current_try += 1
            current_limit += doc_limit_increment
            extra_context = True  # includes the previous and subsequent document of every hit
        
        return response  # Return last response if all retries failed
