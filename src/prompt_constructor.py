from .prompts import USER_PROMPT
from typing import List
from .data_models import Document

class PromptConstructor:
    def __init__(self):
        pass

    def construct_prompt(self, query, retrieved_docs: List[Document]):
        documents = ""

        for i_doc, doc in enumerate(retrieved_docs):
            header = f"Treffer {i_doc+1} von {len(retrieved_docs)}:\nDatei {doc.filename} Seite {doc.page} vom {doc.document_date}"
            documents += f"{header}\n{doc.text}\n\n"
        
        prompt = USER_PROMPT.format(question=query, documents=documents)
        return prompt

        

