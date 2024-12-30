from docarray import BaseDoc
from docarray.typing import NdArray

class Document(BaseDoc):
    text: str
    embedding: NdArray
    id: str
    full_path: str
    filename: str
    page: str
    previous_id: str | None
    next_id: str | None
    document_date: str