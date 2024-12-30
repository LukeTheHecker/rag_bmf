from pydantic import BaseModel
import numpy as np
from typing import Optional

class Document(BaseModel):
    text: str
    embedding: np.ndarray
    id: str
    full_path: str
    filename: str
    page: str
    previous_id: Optional[str] = None
    next_id: Optional[str] = None
    document_date: str

    class Config:
        arbitrary_types_allowed = True  # Required for numpy array support