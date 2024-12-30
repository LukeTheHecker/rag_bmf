# README
Retrieval Augmented Generation (RAG) for BMF.

By [Lukas Hecker](https://github.com/lukethehecker).

## Installation
The project is written in python 3.11.
```bash
pip install -r requirements.txt
```

## Usage

```python
python main.py
```

## Explanation of the code

### Reading the PDFs
PDFs are read using the [PyMuPDF](https://github.com/pymupdf/PyMuPDF) library. Segments are extracted using several heuristics to detect:
- paragraphs
- headings
- broken paragraphs on new pages

This part is pre-computed using [this script](scripts/extract_segments_from_pdfs.py).

### Embeddings
An embedding model from the Hugging Face library is used to embed the segments ([danielheinz/e5-base-sts-en-de](https://huggingface.co/danielheinz/e5-base-sts-en-de)
). The embeddings are normalized to unit length (using the `L2-norm`) and can be compared using cosine similarity.

The embeddings are pre-computed using [this script](scripts/compute_segment_embeddings.py) and can be found in the `data/segments/segments_with_embeddings.pkl` file.

### Database
The [chromadb](https://github.com/chroma-core/chroma) library is utilized to enable efficient storage of embeddings, text, and metadata in Python, ensuring minimal overhead.

### Retrieval and Generation
The query is encoded using an embedding model, and the top 5 matches are retrieved based on cosine similarity using HNSW (Hierarchical Navigable Small World), an approximate nearest neighbor approach that balances speed and accuracy. These matches are sent to the LLM as context for generation, which is performed using the GPT-4o model from the OpenAI library. If the query cannot be answered with the provided context, the LLM is instructed to return certain keyword which triggers a retry.

In such cases, the context is expanded by by 2 documents (totalling 7). Furthermore, the preceding and subsequent documents of each "hit" is added as context, too. This iterative process is repeated up to two times. If the query still cannot be resolved, the system indicates that the query cannot be answered with the available documents, which will be communicated to the user.

## Suggested Improvements
### PDF Extraction
Bad data is the root of all evil. Therefore it is crucial to have a robust data pipeline for extracting the segments from the PDFs.

* If documents of different types are going to be added in the future, the current heuristic for extracting the segments from the PDFs is not going to be sufficient.
* Find more accurate heuristics for structured extraction by incorporating the styling of the text (example: a larger font may indicate a heading)

### Embeddings
Embeddings can become inaccurate if the data is not clean. For example, the extracted text segments sometimes have minor formatting issues (e.g. a missing space between two words), which can lead to a significant deviation in the embeddings. 

The quality of the embeddings can be tested using the files `scripts/test_embedding_model_formatting.py` and `scripts/test_embedding_model_query_context.py`. I find that the model returns highly similar embeddings for paraphrased questions, but it lacks robustness when it comes to formatting issues (corrupted spaces or added special characters).

### Vector Database
In production we would probably have more than four PDFs to work with. Therefore it is crucial to have a scalable solution for storing the embeddings.

* Use a vector database like postgres pgvector for storing the embeddings for a more scalable solution

### Generation
Due to its easily accessible API, I used the `GPT-4o` model from the OpenAI library. This might not be an option for clients with data privacy obligations. In this case, open source models from Meta (Llama family) or from Mistral AI are recommended. These models can be accessed through self-hosting (high data privacy, high cost) or using third parties (lower cost, data privacy needs to be assessed.).

### Testing
Writing tests for various components is crucial to ensure a robust system.