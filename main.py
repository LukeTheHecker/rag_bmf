from src.rag_pipeline import RAGPipeline

if __name__ == "__main__":
    rag_pipeline = RAGPipeline(verbose=True)

    # Valid queries
    valid_queries = [
        "Wie hoch ist die Grundzulage?",
        # "Wie werden Versorgungsleistungen aus einer Direktzusage oder einer Unterstützungskasse steuerlich behandelt?",
        # "Wie werden Leistungen aus einer Direktversicherung, Pensionskasse oder einem Pensionsfonds in der Auszahlungsphase besteuert?",
        # "Wie kann der Wert der Altersversorgung auf den neuen Arbeitgeber übertragen werden?"
    ]

    for query in valid_queries:
        response = rag_pipeline.run_with_retry(query)
        print("-" * 100)
        print(f"Query: {query}\n\n{response}\n\n")
    
    # Invalid queries
    invalid_queries = [
        # "Wie lange war Barack Obama Präsident von den USA?",  # Not relevant to the documents
        # "Wie hoch ist der Sozialversicherungsbeitrag in Deutschland?"  # Not relevant to the documents
    ]
    for query in invalid_queries:
        response = rag_pipeline.run_with_retry(query)
        print("-" * 100)
        print(f"Query: {query}\n\nResponse:\n{response}\n\n")
    
