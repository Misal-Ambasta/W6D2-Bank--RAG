# Embedding and Retrieval Log

This file documents embedding generation, caching, and retrieval details for the banking RAG project.

## Embedding Model
- Model: `all-MiniLM-L6-v2`
- Framework: sentence-transformers

## Embedding Generation
- Text chunks are encoded using the MiniLM model.
- Embeddings are cached using a content hash to avoid recomputation.

## Logging
- This file will be updated with details of embedding runs, cache hits/misses, and any relevant statistics.


## ChromaDB Vector Store

- Chunks (text + metadata) are stored in a local Chroma collection.
- Embeddings are generated using the MiniLM model and stored alongside metadata.
- LangChain's Chroma integration is used for ingestion and retrieval.
- Semantic queries are run to validate retrieval quality.

Example usage:
```python
from app.chroma_utils import build_chroma_collection, semantic_query
vectorstore = build_chroma_collection(chunks)
results = semantic_query("What are the amortization terms?", vectorstore)
```


## Retrieval Chain with LangChain

- The system uses `ConversationalRetrievalChain` for retrieval-augmented generation with memory.
- Conversation history is managed via `ConversationBufferMemory`.
- The retriever is configured with ChromaDB and search parameter `k=4`.
- Example RAG flow is provided below.

Example usage:
```python
from app.retrieval_chain_utils import build_conversational_chain, example_rag_flow
result = example_rag_flow()
print(result)
```

---

Embedding run: 30 texts | Cache hits: 0 | Cache misses: 30

Embedding run: 30 texts | Cache hits: 30 | Cache misses: 0

Embedding run: 30 texts | Cache hits: 30 | Cache misses: 0

Embedding run: 1 texts | Cache hits: 0 | Cache misses: 1

Embedding run: 1 texts | Cache hits: 1 | Cache misses: 0

Embedding run: 1 texts | Cache hits: 1 | Cache misses: 0

Embedding run: 1 texts | Cache hits: 1 | Cache misses: 0
