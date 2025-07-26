# System Architecture

This document outlines the architecture of the Banking RAG system, detailing the flow of data and the interaction between its core components.

## Core Components

1.  **Streamlit UI (`streamlit_app.py`)**: The user-facing web application. It handles file uploads, displays the chat interface, and renders the LLM's responses and source document chunks.

2.  **Document Loading (`load_documents.py`)**: A module responsible for ingesting various document formats (PDF, DOCX, Excel). It uses `langchain-docling` to parse files into a structured format.

3.  **Chunking (`HybridChunker`)**: The structure-aware chunking mechanism from `langchain-docling`. It intelligently splits documents into text chunks while preserving the integrity of tables and other structural elements.

4.  **Embedding (`embedding_utils.py`)**: This component uses the `all-MiniLM-L6-v2` sentence transformer model to convert text chunks into dense vector embeddings. It includes a caching mechanism to avoid re-computing embeddings for unchanged content.

5.  **Vector Store (`chroma_utils.py`)**: A local ChromaDB instance that stores the vector embeddings and their associated metadata (e.g., source document, page number). It enables efficient semantic similarity searches.

6.  **Retrieval & Generation (`retrieval_chain_utils.py`)**: The core of the RAG pipeline, built with LangChain. It orchestrates the retrieval of relevant document chunks from ChromaDB and the generation of a final answer by the Gemini LLM.

## Data Flow

### Indexing Flow

1.  A user uploads one or more documents through the Streamlit UI.
2.  The files are passed to the appropriate loader in `load_documents.py`.
3.  The `HybridChunker` processes the loaded documents, creating a list of structured text chunks.
4.  `embedding_utils.py` generates vector embeddings for each chunk.
5.  `chroma_utils.py` stores these embeddings and their metadata in the persistent ChromaDB collection.

### Query Flow

1.  A user submits a question through the Streamlit chat input.
2.  The `ConversationalRetrievalChain` is invoked.
3.  The user's question is used to query the ChromaDB vector store, which returns the most semantically similar document chunks.
4.  The question, chat history, and the retrieved chunks are formatted into a prompt.
5.  The prompt is sent to the Gemini LLM (`gemini-2.0-flash`).
6.  The LLM generates a comprehensive answer based on the provided context.
7.  The answer and the source chunks are displayed back to the user in the Streamlit UI.
