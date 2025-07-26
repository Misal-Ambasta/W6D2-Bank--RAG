# Chunking Strategy

Effective chunking is critical for the success of a RAG pipeline, as it determines the quality of the context provided to the LLM. This project uses a structure-aware chunking strategy to handle the complex nature of banking documents, which often contain a mix of text and tables.

## The Challenge with Banking Documents

Standard recursive text splitters are often insufficient for financial documents because they:
- Can split tables across multiple chunks, destroying their meaning.
- Fail to associate text with the correct table or section.
- Lose important structural context (e.g., headers, footers).

## Our Solution: `langchain_docling.HybridChunker`

To address these challenges, we use the `HybridChunker` from the `langchain-docling` library. This chunker is designed to be structure-aware and offers several key advantages:

1.  **Table Preservation**: It identifies tables within documents and treats them as single, coherent chunks. This ensures that the full context of a table is preserved when retrieved.

2.  **Structured Content**: `HybridChunker` parses the document into a structured format, allowing it to differentiate between paragraphs, tables, and other elements.

3.  **Metadata Richness**: Each chunk is automatically enriched with valuable metadata, including:
    - `source`: The original file name.
    - `document_type`: The file format (e.g., PDF, DOCX).
    - `page`: The page number where the chunk originated.
    - `table_ref`: A reference to any table contained within the chunk.

## Implementation

The `HybridChunker` is integrated into our document loading functions in `app/load_documents.py`. When a document is loaded, it is immediately processed by the chunker, which returns a list of `Docling` document objects ready for embedding.

For document types not fully supported by the chunker, the system gracefully falls back to standard text splitting methods, ensuring robustness across all file formats.

This approach ensures that the context provided to the Gemini LLM is as accurate and complete as possible, leading to higher-quality answers.
