# RAG Implementation for Banking Knowledge Base  
Using LangChain ≥ 0.3, ChromaDB, SentenceTransformers (MiniLM), Gemini LLM, Streamlit, and langchain-docling ([integration doc](https://python.langchain.com/docs/integrations/document_loaders/docling/))

---

## Phase 1: Project Setup

- [x] Create GitHub repository and basic folder structure (`/app`, `/docs`, `/data`)
- [x] Set up virtual environment and install dependencies:
  - langchain
  - chromadb
  - sentence-transformers
  - streamlit
  - google-generativeai
  - langchain-docling
- [x] Create `.env` for configuration values:
  - Gemini API key or credentials
  - Chunk size, embedding model path, collection name, etc.
- [x] Initialize Docling project using `docling init` (via langchain-docling)
- [x] Create documentation sections:
  - `architecture.md`
  - `retrieval.md`
  - `chunking-strategy.md`
  - `cost-analysis.md`
  - `prompts.md`

---

## Phase 2: Document Loading and Preprocessing

- [x] Implement support for banking document types:
  - PDF using PyPDFLoader or UnstructuredPDFLoader
  - DOCX using UnstructuredFileLoader
  - Excel using Pandas + custom parser
- [x] Normalize extracted text (remove noise, fix encoding issues)
- [x] Tag each document chunk with metadata:
  - source
  - page number
  - document type
  - table ID (if applicable)

---

## Phase 3: Chunking Strategy with Table Preservation

- [x] Use `langchain_docling.HybridChunker` as the structure-aware chunker (recommended by project review)
- [x] Develop a custom chunker to:
  - Keep table headers and rows together
  - Preserve inline references such as “See Table 3.2”
  - Avoid splitting data across pages
- [x] Store mapping of original document structure in metadata
- [x] Document the approach in `docs/chunking-strategy.md`

---

## Phase 4: Embedding Generation with Sentence Transformers (MiniLM)

- [x] Load MiniLM embedding model:
  - `all-MiniLM-L6-v2` from sentence-transformers
- [x] Encode text chunks using:
  ```python
  from sentence_transformers import SentenceTransformer
  model = SentenceTransformer("all-MiniLM-L6-v2")
  embeddings = model.encode(texts, convert_to_tensor=True)
  ```
- [x] Cache embeddings using content hash to avoid recomputation
- [x] Log embedding details in `docs/retrieval.md`

---

## Phase 5: ChromaDB Vector Store Setup

- [x] Initialize local Chroma collection and store vectors:
  - Text content
  - Embeddings
  - Metadata (source, page, document type, table reference)
- [x] Use LangChain integration:
  ```python
  from langchain.vectorstores import Chroma
  vectorstore = Chroma.from_documents(docs, embedding_function, collection_name="bank-kb")
  ```
- [x] Run semantic retrieval test queries
- [x] Document ChromaDB usage in `docs/retrieval.md`

---

## Phase 6: Retrieval Chain with LangChain

- [x] Set up ConversationalRetrievalChain 
- [x] Add memory support using ConversationBufferMemory
- [x] Connect Chroma retriever and configure search parameters:
  - k = 4
  - threshold or filter logic if needed
- [x] Save example RAG flow in `docs/retrieval.md`

---

## Phase 7: Gemini LLM Integration

- [x] Use Gemini via Google Vertex AI or Google Generative AI SDK
- [x] Initialize model:
  ```python
  from langchain_google_genai import ChatGoogleGenerativeAI
  llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
  ```
- [x] Design prompts using LangChain PromptTemplate
- [x] Add fallback or retry mechanism for API timeouts
- [x] Document usage and examples in `docs/prompts.md`

---

## Phase 8: Streamlit Interface

- [x] Build basic UI:
  - File/document uploader
  - Chat input and response area
  - Display top matching chunks with metadata
- [x] Optionally include:
  - Feedback thumbs (👍 / 👎)
  - Filters (document type, topic)
- [x] Capture UI features in `docs/frontend.md`

---

## Phase 9: Cost Optimization and ROI Analysis

- [x] Create cost breakdown in `docs/cost-analysis.md`
- [x] Include:
  - Gemini API usage estimates
  - Self-hosted ChromaDB cost (typically $0)
  - MiniLM embedding computation (done offline)
- [x] Compare premium vs. optimized vs. hybrid stack:
  - GPT-4 + Pinecone + Cloud APIs
  - Gemini + ChromaDB + MiniLM
  - Hybrid local+cloud combo
- [x] Estimate monthly cost for 1,000 queries/day
- [x] Provide ROI suggestions and scaling scenarios

---

## Phase 10: Final Deliverables

- [x] LangChain-based RAG pipeline using:
  - Gemini LLM
  - ChromaDB
  - MiniLM Embeddings
- [x] Streamlit application with banking use case UI
- [x] Table-aware custom chunking logic
- [x] Complete technical documentation via langchain-docling:
  - Architecture
  - Retrieval logic
  - Chunking strategy
  - Prompt design
  - Cost optimization
- [x] GitHub repository with README and deployment steps

---

## Optional Enhancements (Bonus)

- [ ] Enable role-based access for internal banking teams
- [ ] Add chatbot feedback logging for performance review
- [ ] Schedule automatic re-indexing of documents
- [ ] Export Q&A history and citations as CSV or PDF

---

Note: Use `docling serve` to preview and `docling build` to generate static docs.
