import os
from typing import List, Dict, Any
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.docstore.document import Document
from langchain_community.vectorstores.utils import filter_complex_metadata
from embedding_utils import embed_texts
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Pinecone Initialization ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY must be set as an environment variable.")

pc = Pinecone(api_key=PINECONE_API_KEY)

# --- Constants ---
DEFAULT_INDEX_NAME = "bank-kb"
VECTOR_DIM = 384  # MiniLM-L6-v2 output size

# --- Embedding Wrapper ---
from langchain.embeddings.base import Embeddings

class CustomEmbedding(Embeddings):
    """Custom LangChain embedding wrapper for our embedding_utils."""
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = embed_texts(texts)
        # Convert numpy arrays to Python lists for Pinecone compatibility
        return [emb.tolist() if hasattr(emb, 'tolist') else list(emb) for emb in embeddings]

    def embed_query(self, text: str) -> List[float]:
        embedding = embed_texts([text])[0]
        # Convert numpy array to Python list for Pinecone compatibility
        return embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)

# --- Core Pinecone Functions ---

def build_pinecone_index(chunks: List[Dict[str, Any]], index_name: str = DEFAULT_INDEX_NAME):
    """Creates a Pinecone index if it doesn't exist and upserts document chunks."""
    if index_name not in pc.list_indexes().names():
        print(f"Creating new serverless index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=VECTOR_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
    
    index = pc.Index(index_name)
    
    # Convert chunk dicts to LangChain Documents
    documents = []
    valid_chunks = 0
    
    for i, chunk in enumerate(chunks):
        try:
            # Only process valid dictionary chunks with required keys
            if isinstance(chunk, dict) and 'text' in chunk and 'metadata' in chunk:
                text_content = chunk['text']
                if text_content and text_content.strip():  # Ensure non-empty text
                    documents.append(
                        Document(
                            page_content=text_content.strip(), 
                            metadata=filter_complex_metadata(chunk['metadata'])
                        )
                    )
                    valid_chunks += 1
                else:
                    print(f"Skipping chunk {i}: empty text content")
            else:
                print(f"Skipping chunk {i}: invalid format (type: {type(chunk)})")
        except Exception as e:
            print(f"Error processing chunk {i}: {e}")
            continue
    
    print(f"Processed {valid_chunks} valid chunks out of {len(chunks)} total chunks")
    
    # Use LangChain's PineconeVectorStore to handle embedding and upserting
    vectorstore = get_pinecone_vectorstore(index_name=index_name)
    vectorstore.add_documents(documents, batch_size=100)
    print(f"Successfully upserted {len(documents)} documents to index '{index_name}'.")
    return index

def get_pinecone_vectorstore(index_name: str = DEFAULT_INDEX_NAME) -> PineconeVectorStore:
    """Initializes and returns a LangChain PineconeVectorStore object."""
    embedding = CustomEmbedding()
    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embedding,
        namespace=None,  # You can specify a namespace if needed
        pinecone_api_key=PINECONE_API_KEY
    )
    return vectorstore

def semantic_query(query: str, vectorstore: PineconeVectorStore, top_k: int = 4) -> List[Document]:
    """Performs a semantic similarity search on the vector store."""
    results = vectorstore.similarity_search(query, k=top_k)
    return results
