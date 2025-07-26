from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain_community.vectorstores.utils import filter_complex_metadata
from embedding_utils import embed_texts

class CustomEmbedding:
    def embed_documents(self, texts):
        return embed_texts(texts)
    def embed_query(self, text):
        return embed_texts([text])[0]

import os
from typing import List, Dict, Any

CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_db")

# Helper to convert our chunk dicts to LangChain Document objects
def dicts_to_documents(chunk_dicts: List[Dict[str, Any]]) -> List[Document]:
    docs = []
    for d in chunk_dicts:
        filtered_metadata = filter_complex_metadata(d["metadata"])
        docs.append(Document(page_content=d["text"], metadata=filtered_metadata))
    return docs

def build_chroma_collection(chunks: List[Dict[str, Any]], collection_name="bank-kb"):
    # Prepare documents
    docs = dicts_to_documents(chunks)
    # Use Chroma vector store (latest API)
    embedding = CustomEmbedding()
    vectorstore = Chroma.from_documents(
        docs,
        embedding=embedding,
        persist_directory=CHROMA_PATH,
        collection_name=collection_name
    )
    vectorstore.persist()
    return vectorstore

def get_chroma_vectorstore(collection_name="bank-kb"):
    # Load from persisted vector store (latest API)
    embedding = CustomEmbedding()
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding,
        collection_name=collection_name
    )
    return vectorstore

def semantic_query(query: str, vectorstore, top_k=4):
    # Returns top_k relevant documents for a query
    results = vectorstore.similarity_search(query, k=top_k)
    return results
