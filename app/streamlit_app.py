import streamlit as st
from load_documents import load_pdf, load_docx, load_excel
from pinecone_utils import build_pinecone_index, get_pinecone_vectorstore
from retrieval_chain_utils import build_conversational_chain
from langchain.prompts import PromptTemplate
import os

st.set_page_config(page_title="Banking RAG Chatbot", layout="wide")
st.title("Banking Knowledge Base Chatbot")

# Sidebar: File uploader
doc_files = st.sidebar.file_uploader(
    "Upload documents (PDF, DOCX, XLSX)",
    type=["pdf", "docx", "xlsx", "xls"],
    accept_multiple_files=True
)
if st.sidebar.button("Process Documents") and doc_files:
    all_chunks = []
    for file in doc_files:
        suffix = file.name.lower().split(".")[-1]
        temp_dir = os.path.join(os.path.dirname(__file__), "..", "temp_files")
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        path = os.path.join(temp_dir, file.name)
        with open(path, "wb") as f:
            f.write(file.getvalue())
        from pathlib import Path as PathLib
        path_obj = PathLib(path)
        if suffix == "pdf":
            all_chunks.extend(load_pdf(path_obj))
        elif suffix == "docx":
            all_chunks.extend(load_docx(path_obj))
        elif suffix in ["xls", "xlsx"]:
            all_chunks.extend(load_excel(path_obj))
    if all_chunks:
        print(f"Total chunks to index: {len(all_chunks)}")
        # Filter out any non-dict chunks before processing
        valid_chunks = [chunk for chunk in all_chunks if isinstance(chunk, dict) and 'text' in chunk and 'metadata' in chunk]
        print(f"Valid chunks: {len(valid_chunks)} out of {len(all_chunks)}")
        
        if valid_chunks:
            build_pinecone_index(valid_chunks)
            st.sidebar.success(f"Processed and indexed {len(valid_chunks)} valid chunks.")
        else:
            st.sidebar.error("No valid chunks found. Check document processing.")
    else:
        st.sidebar.warning("No valid chunks extracted.")

# Main: Chat interface
st.subheader("Chat with your banking documents")
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Optional filters
col1, col2 = st.columns(2)
with col1:
    doc_type_filter = st.selectbox("Filter by document type", ["All", "pdf", "docx", "excel"])
with col2:
    topic_filter = st.text_input("Filter by topic (optional)")

user_input = st.text_input("Ask a question:")
if st.button("Send") and user_input:
    try:
        vectorstore = get_pinecone_vectorstore()
        chain = build_conversational_chain(vectorstore=vectorstore)
        if chain is None:
            st.error("❌ Cannot create conversational chain. Please check your GOOGLE_API_KEY in the .env file.")
            st.info("💡 Make sure you have a .env file with: GOOGLE_API_KEY=your_api_key_here")
        else:
            # Optionally filter docs before retrieval (not implemented here)
            result = chain({"question": user_input, "chat_history": st.session_state["chat_history"]})
            st.session_state["chat_history"].append((user_input, result["answer"]))
            
            # Display the result
            st.markdown(f"**Bot:** {result['answer']}")
            # Show top chunks
            st.markdown("### Top Matching Chunks:")
            for doc in result.get("source_documents", [])[:4]:
                meta = doc.metadata
                if doc_type_filter != "All" and meta.get("document_type") != doc_type_filter:
                    continue
                if topic_filter and topic_filter.lower() not in doc.page_content.lower():
                    continue
                st.info(f"**Source:** {meta.get('source')} | **Type:** {meta.get('document_type')} | **Page:** {meta.get('page', 'N/A')}")
                st.text(doc.page_content[:300] + "...")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.info("💡 Please check your API keys and try again.")
        # Feedback thumbs
        st.write("Feedback:", "👍", "👎")

# Show chat history
if st.session_state["chat_history"]:
    st.markdown("---")
    st.markdown("### Chat History")
    for q, a in st.session_state["chat_history"]:
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Bot:** {a}")
