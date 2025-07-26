from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
import os

CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_db")

# Set up memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def get_chroma_vectorstore(collection_name="bank-kb"):
    return Chroma(persist_directory=CHROMA_PATH, collection_name=collection_name)

def build_conversational_chain(vectorstore=None, llm=None, k=4, **kwargs):
    if vectorstore is None:
        vectorstore = get_chroma_vectorstore()
    if llm is None:
        # Gemini 2.0 Flash LLM
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever,
        memory=memory,
        return_source_documents=True,
        **kwargs
    )
    return chain

def example_rag_flow():
    vectorstore = get_chroma_vectorstore()
    chain = build_conversational_chain(vectorstore=vectorstore)
    chat_history = []
    query = "What are the regulatory requirements for loan amortization?"
    result = chain({"question": query, "chat_history": chat_history})
    return result
