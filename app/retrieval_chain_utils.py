import os
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from pinecone_utils import get_pinecone_vectorstore
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up memory with output key specification
memory = ConversationBufferMemory(
    memory_key="chat_history", 
    return_messages=True,
    output_key="answer"  # Specify which output to store in memory
)

def build_conversational_chain(vectorstore=None, llm=None, k=4, **kwargs):
    if vectorstore is None:
        vectorstore = get_pinecone_vectorstore()
    if llm is None:
        # Gemini 2.0 Flash LLM with explicit API key loading
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            print("Warning: GOOGLE_API_KEY not found. Please set it in your .env file.")
            print("For now, returning None LLM - you'll need to provide one manually.")
            return None
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash", 
                temperature=0.3,
                google_api_key=google_api_key
            )
        except Exception as e:
            print(f"Error initializing Google LLM: {e}")
            return None
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever,
        memory=memory,
        return_source_documents=True,
        verbose=True,  # Add verbose for debugging
        **kwargs
    )
    return chain

def example_rag_flow():
    vectorstore = get_pinecone_vectorstore()
    chain = build_conversational_chain(vectorstore=vectorstore)
    chat_history = []
    query = "What are the regulatory requirements for loan amortization?"
    result = chain({"question": query, "chat_history": chat_history})
    return result
