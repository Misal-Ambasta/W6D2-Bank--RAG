# Gemini LLM Prompt Design and Usage

## Model Initialization

The Gemini LLM is used via `langchain_google_genai.ChatGoogleGenerativeAI`.

Example:
```python
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
```

## Prompt Design

Prompts are constructed using LangChain's `PromptTemplate` for flexible and robust RAG workflows.

Example:
```python
from langchain.prompts import PromptTemplate
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a banking expert AI. Use the following context to answer the user question:

Context:
{context}

Question:
{question}

If the answer is not in the context, say 'I don't know.'
"""
)
```

## Fallback/Retry Mechanism

For production, wrap LLM calls with retry logic to handle API timeouts:

```python
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def safe_llm_call(llm, prompt, **kwargs):
    return llm(prompt, **kwargs)
```

## Usage Example

```python
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
response = safe_llm_call(llm, prompt.format(context="...", question="What is amortization?"))
print(response)
```
