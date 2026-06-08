# RAG PDF System with Ollama

![RAG PDF Question Answering Architecture](images/rag_pdf_architecture.png)

## 1. Project Introduction

This project is a **RAG-based PDF Question Answering system** built using:

- Python
- LangChain
- Hugging Face BGE Embeddings
- FAISS Vector Database
- Ollama Local LLM
- Jupyter Notebook
- Streamlit UI

The system reads PDF documents, splits the text into smaller chunks, converts those chunks into embeddings, stores them in a FAISS vector database, and answers user questions by retrieving the most relevant PDF content.

This updated version does **not require a Hugging Face API token** because the final answer is generated using a local Ollama model.

---

## 2. Project Objective

The main objective of this project is to build an AI assistant that can answer questions from PDF documents.

Instead of manually reading a long PDF, the user can upload or load a PDF and ask questions such as:

```text
What is health insurance coverage?
Which state had the highest uninsured rate in 2022?
What changed in uninsured rates from 2021 to 2022?
```

The system searches the PDF content and generates an answer using the retrieved document context.

---

## 3. What is RAG?

**RAG** stands for **Retrieval Augmented Generation**.

Simple meaning:

> First retrieve relevant information from documents, then generate an answer using an LLM.

Normal LLMs answer from general knowledge, but a RAG system answers from the documents that we provide.

In this project:

```text
PDF → Text Chunks → Embeddings → FAISS Search → Ollama LLM → Final Answer
```

---

## 4. PDF Used in This Project

The sample PDF used in this project is:

**Health Insurance Coverage Status and Type by Geography: 2021 and 2022**

This PDF contains information related to:

- Health insurance coverage
- Private insurance
- Public insurance
- Uninsured rate
- State-wise comparison
- Year-wise comparison between 2021 and 2022
- Medicaid expansion and non-expansion states

This PDF is useful for RAG testing because it contains clear headings, facts, percentages, tables, and state-wise comparison data.

---

## 5. Project Architecture

![Project Steps](images/rag_steps.svg)

The project follows this flow:

```text
PDF Document
   ↓
Load PDF
   ↓
Split text into chunks
   ↓
Create embeddings
   ↓
Store embeddings in FAISS
   ↓
User asks a question
   ↓
Retrieve relevant chunks
   ↓
Generate answer using Ollama LLM
   ↓
Show answer with source chunks
```

---

## 6. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Jupyter Notebook | Development and testing |
| Streamlit | Web app user interface |
| LangChain | Framework for building RAG pipeline |
| PyPDFLoader | Loads PDF document |
| RecursiveCharacterTextSplitter | Splits large text into smaller chunks |
| HuggingFaceBgeEmbeddings | Converts text chunks into embeddings |
| FAISS | Stores vectors and performs similarity search |
| Ollama | Runs local LLM without API token |
| Phi-3 / Llama 3.2 | Local LLM model for answer generation |

---

## 7. Folder Structure

```text
rag-pdf-question-answering/
│
├── rag_pdf_With_llm.ipynb
├── app.py
├── health-insurance-coverage.pdf
├── README.md
│
└── images/
    ├── rag_pdf_architecture.png
    └── rag_steps.svg
```

---

## 8. Installation Steps

### Step 1: Clone the repository

```bash
git clone https://github.com/your-username/rag-pdf-question-answering.git
cd rag-pdf-question-answering
```

### Step 2: Create a virtual environment

```bash
python -m venv venv
```

### Step 3: Activate virtual environment

For Windows:

```bash
venv\Scripts\activate
```

For Linux or Mac:

```bash
source venv/bin/activate
```

### Step 4: Install required Python libraries

```bash
pip install langchain langchain-community langchain-core langchain-text-splitters langchain-classic
pip install pypdf faiss-cpu sentence-transformers streamlit
pip install langchain-ollama
```

---

## 9. Install Ollama

This project uses Ollama for local LLM answer generation.

### Step 1: Download Ollama

Download Ollama from:

```text
https://ollama.com/download
```

Install Ollama for Windows.

### Step 2: Verify Ollama installation

Open Command Prompt and run:

```bash
ollama --version
```

If Ollama is installed correctly, it will show the version.

### Step 3: Download local model

For a normal laptop, use Phi-3:

```bash
ollama pull phi3
```

Optional model:

```bash
ollama pull llama3.2
```

### Step 4: Test Ollama

```bash
ollama run phi3
```

Ask:

```text
What is health insurance coverage?
```

If it answers, Ollama is working.

---

## 10. Main Code Explanation

### Import required libraries

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_ollama import OllamaLLM

import os
from pathlib import Path
```

These imports are required for:

- Loading PDF
- Splitting text
- Creating embeddings
- Creating FAISS vector database
- Creating prompt
- Building RetrievalQA chain
- Using Ollama local LLM

---

### Load PDF document

```python
pdf_path = r"E:\Rahul Verma\document\D drive\PROJECTS\Vscode\RAG\health-insurance-coverage.pdf"

loader = PyPDFLoader(pdf_path)
documents = loader.load()

print(f"Total PDF pages loaded: {len(documents)}")
print(documents[0].page_content[:500])
```

This code loads the PDF file and extracts text page by page.

Update `pdf_path` according to your local PDF file location.

---

### Split document into chunks

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

final_documents = text_splitter.split_documents(documents)

print(f"Total chunks created: {len(final_documents)}")
```

This breaks the PDF text into smaller parts.

Meaning:

```text
chunk_size=1000     → each chunk contains around 1000 characters
chunk_overlap=200   → 200 characters are repeated between chunks
```

Chunk overlap helps preserve context between chunks.

---

### Create embeddings

```python
huggingface_embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
```

This creates embeddings using the BGE small English embedding model.

Important:

> This does not require a Hugging Face API token.

The embedding model is used locally to convert text chunks into numerical vectors.

---

### Store embeddings in FAISS

```python
vectorstore = FAISS.from_documents(
    final_documents,
    huggingface_embeddings
)
```

FAISS stores all document vectors and helps search similar chunks when a user asks a question.

---

### Test similarity search

```python
query = "What is health insurance coverage?"

relevant_documents = vectorstore.similarity_search(query, k=3)

for i, doc in enumerate(relevant_documents, start=1):
    print(f"\n--- Relevant Chunk {i} ---")
    print(doc.page_content[:700])
```

This retrieves the top 3 relevant chunks from the PDF.

This confirms that the retrieval part of RAG is working.

---

### Create retriever

```python
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)
```

The retriever fetches the most relevant chunks from FAISS.

Here:

```text
k=3
```

means it will retrieve the top 3 matching chunks.

---

### Create Ollama LLM

```python
llm = OllamaLLM(model="phi3")
```

This connects the project with the local Ollama model.

Before running this, make sure this command has been executed in Command Prompt:

```bash
ollama pull phi3
```

---

### Quick LLM test

```python
response = llm.invoke("What is health insurance coverage?")
print(response)
```

This tests whether the Ollama model is working.

This test does not use the PDF. It only checks the LLM.

---

### Create prompt template

```python
prompt_template = """
You are a helpful assistant answering questions from a PDF document.
Use only the given context to answer the question.
If the answer is not available in the context, say:
"The document does not mention this information."

Context:
{context}

Question:
{question}

Answer:
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)
```

This prompt tells the model to answer only from the PDF context.

If the answer is not found in the retrieved chunks, the model should not make up an answer.

---

### Create RetrievalQA chain

```python
retrievalQA = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)
```

This connects:

```text
Retriever + Prompt + Ollama LLM
```

Meaning:

- Retriever gets relevant PDF chunks.
- Prompt formats the question and context.
- Ollama LLM generates the final answer.
- Source documents are also returned.

---

### Ask question using full RAG

```python
query = "What is health insurance coverage?"

result = retrievalQA.invoke({"query": query})

print("Answer:")
print(result["result"])
```

This is the final RAG question-answering step.

It performs:

```text
Question → Search FAISS → Retrieve chunks → Send to Ollama → Generate answer
```

---

### Show source documents

```python
for i, doc in enumerate(result["source_documents"], start=1):
    print(f"\n--- Source Document {i} ---")
    print(doc.page_content[:700])
```

This shows which PDF chunks were used to generate the answer.

---

## 11. Streamlit App

This project also supports an attractive Streamlit app.

### Run the app

```bash
streamlit run app.py
```

The app provides:

- PDF upload
- Process PDF button
- Question input box
- AI-generated answer
- Source chunks
- Animated background
- Glass-style UI

---

## 12. Sample Questions

After running the notebook or Streamlit app, try asking:

```text
What is health insurance coverage?
What was the uninsured rate in 2022?
Which state had the lowest uninsured rate in 2022?
Which state had the highest uninsured rate in 2022?
What is the difference between private and public health insurance?
What changed in uninsured rates from 2021 to 2022?
```

---

## 13. Expected Output

Example question:

```text
What is health insurance coverage?
```

Expected answer:

```text
Health insurance coverage refers to comprehensive coverage that helps cover basic health care needs. The document classifies health insurance coverage into private insurance and public insurance.
```

The exact answer may vary depending on the retrieved chunks and the selected Ollama model.

---

## 14. Key Learnings from This Project

By completing this project, you will learn:

- How to load PDF documents using LangChain
- How to split PDF text into chunks
- How embeddings work
- How to create vector embeddings using BGE embeddings
- How to store embeddings in FAISS
- How similarity search works
- How to create a retriever
- How to use a local LLM with Ollama
- How to create a prompt template
- How to connect retriever with LLM
- How to build a basic RAG chatbot
- How to create a Streamlit interface for RAG

---

## 15. Limitations

Current limitations of this project:

- It works mainly with text-based PDFs.
- Scanned PDFs may need OCR.
- The quality of the answer depends on the selected Ollama model.
- If wrong chunks are retrieved, the answer may be weak.
- For large PDFs, retrieval performance may need optimization.
- The current version does not include reranking.
- Chat history is not added yet.
- FAISS index is created again unless saved locally.

---

## 16. Future Improvements

Possible improvements:

- Add chat history
- Add multiple PDF support
- Save and reload FAISS index locally
- Add source citation with page number
- Add hybrid search
- Add reranking
- Add OCR support for scanned PDFs
- Add authentication for app users
- Add better UI with chat-style interface
- Add document summary feature
- Add downloadable answer report

---

## 17. Repository Name

Recommended repository name:

```text
rag-pdf-question-answering-ollama
```

---

## 18. GitHub Description

```text
A RAG-based PDF Question Answering system using LangChain, Hugging Face BGE embeddings, FAISS vector store, Ollama local LLM, and Streamlit UI to answer questions from PDF documents without using API tokens.
```

---

## 19. Final Summary

This project demonstrates how to build a complete basic RAG pipeline for PDF question answering.

The system loads a PDF, splits it into chunks, converts those chunks into embeddings, stores them in FAISS, retrieves relevant chunks based on the user question, and generates an answer using a local Ollama LLM.

In simple words:

> This is a PDF-based AI chatbot that answers questions from documents without needing any paid API key or Hugging Face token.
