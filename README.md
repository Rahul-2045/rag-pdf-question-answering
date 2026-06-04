# RAG PDF Question Answering System

![RAG Architecture](RAG Architecture PNG)

## 1. Project Introduction

This project is a **RAG-based PDF Question Answering system** built using:

- **LangChain**
- **Hugging Face Embeddings**
- **FAISS Vector Database**
- **Hugging Face LLM**
- **Jupyter Notebook**

The system reads PDF documents, splits the text into smaller chunks, converts those chunks into embeddings, stores them in a FAISS vector database, and answers user questions by retrieving the most relevant content from the PDF.

---

## 2. Project Objective

The main objective of this project is to build an AI assistant that can answer questions from PDF documents.

Instead of manually reading a long PDF, the user can ask questions such as:

```text
What is health insurance coverage?
```

```text
Which state had the highest uninsured rate in 2022?
```

```text
What changed in uninsured rates from 2021 to 2022?
```

The system searches the PDF content and generates an answer using the retrieved document context.

---

## 3. What is RAG?

**RAG** stands for **Retrieval Augmented Generation**.

Simple meaning:

> First retrieve relevant information from documents, then generate an answer using an LLM.

Normal LLMs answer from general knowledge, but a RAG system answers from the documents that we provide.

---

## 4. PDF Used in This Project

The sample PDF used in this project is:

**Health Insurance Coverage Status and Type by Geography: 2021 and 2022**

This PDF is published by the **U.S. Census Bureau** and contains data related to:

- Health insurance coverage
- Private insurance
- Public insurance
- Uninsured rate
- State-wise comparison
- Year-wise comparison between 2021 and 2022
- Medicaid expansion and non-expansion states

This PDF is useful for RAG testing because it has clear headings, structured content, facts, percentages, tables, and state-wise comparison data.

---

## 5. Project Architecture

![Project Steps](images/rag_steps.svg)

The project follows this flow:

```text
PDF Documents
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
Generate answer using LLM
```

---

## 6. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Jupyter Notebook | Development environment |
| LangChain | Framework for building RAG pipeline |
| PyPDFDirectoryLoader | Loads PDF documents |
| RecursiveCharacterTextSplitter | Splits large text into chunks |
| HuggingFaceBgeEmbeddings | Converts text into embeddings |
| FAISS | Stores vectors and performs similarity search |
| Hugging Face LLM | Generates final answer |
| Mistral Model | Language model used for answer generation |

---

## 7. Folder Structure

```text
rag-pdf-question-answering/
│
├── rag_pdf_question_answering.ipynb
├── acsbr-015-health-insurance-coverage.pdf
├── README.md
│
└── images/
    ├── rag_architecture.svg
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

### Step 4: Install required libraries

```bash
pip install langchain langchain-community pypdf faiss-cpu sentence-transformers huggingface_hub jupyter
```

### Step 5: Start Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
rag_pdf_question_answering.ipynb
```

---

## 9. Main Code Explanation

### Load PDF documents

```python
from langchain_community.document_loaders import PyPDFDirectoryLoader

loader = PyPDFDirectoryLoader("./")
documents = loader.load()
```

This loads PDF files from the project folder.

---

### Split document into chunks

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

final_documents = text_splitter.split_documents(documents)
```

This breaks the PDF text into smaller parts so that the model can search and process the content properly.

---

### Create embeddings

```python
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

huggingface_embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
```

Embeddings convert text into numerical vectors so that similar meanings can be searched.

---

### Store embeddings in FAISS

```python
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(
    final_documents,
    huggingface_embeddings
)
```

FAISS stores the vectors and helps retrieve similar content when the user asks a question.

---

### Create retriever

```python
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)
```

The retriever fetches the top 3 most relevant document chunks.

---

### Create prompt template

```python
from langchain.prompts import PromptTemplate

prompt_template = """
Use the following context to answer the question.
Answer only based on the given context.

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

The prompt instructs the model to answer only from the retrieved PDF context.

---

### Create RetrievalQA chain

```python
from langchain.chains import RetrievalQA

retrievalQA = RetrievalQA.from_chain_type(
    llm=hf,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)
```

This connects the retriever and the LLM to create the final question-answering system.

---

## 10. Sample Questions

After running the notebook, try asking:

```text
What is health insurance coverage?
```

```text
What was the uninsured rate in 2022?
```

```text
Which state had the lowest uninsured rate in 2022?
```

```text
Which state had the highest uninsured rate in 2022?
```

```text
What is the difference between private and public health insurance?
```

```text
What changed in uninsured rates from 2021 to 2022?
```

---

## 11. Expected Output

Example question:

```text
What is health insurance coverage?
```

Expected answer:

```text
Health insurance coverage refers to comprehensive coverage reported by respondents at the time of interview. The report classifies coverage broadly into private insurance and public insurance.
```

---

## 12. Key Learnings from This Project

By completing this project, you will learn:

- How to load PDF documents
- How to split text into chunks
- How embeddings work
- How to store embeddings in FAISS
- How similarity search works
- How to use a retriever
- How to create a prompt template
- How to connect retriever with LLM
- How to build a basic RAG chatbot

---

## 13. Limitations

Current limitations of this project:

- It works mainly with PDF text.
- Scanned PDFs may need OCR.
- The model quality depends on the selected LLM.
- If wrong chunks are retrieved, the answer may be weak.
- For production use, API keys should be stored securely.
- For large documents, metadata and reranking should be added.

---

## 14. Future Improvements

Possible improvements:

- Add Streamlit user interface
- Add PDF upload feature
- Add source citation in final answer
- Store FAISS index locally
- Add support for multiple PDFs
- Add metadata such as page number and document name
- Use an instruction-tuned LLM
- Add chat history
- Add hybrid search
- Add reranking

---

## 15. Repository Name

Recommended repository name:

```text
rag-pdf-question-answering
```

---

## 16. GitHub Description

```text
A RAG-based PDF Question Answering system using LangChain, Hugging Face embeddings, FAISS vector store, and Hugging Face LLM to answer questions from PDF documents.
```

---

## 17. Final Summary

This project demonstrates how to build a basic RAG pipeline for PDF question answering.

The system loads a PDF, splits it into chunks, converts those chunks into embeddings, stores them in FAISS, retrieves relevant chunks based on the user question, and generates an answer using a Hugging Face LLM.

In simple words:

> This is a PDF-based AI chatbot that answers questions from documents.
