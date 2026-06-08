import os
import tempfile
import streamlit as st
from pypdf import PdfReader

from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA


# Page Config
st.set_page_config(
    page_title="AI RAG Assistant",
    page_icon="🤖",
    layout="wide"
)


# Custom Animated CSS
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(-45deg, #0f172a, #1e293b, #111827, #020617);
        background-size: 400% 400%;
        animation: gradientBG 12s ease infinite;
        color: white;
    }

    @keyframes gradientBG {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }

    .main-title {
        text-align: center;
        font-size: 48px;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #a78bfa, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: glow 2s ease-in-out infinite alternate;
        margin-bottom: 5px;
    }

    @keyframes glow {
        from { text-shadow: 0 0 10px #38bdf8; }
        to { text-shadow: 0 0 25px #f472b6; }
    }

    .sub-title {
        text-align: center;
        font-size: 18px;
        color: #cbd5e1;
        margin-bottom: 35px;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 25px;
        border: 1px solid rgba(255, 255, 255, 0.15);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
        backdrop-filter: blur(12px);
        animation: fadeIn 0.8s ease-in-out;
    }

    @keyframes fadeIn {
        from {opacity: 0; transform: translateY(20px);}
        to {opacity: 1; transform: translateY(0);}
    }

    .status-box {
        background: rgba(34, 197, 94, 0.15);
        border-left: 5px solid #22c55e;
        padding: 15px;
        border-radius: 12px;
        color: #dcfce7;
        margin-top: 15px;
    }

    .warning-box {
        background: rgba(251, 191, 36, 0.15);
        border-left: 5px solid #fbbf24;
        padding: 15px;
        border-radius: 12px;
        color: #fef3c7;
        margin-top: 15px;
    }

    .answer-box {
        background: rgba(14, 165, 233, 0.12);
        border: 1px solid rgba(56, 189, 248, 0.35);
        padding: 25px;
        border-radius: 18px;
        color: #e0f2fe;
        font-size: 17px;
        line-height: 1.7;
        animation: fadeIn 0.8s ease-in-out;
    }

    .source-box {
        background: rgba(255, 255, 255, 0.07);
        border-radius: 14px;
        padding: 15px;
        margin-top: 10px;
        color: #e5e7eb;
        font-size: 14px;
    }

    div.stButton > button {
        width: 100%;
        border-radius: 14px;
        background: linear-gradient(90deg, #2563eb, #7c3aed, #db2777);
        color: white;
        font-weight: 700;
        border: none;
        padding: 14px;
        font-size: 17px;
        transition: 0.3s;
    }

    div.stButton > button:hover {
        transform: scale(1.03);
        box-shadow: 0 0 20px rgba(168, 85, 247, 0.8);
    }

    .stTextInput > div > div > input {
        border-radius: 14px;
        padding: 14px;
        background-color: rgba(255,255,255,0.12);
        color: white;
    }

    .stFileUploader {
        background: rgba(255,255,255,0.06);
        padding: 15px;
        border-radius: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Header

st.markdown(
    '<div class="main-title">🤖 AI PDF RAG Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Upload a PDF and ask questions. Powered by FAISS + Ollama Local LLM.</div>',
    unsafe_allow_html=True
)


# Session State
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "retrieval_chain" not in st.session_state:
    st.session_state.retrieval_chain = None

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False


# Load Ollama Embeddings
@st.cache_resource
def load_embeddings():
    return OllamaEmbeddings(
        model="nomic-embed-text"
    )


# PDF Text Extraction
def extract_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    pages_text = []

    for page_no, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            pages_text.append(
                {
                    "page": page_no,
                    "text": text
                }
            )

    return pages_text


# Manual Text Splitter
def split_text_into_chunks(pages_text, chunk_size=1000, chunk_overlap=200):
    chunks = []
    metadatas = []

    for page_data in pages_text:
        text = page_data["text"]
        page_no = page_data["page"]

        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]

            if chunk.strip():
                chunks.append(chunk)
                metadatas.append({"page": page_no})

            start = end - chunk_overlap

            if start < 0:
                start = 0

            if start >= len(text):
                break

    return chunks, metadatas



# Process Uploaded PDF
def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_pdf_path = temp_file.name

    try:
        pages_text = extract_pdf_text(temp_pdf_path)

        if not pages_text:
            raise ValueError("No readable text found in this PDF. It may be scanned and may require OCR.")

        chunks, metadatas = split_text_into_chunks(
            pages_text,
            chunk_size=1000,
            chunk_overlap=200
        )

        embeddings = load_embeddings()

        vectorstore = FAISS.from_texts(
            texts=chunks,
            embedding=embeddings,
            metadatas=metadatas
        )

        return vectorstore, len(pages_text), len(chunks)

    finally:
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)


# Create RAG Chain
def create_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    llm = OllamaLLM(
        model="phi3"
    )

    prompt_template = """
You are a helpful PDF assistant.

Answer the question using only the context given below.
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

    retrieval_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    return retrieval_chain


# Layout
left_col, right_col = st.columns([1, 2])


with left_col:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    st.subheader("📄 Upload PDF")

    uploaded_file = st.file_uploader(
        "Choose your PDF file",
        type=["pdf"]
    )

    if uploaded_file is not None:
        st.info(f"Selected file: {uploaded_file.name}")

    process_button = st.button("🚀 Process PDF")

    if process_button:
        if uploaded_file is None:
            st.markdown(
                '<div class="warning-box">⚠️ Please upload a PDF first.</div>',
                unsafe_allow_html=True
            )
        else:
            with st.spinner("Reading PDF, creating chunks and building vector database..."):
                try:
                    vectorstore, page_count, chunk_count = process_pdf(uploaded_file)

                    st.session_state.vectorstore = vectorstore
                    st.session_state.retrieval_chain = create_rag_chain(vectorstore)
                    st.session_state.pdf_processed = True

                    st.markdown(
                        f"""
                        <div class="status-box">
                        ✅ PDF processed successfully!<br>
                        📘 Pages Loaded: {page_count}<br>
                        🧩 Chunks Created: {chunk_count}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                except Exception as e:
                    st.error(f"Error while processing PDF: {e}")

    st.markdown("</div>", unsafe_allow_html=True)


with right_col:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    st.subheader("💬 Ask Question")

    question = st.text_input(
        "Enter your question from the PDF",
        placeholder="Example: What is health insurance coverage?"
    )

    ask_button = st.button("✨ Generate Answer")

    if ask_button:
        if not st.session_state.pdf_processed:
            st.markdown(
                '<div class="warning-box">⚠️ Please upload and process a PDF first.</div>',
                unsafe_allow_html=True
            )
        elif question.strip() == "":
            st.markdown(
                '<div class="warning-box">⚠️ Please enter a question.</div>',
                unsafe_allow_html=True
            )
        else:
            with st.spinner("Searching PDF and generating answer..."):
                try:
                    result = st.session_state.retrieval_chain.invoke(
                        {"query": question}
                    )

                    st.markdown("### ✅ Answer")
                    st.markdown(
                        f'<div class="answer-box">{result["result"]}</div>',
                        unsafe_allow_html=True
                    )

                    st.markdown("### 📚 Source Chunks")

                    for i, doc in enumerate(result["source_documents"], start=1):
                        source_text = doc.page_content[:700]
                        page_number = doc.metadata.get("page", "N/A")

                        st.markdown(
                            f"""
                            <div class="source-box">
                            <b>Source Chunk {i}</b><br>
                            <b>Page:</b> {page_number}<br><br>
                            {source_text}...
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                except Exception as e:
                    st.error(f"Error while generating answer: {e}")

    st.markdown("</div>", unsafe_allow_html=True)


# Footer
st.markdown(
    """
    <br>
    <div style="text-align:center; color:#94a3b8;">
        Built with LangChain | FAISS | Ollama | Streamlit
    </div>
    """,
    unsafe_allow_html=True
)