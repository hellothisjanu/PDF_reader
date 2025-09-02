import sys
import os
import streamlit as st
import tempfile
import PyPDF2
import subprocess

# --- SQLite Patch (fix for Streamlit Cloud / ChromaDB) ---
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except Exception:
    pass

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings  # ✅ FREE embeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import BaseLLM

# ------------------------
# Custom Ollama LLM Wrapper
# ------------------------
class OllamaLLM(BaseLLM):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def _call(self, prompt: str) -> str:
        result = subprocess.run(
            ["ollama", "run", self.model_name, "--text", prompt],
            capture_output=True,
            text=True
        )
        return result.stdout.strip()

# ------------------------
# API Key setup (stored in .streamlit/secrets.toml)
# ------------------------
# No API key needed for Ollama

# ------------------------
# Utility: Extract text from PDF
# ------------------------
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

# ------------------------
# Utility: Split text into chunks
# ------------------------
def create_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = splitter.split_text(text)
    docs = [Document(page_content=chunk) for chunk in chunks]
    return docs

# ------------------------
# Build Vector Store (FREE HuggingFace embeddings)
# ------------------------
def build_vectorstore(docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(docs, embedding=embeddings, persist_directory="chroma_db")
    vectordb.persist()
    return vectordb

# ------------------------
# Build QA Chain with Ollama
# ------------------------
def build_qa_chain(vectordb):
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    llm = OllamaLLM(model_name="llama2")  # Replace with your desired model
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(page_title="📚 Universal Document Chatbot", layout="wide")
st.title("📚 Universal Document Intelligence Chatbot")

# Sidebar
st.sidebar.header("Upload PDF Documents")
uploaded_files = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

# Session State for persistence
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Step 1: Process uploaded files
if uploaded_files:
    all_text = ""
    for uploaded_file in uploaded_files:
        text = extract_text_from_pdf(uploaded_file)
        all_text += text

    # Split into chunks
    docs = create_chunks(all_text)

    # Build Vector DB
    vectordb = build_vectorstore(docs)

    # Build QA Chain
    st.session_state.qa_chain = build_qa_chain(vectordb)
    st.success("✅ Documents processed and ready!")

# Step 2: Chat interface
query = st.text_input("Ask a question about your documents:")

if query and st.session_state.qa_chain:
    try:
        result = st.session_state.qa_chain({
            "question": query,
            "chat_history": st.session_state.chat_history
        })

        st.session_state.chat_history.append((query, result["answer"]))

        # Show answer
        st.markdown(f"**🤖 Answer:** {result['answer']}")

        # Show sources
        if result.get("source_documents"):
            with st.expander("📄 Sources"):
                for doc in result["source_documents"]:
                    st.markdown(f"- {doc.page_content[:200]}...")  # preview of chunk
    except Exception as e:
        st.error(f"⚠️ Error while querying: {e}")
