import os
import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from transformers import pipeline

# ------------------------
# LLM Setup using HuggingFace pipeline
# ------------------------
@st.cache_resource
def load_hf_model():
    # Using a free model from HuggingFace
    return pipeline("text-generation", model="tiiuae/falcon-7b-instruct", max_length=512)

hf_model = load_hf_model()

# Wrapper to mimic LangChain chat interface
class HuggingFaceLLM:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def __call__(self, prompt, **kwargs):
        result = self.pipeline(prompt, **kwargs)
        return result[0]['generated_text']

# ------------------------
# PDF Processing
# ------------------------
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

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
# Vector Store using free HuggingFace embeddings
# ------------------------
def build_vectorstore(docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(docs, embedding=embeddings, persist_directory="chroma_db")
    vectordb.persist()
    return vectordb

# ------------------------
# QA Chain
# ------------------------
def build_qa_chain(vectordb):
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    # Use HuggingFace model wrapper
    llm = HuggingFaceLLM(hf_model)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(page_title="📚 Free PDF Chatbot", layout="wide")
st.title("📚 Free PDF Document Chatbot")

# Sidebar: Upload PDFs
st.sidebar.header("Upload PDF Documents")
uploaded_files = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

# Session state
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Process uploaded PDFs
if uploaded_files:
    all_text = ""
    for file in uploaded_files:
        all_text += extract_text_from_pdf(file)

    docs = create_chunks(all_text)
    vectordb = build_vectorstore(docs)
    st.session_state.qa_chain = build_qa_chain(vectordb)
    st.success("✅ Documents processed and ready!")

# Chat interface
query = st.text_input("Ask a question about your documents:")

if query and st.session_state.qa_chain:
    try:
        result = st.session_state.qa_chain({
            "question": query,
            "chat_history": st.session_state.chat_history
        })

        st.session_state.chat_history.append((query, result["answer"]))

        st.markdown(f"**🤖 Answer:** {result['answer']}")

        if result.get("source_documents"):
            with st.expander("📄 Sources"):
                for doc in result["source_documents"]:
                    st.markdown(f"- {doc.page_content[:200]}...")
    except Exception as e:
        st.error(f"⚠️ Error: {e}")
