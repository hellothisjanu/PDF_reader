import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain

import google.generativeai as genai

# ------------------------
# Google Gemini API key
# ------------------------
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    st.error("❌ GEMINI_API_KEY not found! Add it in Streamlit Secrets.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(page_title="📚 PDF Chatbot", layout="wide")
st.title("📚 PDF Chatbot with LangChain + Google Gemini")

# Sidebar: upload PDFs
st.sidebar.header("Upload PDF Documents")
uploaded_files = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ------------------------
# Helper functions
# ------------------------
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
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

def build_vectorstore(docs):
    # Using FAISS for retrieval
    vectordb = FAISS.from_documents(docs, embedding=None)  # embeddings handled by Gemini
    return vectordb

# ------------------------
# Chat wrapper using Gemini API
# ------------------------
def query_gemini(prompt):
    response = genai.chat.create(
        model="gemini-1.5-t",
        messages=[{"author": "user", "content": prompt}],
        temperature=0.7
    )
    return response.last

# ------------------------
# Process uploaded PDFs
# ------------------------
if uploaded_files:
    all_text = ""
    for uploaded_file in uploaded_files:
        all_text += extract_text_from_pdf(uploaded_file)

    docs = create_chunks(all_text)
    vectordb = build_vectorstore(docs)
    st.session_state.vectordb = vectordb
    st.success("✅ Documents processed and ready!")

# ------------------------
# Chat interface
# ------------------------
query = st.text_input("Ask a question about your documents:")

if query and "vectordb" in st.session_state:
    # Simple retrieval: find relevant chunks
    retriever = st.session_state.vectordb.as_retriever(search_kwargs={"k":3})
    relevant_docs = retriever.get_relevant_documents(query)
    context_text = "\n\n".join([doc.page_content for doc in relevant_docs])

    prompt = f"Context:\n{context_text}\n\nQuestion: {query}"
    answer = query_gemini(prompt)

    st.session_state.chat_history.append((query, answer))
    st.markdown(f"**🤖 Answer:** {answer}")
