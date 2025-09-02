import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain

import google.generativeai as genai
import numpy as np

# ------------------------
# Configure Google Gemini API key
# ------------------------
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# ------------------------
# Extract text from PDF
# ------------------------
def extract_text_from_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

# ------------------------
# Split text into chunks
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
# Dummy embeddings for FAISS (just converts text to numeric vectors)
# ------------------------
def get_dummy_embeddings(texts):
    embeddings = []
    for t in texts:
        arr = np.array([ord(c) for c in t[:512]])  # simple char-to-int vector
        # Pad or truncate to fixed length
        if len(arr) < 512:
            arr = np.pad(arr, (0, 512 - len(arr)))
        else:
            arr = arr[:512]
        embeddings.append(arr)
    return np.array(embeddings)

# ------------------------
# Build FAISS vector store manually
# ------------------------
def build_vectorstore(docs):
    texts = [doc.page_content for doc in docs]
    embeddings = get_dummy_embeddings(texts)
    vectordb = FAISS(embeddings, texts)  # build FAISS manually
    return vectordb

# ------------------------
# Query Gemini
# ------------------------
def query_gemini(prompt):
    response = genai.chat(
        model="chat-bison-001",
        messages=[{"author": "user", "content": prompt}]
    )
    return response.last

# ------------------------
# Streamlit UI
# ------------------------
st.title("📄 PDF Q&A with Google Gemini")

uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files:
    all_text = ""
    for uploaded_file in uploaded_files:
        all_text += extract_text_from_pdf(uploaded_file)

    docs = create_chunks(all_text)
    vectordb = build_vectorstore(docs)
    st.success("✅ PDF processed!")

    query = st.text_input("Ask a question about your documents:")
    if query:
        # Retrieve top 3 relevant chunks using cosine similarity
        relevant_docs = vectordb.similarity_search(query, k=3)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        prompt = f"Answer the question based on the following context:\n{context}\n\nQuestion: {query}"
        answer = query_gemini(prompt)
        st.markdown(f"**🤖 Answer:** {answer}")
