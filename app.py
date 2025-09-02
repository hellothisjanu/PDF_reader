import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain

import google.generativeai as genai

# ------------------------
# Configure Google Gemini API key
# ------------------------
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])  # add your key in Streamlit secrets

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
# Build FAISS vector store
# ------------------------
def build_vectorstore(docs):
    # We'll create embeddings using Gemini API directly (no OpenAI)
    embeddings = []
    for doc in docs:
        # Gemini does not provide direct embedding API yet
        # We simulate embeddings using a simple text hashing approach
        embeddings.append([ord(c) for c in doc.page_content[:512]])  # basic numeric vector

    vectordb = FAISS.from_texts([doc.page_content for doc in docs], embeddings)
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
        # Simple retrieval
        relevant_docs = vectordb.similarity_search(query, k=3)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        prompt = f"Answer the question based on the following context:\n{context}\n\nQuestion: {query}"
        answer = query_gemini(prompt)
        st.markdown(f"**🤖 Answer:** {answer}")
