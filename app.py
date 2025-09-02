import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub

# ------------------------
# Load PDF
# ------------------------
def load_pdf(pdf_file):
    loader = PyPDFLoader(pdf_file)
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(pages)
    return docs

# ------------------------
# Build Vector Store
# ------------------------
def build_vectorstore(docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(docs, embedding=embeddings, persist_directory="chroma_db")
    vectordb.persist()
    return vectordb

# ------------------------
# Build QA Chain
# ------------------------
def build_qa_chain(vectordb):
    retriever = vectordb.as_retriever()
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-small",  # free lightweight model
        model_kwargs={"temperature": 0.1, "max_length": 512}
    )
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain

# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(page_title="PDF Q&A App", page_icon="📄", layout="wide")

st.title("📄 Ask Questions from your PDF (Free HuggingFace Version)")

pdf_file = st.file_uploader("Upload your PDF", type=["pdf"])

if pdf_file:
    # Save uploaded file temporarily
    with open("uploaded.pdf", "wb") as f:
        f.write(pdf_file.read())

    st.success("✅ PDF uploaded successfully!")

    if "qa_chain" not in st.session_state:
        docs = load_pdf("uploaded.pdf")
        vectordb = build_vectorstore(docs)
        st.session_state.qa_chain = build_qa_chain(vectordb)

    query = st.text_input("Ask a question about the PDF:")

    if query:
        with st.spinner("Thinking..."):
            result = st.session_state.qa_chain.run(query)
        st.write("### Answer:")
        st.write(result)
