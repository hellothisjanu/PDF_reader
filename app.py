import os
import tempfile
import shutil
import requests
import PyPDF2
import streamlit as st

# Updated imports for new LangChain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain



# ---------------- CONFIG ----------------
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
CHROMA_PERSIST_DIR = os.path.join(tempfile.gettempdir(), "chroma_store")
TOP_K = 4


# ---------------- HELPERS ----------------
def extract_text_from_pdf(uploaded_file):
    """Extract text from each page of PDF."""
    reader = PyPDF2.PdfReader(uploaded_file)
    docs = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            docs.append(Document(page_content=text, metadata={
                "source": uploaded_file.name,
                "page": i + 1
            }))
    return docs


def chunk_documents(docs):
    """Split text into chunks with overlap."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    split_docs = []
    for d in docs:
        for i, chunk in enumerate(splitter.split_text(d.page_content)):
            meta = dict(d.metadata)
            meta["chunk"] = i + 1
            split_docs.append(Document(page_content=chunk, metadata=meta))
    return split_docs


def get_vectorstore(docs):
    """Build Chroma vector DB from documents."""
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=os.environ["OPENAI_API_KEY"])
    vectordb = Chroma.from_documents(docs, embedding_function=embeddings, persist_directory=CHROMA_PERSIST_DIR)
    vectordb.persist()
    return vectordb


def serper_search(query):
    """Use Serper.dev for web search if needed."""
    api_key = os.environ.get("SERPER_API_KEY")
    if not api_key:
        return []
    url = "https://api.serper.dev/search"
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    payload = {"q": query, "num": 3}
    resp = requests.post(url, headers=headers, json=payload)
    data = resp.json()
    results = []
    for item in data.get("organic", [])[:3]:
        results.append(f"{item.get('title')}: {item.get('snippet')} ({item.get('link')})")
    return results


# ---------------- STREAMLIT APP ----------------
st.set_page_config(page_title="Universal Document Intelligence", layout="wide")
st.title("🧠 Universal Document Intelligence Chatbot")

with st.sidebar:
    st.header("Upload PDFs & Settings")
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    if st.button("Clear Vector DB"):
        if os.path.exists(CHROMA_PERSIST_DIR):
            shutil.rmtree(CHROMA_PERSIST_DIR)
            st.success("Cleared stored embeddings.")

    openai_key = st.text_input("OpenAI API Key", type="password")
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key

    serp_key = st.text_input("Serper.dev API Key (optional)", type="password")
    if serp_key:
        os.environ["SERPER_API_KEY"] = serp_key


# Build vector DB
if st.button("Ingest Documents"):
    if not uploaded_files:
        st.warning("Please upload at least one PDF.")
    else:
        all_docs = []
        for f in uploaded_files:
            all_docs.extend(extract_text_from_pdf(f))
        chunks = chunk_documents(all_docs)
        vectordb = get_vectorstore(chunks)
        st.success(f"Ingested {len(chunks)} chunks.")


# Chat Interface
query = st.text_input("Ask a question about your documents")
if st.button("Get Answer"):
    if not query:
        st.warning("Enter a question first.")
    elif not os.path.exists(CHROMA_PERSIST_DIR):
        st.error("Please upload and ingest documents first.")
    else:
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=os.environ["OPENAI_API_KEY"])
        vectordb = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)
        retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K})

        chat_llm = ChatOpenAI(model_name=LLM_MODEL, temperature=0, openai_api_key=os.environ["OPENAI_API_KEY"])
        qa_chain = ConversationalRetrievalChain.from_llm(chat_llm, retriever)

        result = qa_chain({"question": query, "chat_history": []})
        answer = result["answer"]

        st.subheader("Answer")
        st.write(answer)

        # Fallback: web search
        if any(word in query.lower() for word in ["latest", "2024", "current", "vs", "price", "trend"]):
            st.subheader("Web Results")
            web_res = serper_search(query)
            if web_res:
                for r in web_res:
                    st.write("- " + r)
            else:
                st.info("No web results (Serper API key missing).")
