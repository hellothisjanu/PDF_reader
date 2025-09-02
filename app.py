import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

# ------------------------
# OpenAI API Key from Streamlit Secrets
# ------------------------
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    st.error("❌ OPENAI_API_KEY not found in Streamlit Secrets!")
    st.stop()

EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

# ------------------------
# Utility: Extract text from PDF
# ------------------------
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# ------------------------
# Split text into chunks
# ------------------------
def create_chunks(text, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = splitter.split_text(text)
    docs = [Document(page_content=chunk) for chunk in chunks]
    return docs

# ------------------------
# Build Vector DB using OpenAI embeddings
# ------------------------
def build_vectorstore(docs):
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=OPENAI_API_KEY)
    vectordb = Chroma.from_documents(docs, embedding=embeddings, persist_directory="chroma_db")
    vectordb.persist()
    return vectordb

# ------------------------
# Build Conversational QA Chain
# ------------------------
def build_qa_chain(vectordb):
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    llm = ChatOpenAI(model=LLM_MODEL, openai_api_key=OPENAI_API_KEY)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(page_title="📚 PDF Chatbot", layout="wide")
st.title("📚 PDF Document Chatbot")

# Upload PDFs
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF files", type=["pdf"], accept_multiple_files=True
)

# Session state
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Process PDFs
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
        st.error(f"⚠️ Error while querying: {e}")
