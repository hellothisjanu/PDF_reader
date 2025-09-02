import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatGoogleGemini
from langchain.embeddings import GoogleGeminiEmbeddings

# ------------------------
# Google API Key from Streamlit secrets
# ------------------------
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "")
if not GOOGLE_API_KEY:
    st.error("❌ GOOGLE_API_KEY not found in Streamlit Secrets! Add it in the app secrets.")
    st.stop()

# ------------------------
# PDF Text Extraction
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
def create_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = splitter.split_text(text)
    return [Document(page_content=chunk) for chunk in chunks]

# ------------------------
# Build FAISS Vector Store
# ------------------------
def build_vectorstore(docs):
    embeddings = GoogleGeminiEmbeddings(api_key=GOOGLE_API_KEY)
    vectordb = FAISS.from_documents(docs, embeddings)
    return vectordb

# ------------------------
# Build QA Chain
# ------------------------
def build_qa_chain(vectordb):
    llm = ChatGoogleGemini(api_key=GOOGLE_API_KEY, model="gemini-1.5-t")
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(page_title="📚 Gemini PDF Chatbot", layout="wide")
st.title("📚 Google Gemini Document Chatbot")

# Sidebar for PDF upload
st.sidebar.header("Upload PDF Files")
uploaded_files = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

# Session state for persistence
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Process uploaded PDFs
if uploaded_files:
    all_text = ""
    for uploaded_file in uploaded_files:
        all_text += extract_text_from_pdf(uploaded_file)

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
                    st.markdown(f"- {doc.page_content[:200]}...")  # preview of chunk

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
