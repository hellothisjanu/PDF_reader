import streamlit as st
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import ConversationalRetrievalChain
from ollama import ChatCompletion

# Initialize Ollama model
llm = ChatCompletion(model="llama2-uncensored")

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

# Function to split text into chunks
def create_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    return [Document(page_content=chunk) for chunk in chunks]

# Function to build vector store
def build_vectorstore(docs):
    embeddings = OllamaEmbeddings(model="llama2-uncensored")
    vectordb = Chroma.from_documents(docs, embedding=embeddings, persist_directory="chroma_db")
    vectordb.persist()
    return vectordb

# Function to build QA chain
def build_qa_chain(vectordb):
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

# Streamlit UI
st.title("📚 PDF Q&A Chatbot")
uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    all_text = ""
    for uploaded_file in uploaded_files:
        all_text += extract_text_from_pdf(uploaded_file)
    docs = create_chunks(all_text)
    vectordb = build_vectorstore(docs)
    qa_chain = build_qa_chain(vectordb)
    st.session_state.qa_chain = qa_chain
    st.success("✅ Documents processed and ready!")

query = st.text_input("Ask a question:")
if query and "qa_chain" in st.session_state:
    result = st.session_state.qa_chain({"question": query, "chat_history": []})
    st.markdown(f"**🤖 Answer:** {result['answer']}")
    if result.get("source_documents"):
        with st.expander("📄 Sources"):
            for doc in result["source_documents"]:
                st.markdown(f"- {doc.page_content[:200]}...")
