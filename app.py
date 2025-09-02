import os
import streamlit as st
from pypdf import PdfReader
from google import generativeai as genai

# ------------------------
# Setup Google Gemini API
# ------------------------
# Set your GEMINI_API_KEY in Streamlit secrets
# Go to "Settings -> Secrets" in Streamlit and add: GEMINI_API_KEY="YOUR_KEY"
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

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
# Streamlit UI
# ------------------------
st.set_page_config(page_title="📚 Gemini PDF Chatbot", layout="wide")
st.title("📚 Google Gemini PDF Chatbot")

uploaded_files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)

# Session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Step 1: Process uploaded PDFs
if uploaded_files:
    all_text = ""
    for pdf_file in uploaded_files:
        all_text += extract_text_from_pdf(pdf_file)
    
    st.session_state.pdf_text = all_text
    st.success("✅ PDF(s) processed successfully!")

# Step 2: Chat interface
query = st.text_input("Ask a question about your documents:")

if query and "pdf_text" in st.session_state:
    try:
        # Construct prompt with PDF content as context
        prompt = f"Use the following document content to answer the question.\n\nDocument Content:\n{st.session_state.pdf_text}\n\nQuestion: {query}\nAnswer:"

        # Generate response using Google Gemini
        response = genai.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        answer = response.text
        st.session_state.chat_history.append((query, answer))

        # Show answer
        st.markdown(f"**🤖 Answer:** {answer}")

        # Show chat history
        with st.expander("💬 Chat History"):
            for q, a in st.session_state.chat_history:
                st.markdown(f"**Q:** {q}")
                st.markdown(f"**A:** {a}\n")

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
