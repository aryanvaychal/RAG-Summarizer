import os
from dotenv import load_dotenv
import streamlit as st
from core.retriever import load_and_index_pdf
from core.qa_engine import get_qa_chain

# --- Load environment variables from .env ---
load_dotenv()

API_KEY = os.getenv("MISTRALAI_API_KEY")
GROQ_KEY = os.getenv("GROQ_KEY")
DOCUMENTS_DIR = "documents"
os.makedirs(DOCUMENTS_DIR, exist_ok=True)

# --- STATE ---
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# --- Utility: Unicode Cleaner ---
def clean_output(s):
    """Remove invalid surrogate pairs."""
    return ''.join(c for c in s if not '\uD800' <= c <= '\uDFFF')

# --- UI: Upload PDF ---
st.title("📄 Research Paper Summarizer & QA")

uploaded_file = st.file_uploader("Upload your research paper (PDF)", type="pdf")

if uploaded_file is not None:
    file_path = os.path.join(DOCUMENTS_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    st.success("✅ File uploaded and saved!")

    with st.spinner("🔍 Processing and indexing..."):
        vectorstore = load_and_index_pdf(file_path, API_KEY)
        st.session_state.vectorstore = vectorstore
        st.session_state.indexed_file = uploaded_file.name
    st.success("📚 Document indexed! You can now ask questions.")

# --- UI: QA Interface ---
if st.session_state.vectorstore is not None:
    qa_chain = get_qa_chain(st.session_state.vectorstore, GROQ_KEY)

    query = st.text_input("💬 Ask a question about the paper:")

    if query:
        response = qa_chain(query)
        clean_result = clean_output(response["result"])

        st.markdown("📌 **Answer:**")
        st.success(clean_result)

        # Optional: Show sources
        with st.expander("📄 Show source chunks"):
            for doc in response["source_documents"]:
                st.markdown(clean_output(doc.page_content))
else:
    st.info("📁 Please upload and process a PDF before asking questions.")
