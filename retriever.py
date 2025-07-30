from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings
import re

def remove_surrogates(text):
    """Remove Unicode surrogate pairs that can't be encoded in UTF-8."""
    return re.sub(r'[\uD800-\uDFFF]', '', text)

def load_and_index_pdf(file_path, api_key):
    # Step 1: Load PDF pages
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Step 2: Clean text content
    for doc in documents:
        doc.page_content = remove_surrogates(doc.page_content)

    # Step 3: Embed documents using Mistral
    try:
        embeddings = MistralAIEmbeddings(model="mistral-embed", api_key=api_key)
    except Exception as e:
        raise RuntimeError("Failed to initialize MistralAIEmbeddings. "
                           "Check your model name or API key.") from e

    # Step 4: Vector index using FAISS
    vectorstore = FAISS.from_documents(documents, embeddings)

    return vectorstore
