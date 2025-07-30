from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings
import re

def remove_surrogates(text):
    return re.sub(r'[\uD800-\uDFFF]', '', text)

def load_and_index_pdf(file_path, api_key):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    for doc in documents:
        doc.page_content = remove_surrogates(doc.page_content)
    try:
        embeddings = MistralAIEmbeddings(model="mistral-embed", api_key=api_key)
    except Exception as e:
        raise RuntimeError("Failed to initialize MistralAIEmbeddings. "
                           "Check your model name or API key.") from e
    vectorstore = FAISS.from_documents(documents, embeddings)

    return vectorstore
