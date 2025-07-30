from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

def get_qa_chain(vectorstore, groq_api_key):
    if vectorstore is None:
        return None

    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama3-70b-8192",
        temperature=0.2
    )
    retriever = vectorstore.as_retriever(search_type="similarity", k=3)
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
