# rag_retrieval_qa.py
import os
from dotenv import load_dotenv

# Disable LangSmith warnings before imports (optional)
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "false"
for var in ["LANGSMITH_API_KEY","LANGCHAIN_API_KEY","LANGCHAIN_ENDPOINT","LANGSMITH_ENDPOINT","LANGCHAIN_PROJECT"]:
    os.environ.pop(var, None)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

PDF_PATH = "pdfs/rag_vs_fine_tuning.pdf"
DB_DIR = "chroma_db"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def build_vectorstore():
    loader = PyPDFLoader(PDF_PATH)
    data = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,      # a bit larger to keep ideas together
        chunk_overlap=100
    )
    docs = splitter.split_documents(data)

    embeddings = OpenAIEmbeddings(
        api_key=OPENAI_API_KEY,
        model="text-embedding-3-small"
    )

    vs = Chroma.from_documents(
        docs,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    return vs, len(docs)

def build_retrieval_qa_chain(vectorstore):
    # Use MMR to diversify + fetch more, then return top-k
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 20, "fetch_k": 20, "lambda_mult": 0.2}
    )

    llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY, temperature=0.1)

    # Build RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    return qa_chain, retriever

def debug_retrieval(retriever, question):
    docs = retriever.invoke(question)
    print(f"\nRetrieved {len(docs)} docs. First snippet:\n")
    for i, d in enumerate(docs[:3], 1):
        page = d.metadata.get("page", "?")
        print(f"--- Doc {i} (page {page}) ---\n{d.page_content[:600]}\n")

def main():
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"Place {PDF_PATH} next to this script.")
    
    vectorstore, n = build_vectorstore()
    print(f"Indexed {n} chunks into Chroma.")

    qa_chain, retriever = build_retrieval_qa_chain(vectorstore)

    question = "Which popular LLMs were considered in the paper?"
    # question = "List all LLMs mentioned anywhere in the paper."
    
    debug_retrieval(retriever, question)  # see what we actually retrieved

    result = qa_chain.invoke({"query": question})
    
    print("\n--- Answer ---\n")
    print(result["result"])
    
    print(f"\n--- Source Documents ({len(result['source_documents'])}) ---")
    for i, doc in enumerate(result["source_documents"][:3], 1):
        page = doc.metadata.get("page", "?")
        print(f"\nSource {i} (page {page}):\n{doc.page_content[:200]}...")

if __name__ == "__main__":
    main()