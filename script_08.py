# rag_pdf_chat_fix.py
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
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

PDF_PATH = "pdfs/rag_vs_fine_tuning.pdf"
DB_DIR = "chroma_db"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def format_docs(docs):
    # helpful for the model + for your own debugging
    chunks = []
    for d in docs:
        page = d.metadata.get("page", "?")
        chunks.append(f"[page {page}] {d.page_content.strip()}")
    return "\n\n".join(chunks)

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

def build_chain(vectorstore):
    # Use MMR to diversify + fetch more, then return top-k
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 20, "fetch_k": 20, "lambda_mult": 0.2}
    )

    message = """
Answer the following question using ONLY the context provided. If the answer is not in the context, say "I don't know."

Context:
{context}

Question:
{question}

Answer:
""".strip()

    prompt = ChatPromptTemplate.from_messages([("human", message)])

    llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY, temperature=0.1)

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain, retriever

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

    rag_chain, retriever = build_chain(vectorstore)

    # question = "Which popular LLMs were considered in the paper?"
    # question = "List all LLMs mentioned anywhere in the paper."
    
    debug_retrieval(retriever, question)  # see what we actually retrieved

    answer = rag_chain.invoke(question)
    print("\n--- Answer ---\n")
    print(answer)

if __name__ == "__main__":
    main()
