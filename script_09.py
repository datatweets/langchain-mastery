# rag_pdf_answer.py — RAG over a PDF using Chroma (new package) + OpenAI

import os
from dotenv import load_dotenv

# ---- Load env and disable LangSmith tracing BEFORE importing langchain bits
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "false"
for var in ["LANGSMITH_API_KEY", "LANGCHAIN_API_KEY", "LANGCHAIN_ENDPOINT", "LANGSMITH_ENDPOINT", "LANGCHAIN_PROJECT"]:
    os.environ.pop(var, None)

# ---- Imports (v0.2+ split packages)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma  # <- NEW: use langchain-chroma package
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# ----------- Config -----------
PDF_PATH = "pdfs/rag_vs_fine_tuning.pdf"
DB_DIR = "chroma_db"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ----------- Helpers -----------
def format_docs(docs, max_chars: int = 4000) -> str:
    """Join retrieved docs into a compact, cited context string."""
    parts, total = [], 0
    for d in docs:
        page = d.metadata.get("page", "?")
        text = d.page_content.strip().replace("\n", " ")
        snippet = f"[p.{page}] {text}"
        if total + len(snippet) > max_chars and parts:
            break
        parts.append(snippet)
        total += len(snippet)
    return "\n\n".join(parts)

def build_or_load_vectorstore():
    """Load PDF, split, and build/load a persisted Chroma vector store."""
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"PDF not found at {PDF_PATH}. Put the paper there or update PDF_PATH.")

    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-3-small")

    # If DB exists, reuse it (no re-embedding)
    if os.path.exists(DB_DIR):
        return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    # Fresh ingest
    loader = PyPDFLoader(PDF_PATH)
    data = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
    docs = splitter.split_documents(data)

    vs = Chroma.from_documents(
        docs,
        embedding_function=embeddings,  # <- NEW: use embedding_function=
        persist_directory=DB_DIR
    )
    return vs

def smart_retrieval_block(vectorstore):
    """
    Retrieval block that:
      1) Tries MMR (k=6, fetch_k=24) for diversity
      2) If context is short, retries with similarity (k=12) for coverage
    Returns a context string.
    """
    retriever_mmr = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 24, "lambda_mult": 0.2},
    )
    retriever_sim = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 12},
    )

    def _get_context(question: str) -> str:
        docs = retriever_mmr.invoke(question)
        ctx = format_docs(docs)
        if len(ctx) < 800:
            docs = retriever_sim.invoke(question)
            ctx = format_docs(docs)
        return ctx

    return RunnableLambda(_get_context)

def build_prompt():
    """
    Best-effort RAG prompt:
      - Prefer context with [p.X] citations
      - If some detail is missing, say "not in context" but still answer concisely
    """
    message = """
You are answering a question about the paper using the context below.
Prefer facts from the context and cite pages like [p.X].
If something important isn't in the context, say "not in context" briefly, but still provide the best possible answer.

Context:
{context}

Question:
{question}

Answer (2–6 bullet points max with page citations):
""".strip()
    return ChatPromptTemplate.from_messages([("human", message)])

def build_chain(vectorstore):
    llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY, temperature=0.1)
    context_block = smart_retrieval_block(vectorstore)
    prompt = build_prompt()

    chain = (
        {"context": context_block, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# ----------- Main -----------
def main():
    vectorstore = build_or_load_vectorstore()
    rag_chain = build_chain(vectorstore)

    # Pick your question:
    # question = "Which popular LLMs were considered in the paper?"
    # question = "List all LLMs mentioned anywhere in the paper."
    question = "Highlight the key contributions and practical takeaways from the paper."

    answer = rag_chain.invoke(question)
    print("\n--- Answer ---\n")
    print(answer)

if __name__ == "__main__":
    main()
