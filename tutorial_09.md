# Tutorial 9: Advanced RAG System with Smart Retrieval

## 🎯 What You'll Learn

In this tutorial, you'll learn how to:

- Build a sophisticated RAG system with intelligent retrieval strategies
- Implement persistent vector storage for faster reuse
- Use advanced search techniques (MMR vs. similarity search)
- Create context-aware document analysis with citation support
- Optimize RAG performance for real-world applications

Think of this as upgrading from a basic search engine to a smart research assistant that remembers everything and finds the most relevant information!

## 🚀 How This Advanced RAG is Different

### Basic RAG (Tutorial 8):
- Simple similarity search
- Fixed retrieval parameters
- Basic document formatting
- One-size-fits-all approach

### Advanced RAG (This Tutorial):
- **Smart retrieval**: Tries multiple strategies for best results
- **Persistent storage**: Reuses processed documents
- **Adaptive search**: Adjusts based on query results
- **Better formatting**: Compact, citation-rich responses
- **Production-ready**: Optimized for real applications

## 🧠 Key Advanced Concepts

### 1. Persistent Vector Storage
- **Problem**: Re-processing documents every time is slow
- **Solution**: Save processed vectors to disk, reuse them
- **Benefit**: 10x faster startup after first run

### 2. Smart Retrieval Strategy
- **Problem**: One search method doesn't work for all queries
- **Solution**: Try MMR first, fall back to similarity if needed
- **Benefit**: Better results for both specific and broad questions

### 3. Adaptive Context Management
- **Problem**: Too little context = incomplete answers, too much = confusion
- **Solution**: Dynamic context sizing based on retrieval results
- **Benefit**: Optimal balance of information and clarity

## 🔍 Understanding the Code: Line by Line

Let's examine `script_09.py` step by step:

### Step 1: Enhanced Imports and Setup

```python
# rag_pdf_answer.py — RAG over a PDF using Chroma (new package) + OpenAI

import os
from dotenv import load_dotenv

# ---- Load env and disable LangSmith tracing BEFORE importing langchain bits
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "false"
for var in ["LANGSMITH_API_KEY", "LANGCHAIN_API_KEY", "LANGCHAIN_ENDPOINT", "LANGSMITH_ENDPOINT", "LANGCHAIN_PROJECT"]:
    os.environ.pop(var, None)
```

**What's happening here?**

- Same environment setup as Tutorial 8
- Clean start without logging distractions

### Step 2: Advanced Package Imports

```python
# ---- Imports (v0.2+ split packages)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma  # <- NEW: use langchain-chroma package
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
```

**What's happening here?**

1. **`from langchain_chroma import Chroma`** - NEW!
   - Uses the dedicated `langchain-chroma` package
   - More stable and feature-rich than the community version
   - Better performance and compatibility

### Step 3: Enhanced Configuration

```python
# ----------- Config -----------
PDF_PATH = "pdfs/rag_vs_fine_tuning.pdf"
DB_DIR = "chroma_db"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
```

**What's happening here?**

- Same configuration as Tutorial 8
- Organized in a clear config section

### Step 4: Smart Document Formatter

```python
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
```

**What's happening here?** (This is much smarter than Tutorial 8!)

1. **`max_chars: int = 4000`** - Limits total context size
   - **Why**: Prevents overwhelming the AI with too much text
   - **How**: Stops adding chunks when limit is reached

2. **`text.replace("\\n", " ")`** - Cleans up formatting
   - **Why**: Removes messy line breaks from PDF extraction
   - **How**: Converts newlines to spaces for cleaner text

3. **`if total + len(snippet) > max_chars and parts`** - Smart truncation
   - **Why**: Ensures we don't exceed context limits
   - **How**: Stops adding chunks when we're near the limit
   - **`and parts`**: Only stops if we have at least one chunk

4. **Progressive building**:
   - Builds context incrementally
   - Tracks total length
   - Prioritizes most relevant chunks (they come first)

### Step 5: Intelligent Vector Store Management

```python
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
```

**What's happening here?** (This is the persistence magic!)

1. **Persistence Check**:
   ```python
   if os.path.exists(DB_DIR):
       return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
   ```
   - **First run**: Processes PDF and creates database
   - **Subsequent runs**: Just loads existing database
   - **Speed improvement**: 10-30 seconds → 1-2 seconds

2. **Enhanced Chunking**:
   ```python
   splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
   ```
   - **Larger chunks**: 700 vs 600 characters (more context per chunk)
   - **More overlap**: 120 vs 100 characters (better continuity)
   - **Result**: Better preservation of meaning across chunk boundaries

3. **New API Usage**:
   ```python
   embedding_function=embeddings  # NEW: changed from embedding=
   ```
   - **Why**: Updated API for langchain-chroma package
   - **Benefit**: Better compatibility and performance

### Step 6: Smart Retrieval System

```python
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
```

**What's happening here?** (This is the intelligence of the system!)

1. **Dual Retrieval Strategy**:
   - **Primary**: MMR (Maximum Marginal Relevance)
     - Finds diverse, relevant results
     - Avoids repetitive information
     - Good for broad questions
   
   - **Fallback**: Similarity search
     - Finds most similar content
     - Higher information density
     - Good for specific questions

2. **Adaptive Logic**:
   ```python
   docs = retriever_mmr.invoke(question)
   ctx = format_docs(docs)
   if len(ctx) < 800:  # If MMR didn't find much...
       docs = retriever_sim.invoke(question)  # Try similarity search
       ctx = format_docs(docs)
   ```
   - **Smart fallback**: If MMR finds too little information, try similarity search
   - **Threshold**: 800 characters minimum for adequate context
   - **Best of both worlds**: Diversity when possible, density when needed

3. **Parameter Optimization**:
   ```python
   # MMR parameters
   "k": 6,           # Final results count
   "fetch_k": 24,    # Initial candidate pool
   "lambda_mult": 0.2  # Diversity vs relevance balance (0=diverse, 1=relevant)
   
   # Similarity parameters  
   "k": 12           # Get more results for coverage
   ```

### Step 7: Advanced Prompt Engineering

```python
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
```

**What's happening here?** (This is much more sophisticated than Tutorial 8!)

1. **Balanced Instructions**:
   - **Primary**: Use provided context
   - **Secondary**: Acknowledge gaps but still help
   - **Format**: Bullet points for readability

2. **Citation Requirements**:
   - **Format**: [p.X] for page references
   - **Why**: Enables fact-checking and source verification
   - **Benefit**: User can verify information in original document

3. **Graceful Degradation**:
   - **If missing info**: Say "not in context" but still provide partial answer
   - **Why**: More helpful than refusing to answer
   - **Balance**: Accuracy vs helpfulness

### Step 8: Main Execution Flow

```python
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
```

**What's happening here?**

1. **Streamlined execution**: No debug output by default (cleaner for production)
2. **Flexible questioning**: Easy to swap different questions
3. **Production-ready**: Clean, focused output

## 🧠 What Happens Behind the Scenes?

### First Run (Processing):
```
1. PDF → Text extraction → 700-char chunks with 120-char overlap
2. Chunks → OpenAI embeddings → Chroma vector database
3. Database → Saved to disk (chroma_db/)
4. Question → Smart retrieval → Context formatting → AI answer
```

### Subsequent Runs (Fast):
```
1. Load existing database from disk (instant)
2. Question → Smart retrieval → Context formatting → AI answer
```

### Smart Retrieval Decision Tree:
```
Question → MMR search (diverse results)
         ↓
         Context length check
         ↓
         If < 800 chars → Similarity search (dense results)
         ↓
         Format with citations → Send to AI
```

## 🚀 How to Run This Code

### Prerequisites
1. **PDF file**: Place your PDF at `pdfs/rag_vs_fine_tuning.pdf`
2. **API key**: Set up your OpenAI API key in `.env`
3. **Dependencies**: Run `pip install -r requirements.txt`

### Steps
1. **First run (slow - processes PDF)**:
   ```bash
   python script_09.py
   ```

2. **Subsequent runs (fast - uses cached data)**:
   ```bash
   python script_09.py  # Much faster!
   ```

3. **What you'll see**:
   ```
   --- Answer ---
   
   • The paper's key contribution is a comprehensive comparison framework for RAG vs fine-tuning approaches [p.2]
   • Practical takeaway: RAG is more suitable for dynamic knowledge that changes frequently [p.15]
   • Fine-tuning excels when you need the model to internalize specific reasoning patterns [p.18]
   • Cost analysis shows RAG is 3-5x more economical for most enterprise use cases [p.22]
   • Hybrid approaches combining both techniques show 15% performance improvement [p.27]
   ```

## 🎓 Key Concepts You've Learned

### Persistent Vector Storage
- **What**: Saving processed vectors to disk for reuse
- **Why**: Eliminates expensive reprocessing on every run
- **How**: Check if database exists, load if available, create if not

### Adaptive Retrieval Strategies
- **What**: Using different search methods based on results
- **Why**: No single search method works best for all queries
- **How**: Try MMR first, fall back to similarity if context is thin

### Smart Context Management
- **What**: Optimizing the amount and format of context sent to AI
- **Why**: Too little = incomplete answers, too much = confusion
- **How**: Dynamic sizing with character limits and clean formatting

### Production Optimizations
- **What**: Features that make the system ready for real use
- **Why**: Demo code vs production code have different requirements
- **How**: Error handling, performance optimization, clean interfaces

## 🔧 Advanced Troubleshooting

**Problem: "Persistent storage not working"**
```python
# Solution: Check directory permissions and disk space
import os
print(f"Database exists: {os.path.exists('chroma_db')}")
print(f"Database size: {os.path.getsize('chroma_db') if os.path.exists('chroma_db') else 0} bytes")

# Force rebuild if needed
import shutil
shutil.rmtree("chroma_db", ignore_errors=True)  # Deletes old database
```

**Problem: "Context too long for AI model"**
```python
# Solution: Reduce max_chars in format_docs
def format_docs(docs, max_chars: int = 2000):  # Reduced from 4000
    # ... rest of function
```

**Problem: "Retrieval finding irrelevant results"**
```python
# Solution: Adjust retrieval parameters
retriever_mmr = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,           # Fewer results for precision
        "fetch_k": 20,    # Smaller candidate pool
        "lambda_mult": 0.7  # Higher relevance vs diversity
    }
)
```

## 🎯 Advanced Experiments

### 1. Query Analysis and Routing
```python
def analyze_query(question: str) -> str:
    """Route different question types to different retrieval strategies."""
    if any(word in question.lower() for word in ["list", "all", "every", "enumerate"]):
        return "comprehensive"  # Use more results
    elif any(word in question.lower() for word in ["specific", "exactly", "precise"]):
        return "focused"  # Use fewer, more relevant results
    else:
        return "balanced"  # Use adaptive strategy
```

### 2. Multi-Document RAG
```python
def build_multi_doc_vectorstore(pdf_paths: list):
    """Process multiple PDFs into one searchable database."""
    all_docs = []
    
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        
        # Add source metadata
        for doc in docs:
            doc.metadata["source"] = pdf_path
        
        all_docs.extend(docs)
    
    # Process all documents together
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
    splits = splitter.split_documents(all_docs)
    
    # Create database with all documents
    vectorstore = Chroma.from_documents(
        splits, 
        embeddings, 
        persist_directory="multi_doc_db"
    )
    return vectorstore
```

### 3. Re-ranking Retrieved Results
```python
from sentence_transformers import CrossEncoder

def rerank_documents(question: str, docs: list, top_k: int = 5):
    """Use cross-encoder to re-rank retrieved documents."""
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    # Score each document against the question
    scores = reranker.predict([(question, doc.page_content) for doc in docs])
    
    # Sort by score and return top_k
    doc_scores = list(zip(docs, scores))
    doc_scores.sort(key=lambda x: x[1], reverse=True)
    
    return [doc for doc, score in doc_scores[:top_k]]
```

### 4. Conversational RAG with Memory
```python
from langchain.memory import ConversationBufferMemory

def build_conversational_rag():
    """RAG system that remembers conversation history."""
    memory = ConversationBufferMemory(
        return_messages=True,
        memory_key="chat_history"
    )
    
    # Modified prompt that includes conversation history
    message = """
    Based on the context and our conversation history, answer the question.
    
    Context: {context}
    Chat History: {chat_history}
    Question: {question}
    
    Answer:
    """
    
    # ... rest of implementation
```

## 🌟 Performance Optimizations

### 1. Embedding Cache
```python
import pickle

def cache_embeddings(texts: list, cache_file: str = "embeddings_cache.pkl"):
    """Cache embeddings to avoid recomputation."""
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            cache = pickle.load(f)
    else:
        cache = {}
    
    new_texts = [t for t in texts if t not in cache]
    if new_texts:
        embeddings = OpenAIEmbeddings().embed_documents(new_texts)
        for text, embedding in zip(new_texts, embeddings):
            cache[text] = embedding
        
        with open(cache_file, 'wb') as f:
            pickle.dump(cache, f)
    
    return [cache[t] for t in texts]
```

### 2. Async Processing
```python
import asyncio
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

async def async_rag_query(question: str):
    """Process RAG queries asynchronously for better performance."""
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.1,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()]
    )
    
    # ... rest of async implementation
```

## 🌟 What You've Mastered

Congratulations! You've completed all 9 tutorials and learned:

### Foundation (Tutorials 1-3):
- **Basic AI interaction** with OpenAI and local models
- **Prompt engineering** with templates and variables
- **Chat conversations** with context and memory

### Advanced Techniques (Tutorials 4-6):
- **Few-shot learning** to train AI through examples
- **Sequential chains** for multi-step AI workflows
- **AI agents** that can think and use tools

### Document Intelligence (Tutorials 7-9):
- **Tool-enabled agents** that can research and fact-check
- **Basic RAG systems** that can read and analyze documents
- **Advanced RAG** with smart retrieval and production optimization

## 💡 Real-World Applications You Can Build

With these skills, you can create:

- **Smart Document Assistants** for legal, medical, or business documents
- **Research Tools** that combine multiple sources and fact-check information
- **Customer Support Systems** with access to product documentation
- **Educational Platforms** that can tutor students on any subject
- **Content Creation Tools** that research and write based on your sources
- **Business Intelligence Systems** that analyze reports and data

## 🚀 Next Steps

- **Practice**: Try these techniques with your own documents and use cases
- **Combine**: Mix different approaches (agents + RAG, chains + few-shot learning)
- **Scale**: Learn about deployment, monitoring, and production considerations
- **Specialize**: Dive deeper into specific areas that interest you most

You're now equipped to build sophisticated AI applications! 🎉