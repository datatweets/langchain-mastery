# Tutorial 8: Building Your First RAG System (AI that Reads Documents)

## 🎯 What You'll Learn

In this tutorial, you'll learn how to:

- Create a RAG (Retrieval-Augmented Generation) system
- Teach AI to read and understand PDF documents
- Build a vector database to store document knowledge
- Ask questions about your own documents and get accurate answers

Think of this as giving AI a photographic memory for your documents - it can instantly find and reference any information!

## 🤔 What is RAG?

**RAG = Retrieval-Augmented Generation**

### Without RAG (Regular AI):
```
You: "What does this research paper say about LLMs?"
AI: "I don't have access to your specific paper. Based on my general knowledge..."
(Can't read your documents!)
```

### With RAG (AI with Document Memory):
```
You: "What does this research paper say about LLMs?"
AI: *searches through your PDF*
AI: "According to page 3 of your paper, it states that 'LLMs show significant improvement in reasoning tasks when...' [continues with exact quotes]"
(Reads your actual documents!)
```

**The Magic**: RAG combines the AI's language abilities with your specific documents!

## 🏗️ How RAG Works (Step by Step)

### Phase 1: Document Processing (Setup)
1. **Load PDF** → Read your document
2. **Split into chunks** → Break into smaller pieces
3. **Create embeddings** → Convert text to numbers (vectors)
4. **Store in database** → Save for fast searching

### Phase 2: Question Answering (Runtime)
1. **User asks question** → You ask about the document
2. **Find relevant chunks** → Search database for related content
3. **Send to AI** → Give AI the relevant pieces + your question
4. **Generate answer** → AI responds using your document content

## 🔍 Understanding the Code: Line by Line

Let's examine `script_08.py` step by step:

### Step 1: Imports and Setup

```python
# script_08.py
import os
from dotenv import load_dotenv

# Disable LangSmith warnings before imports (optional)
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "false"
for var in ["LANGSMITH_API_KEY","LANGCHAIN_API_KEY","LANGCHAIN_ENDPOINT","LANGSMITH_ENDPOINT","LANGCHAIN_PROJECT"]:
    os.environ.pop(var, None)
```

**What's happening here?**

1. **Environment cleanup** - Disables various logging services
2. **Why do this?** - Prevents warning messages from cluttering output
3. **Optional** - You can remove this if you don't mind the warnings

### Step 2: RAG-Specific Imports

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
```

**What's happening here?**

1. **`PyPDFLoader`** - Reads PDF files
2. **`RecursiveCharacterTextSplitter`** - Breaks documents into chunks
3. **`Chroma`** - Vector database for storing document pieces
4. **`OpenAIEmbeddings`** - Converts text to vector numbers
5. **`RunnablePassthrough, RunnableLambda`** - Chain components for RAG workflow

### Step 3: Configuration

```python
PDF_PATH = "pdfs/rag_vs_fine_tuning.pdf"
DB_DIR = "chroma_db"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
```

**What's happening here?**

1. **`PDF_PATH`** - Location of your PDF file
2. **`DB_DIR`** - Where to store the vector database
3. **`OPENAI_API_KEY`** - Your OpenAI API key for embeddings and chat

### Step 4: Document Formatting Helper

```python
def format_docs(docs):
    # helpful for the model + for your own debugging
    chunks = []
    for d in docs:
        page = d.metadata.get("page", "?")
        chunks.append(f"[page {page}] {d.page_content.strip()}")
    return "\n\n".join(chunks)
```

**What's happening here?**

1. **Purpose**: Formats retrieved document chunks for the AI
2. **`d.metadata.get("page", "?")`** - Gets page number (or "?" if unknown)
3. **`f"[page {page}] {content}"`** - Adds page citations to each chunk
4. **Why important**: AI can cite specific pages in its answers

### Step 5: Building the Vector Database

```python
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
```

**What's happening here?** (This is the core of RAG!)

1. **`PyPDFLoader(PDF_PATH)`** - Opens and reads your PDF
2. **`loader.load()`** - Extracts all text from the PDF

3. **Text Splitting**:
   ```python
   RecursiveCharacterTextSplitter(
       chunk_size=600,      # Each chunk ~600 characters
       chunk_overlap=100    # 100 characters overlap between chunks
   )
   ```
   - **Why split?** AI has limited context, can't read entire documents
   - **Chunk size**: Balance between detail and efficiency
   - **Overlap**: Ensures no information is lost between chunks

4. **Embeddings Creation**:
   ```python
   OpenAIEmbeddings(model="text-embedding-3-small")
   ```
   - **What are embeddings?** Numbers that represent the meaning of text
   - **Why needed?** Computers can't search text directly, but can search numbers
   - **Model choice**: "text-embedding-3-small" is fast and efficient

5. **Vector Database Storage**:
   ```python
   Chroma.from_documents(docs, embedding=embeddings, persist_directory=DB_DIR)
   ```
   - **Chroma**: A database optimized for vector search
   - **persist_directory**: Saves database to disk for reuse
   - **Result**: Fast searchable database of your document

### Step 6: Building the RAG Chain

```python
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
```

**What's happening here?** (This is the RAG workflow!)

1. **Retriever Setup**:
   ```python
   retriever = vectorstore.as_retriever(
       search_type="mmr",  # Maximum Marginal Relevance
       search_kwargs={"k": 20, "fetch_k": 20, "lambda_mult": 0.2}
   )
   ```
   - **MMR**: Smart search that finds diverse, relevant results
   - **k=20**: Return top 20 most relevant chunks
   - **lambda_mult=0.2**: Balance between relevance and diversity

2. **RAG Prompt Template**:
   ```python
   message = """Answer using ONLY the context provided..."""
   ```
   - **Critical instruction**: AI must use only the provided context
   - **Prevents hallucination**: AI won't make up information
   - **Format**: Context + Question → Answer

3. **RAG Chain Construction**:
   ```python
   rag_chain = (
       {
           "context": retriever | RunnableLambda(format_docs),
           "question": RunnablePassthrough(),
       }
       | prompt
       | llm
       | StrOutputParser()
   )
   ```
   
   **Step-by-step breakdown**:
   - `retriever`: Finds relevant document chunks
   - `format_docs`: Formats chunks with page numbers
   - `RunnablePassthrough()`: Passes the question through unchanged
   - `prompt`: Combines context + question into final prompt
   - `llm`: AI generates answer
   - `StrOutputParser()`: Cleans up the response

### Step 7: Debugging Function

```python
def debug_retrieval(retriever, question):
    docs = retriever.invoke(question)
    print(f"\nRetrieved {len(docs)} docs. First snippet:\n")
    for i, d in enumerate(docs[:3], 1):
        page = d.metadata.get("page", "?")
        print(f"--- Doc {i} (page {page}) ---\n{d.page_content[:600]}\n")
```

**What's happening here?**

- **Purpose**: Shows what documents the RAG system found
- **Helpful for**: Understanding why the AI gave a certain answer
- **Shows**: Top 3 relevant document chunks with page numbers

### Step 8: Main Function (Putting It All Together)

```python
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
```

**What's happening here?**

1. **File Check**: Ensures PDF exists before starting
2. **Build Database**: Creates vector database from your PDF
3. **Build Chain**: Creates the RAG question-answering system
4. **Debug**: Shows what document chunks were found
5. **Answer**: Gets AI's response based on document content

## 🧠 What Happens Behind the Scenes?

### When you run the script:

1. **PDF Processing**:
   ```
   PDF → Text extraction → Split into 600-character chunks → 
   Create embeddings → Store in Chroma database
   ```

2. **Question Processing**:
   ```
   Question → Find relevant chunks → Format with page numbers → 
   Create prompt → Send to AI → Get answer with citations
   ```

3. **AI Response**:
   ```
   AI sees: "Based on this context from pages 2-5 of the document: [chunks]
   Question: What does the paper say about LLMs?
   Answer using only this context:"
   ```

## 🚀 How to Run This Code

### Prerequisites
1. **PDF file**: Place your PDF at `pdfs/rag_vs_fine_tuning.pdf`
2. **API key**: Set up your OpenAI API key in `.env`

### Steps
1. **Create the PDF directory**:
   ```bash
   mkdir -p pdfs
   # Place your PDF file in the pdfs/ directory
   ```

2. **Run the script**:
   ```bash
   python script_08.py
   ```

3. **What you'll see**:
   ```
   Indexed 45 chunks into Chroma.
   
   Retrieved 20 docs. First snippet:
   
   --- Doc 1 (page 3) ---
   Large Language Models (LLMs) have shown remarkable capabilities...
   
   --- Answer ---
   
   Based on the document, the paper discusses several popular LLMs including GPT-3, GPT-4, BERT, and T5. According to page 3, these models were evaluated for their performance in various tasks...
   ```

## 🎓 Key Concepts You've Learned

### RAG (Retrieval-Augmented Generation)
- **What**: AI system that reads your documents to answer questions
- **Why**: Gives AI access to your specific information
- **How**: Vector search + AI generation

### Vector Embeddings
- **What**: Mathematical representations of text meaning
- **Why**: Computers can search numbers faster than text
- **How**: OpenAI's embedding models convert text to vectors

### Vector Databases
- **What**: Specialized databases for similarity search
- **Why**: Find relevant document chunks quickly
- **How**: Store embeddings and search by similarity

### Document Chunking
- **What**: Breaking long documents into smaller pieces
- **Why**: AI has limited context window
- **How**: Split by characters with overlap to preserve meaning

### Retrieval Strategies
- **What**: Methods for finding relevant information
- **Why**: Better retrieval = better answers
- **How**: MMR (Maximum Marginal Relevance) for diverse results

## 🔧 Common Issues and Solutions

**Problem: "PDF not found"**
```bash
# Solution: Create directory and add PDF
mkdir -p pdfs
# Place your PDF file as: pdfs/rag_vs_fine_tuning.pdf
```

**Problem: "Empty or poor answers"**
```python
# Solution: Adjust chunk size and overlap
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,     # Larger chunks for more context
    chunk_overlap=200   # More overlap to preserve meaning
)
```

**Problem: "Retrieval finds wrong information"**
```python
# Solution: Adjust retrieval parameters
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 10,           # Fewer results for precision
        "fetch_k": 30,     # More candidates to choose from
        "lambda_mult": 0.5 # Higher relevance vs. diversity
    }
)
```

**Problem: "AI makes up information"**
```python
# Solution: Strengthen the prompt instructions
message = """
CRITICAL: Answer ONLY using the provided context. 
If the information is not in the context, you MUST respond with "I don't know."
Do NOT use your general knowledge. Only use the context below.

Context: {context}
Question: {question}
Answer:
"""
```

## 🎯 Try These Experiments

### 1. Different Types of Questions
```python
# Factual questions
question = "What are the main conclusions of this paper?"

# Specific details
question = "What datasets were used in the experiments?"

# Comparative questions
question = "How does approach A compare to approach B?"
```

### 2. Adjust Chunk Sizes
```python
# For detailed analysis - larger chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,    # More context per chunk
    chunk_overlap=200
)

# For quick facts - smaller chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,     # Focused, specific chunks
    chunk_overlap=50
)
```

### 3. Different Document Types
```python
# For different file types, use different loaders:
from langchain_community.document_loaders import TextLoader, CSVLoader

# For .txt files
loader = TextLoader("document.txt")

# For .csv files  
loader = CSVLoader("data.csv")
```

### 4. Multiple Documents
```python
def build_multi_doc_vectorstore():
    all_docs = []
    
    # Load multiple PDFs
    for pdf_path in ["doc1.pdf", "doc2.pdf", "doc3.pdf"]:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        all_docs.extend(docs)
    
    # Process all documents together
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    splits = splitter.split_documents(all_docs)
    
    # Create combined vector store
    vectorstore = Chroma.from_documents(splits, embeddings, persist_directory=DB_DIR)
    return vectorstore
```

## 🌟 Advanced RAG Techniques

### 1. Metadata Filtering
```python
# Add metadata to documents before storing
for doc in docs:
    doc.metadata["source"] = "research_paper"
    doc.metadata["section"] = "methodology"

# Filter retrieval by metadata
retriever = vectorstore.as_retriever(
    search_kwargs={
        "filter": {"source": "research_paper"}
    }
)
```

### 2. Re-ranking Results
```python
# Get more candidates, then re-rank
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 5,          # Final number
        "fetch_k": 20,   # Initial candidates
    }
)
```

### 3. Query Expansion
```python
# Expand user question before searching
expand_prompt = PromptTemplate.from_template(
    "Rephrase this question to be more specific and detailed: {question}"
)

expanded_question_chain = expand_prompt | llm | StrOutputParser()
```

## 🌟 What's Next?

Now that you understand basic RAG, you're ready to learn about:

- **Advanced RAG** (Tutorial 9) - Smart retrieval, persistent storage, and optimization techniques

Congratulations! You've built an AI that can read and understand your documents! 🎉

## 💡 Real-World Applications

- **Legal Research**: Query legal documents and case files
- **Medical Analysis**: Analyze research papers and medical records  
- **Business Intelligence**: Extract insights from company reports
- **Academic Research**: Search through research papers and literature
- **Customer Support**: Create AI that knows your product documentation
- **Personal Knowledge Base**: Ask questions about your notes and documents