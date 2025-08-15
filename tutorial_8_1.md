# Tutorial 8.1: RetrievalQA vs Custom RAG Chain

This tutorial compares two approaches for building RAG (Retrieval-Augmented Generation) systems in LangChain: using the high-level `RetrievalQA` chain vs building a custom chain with individual components.

## Overview

We have two implementations:
- **script_08.py**: Custom RAG chain using `RunnablePassthrough` and `RunnableLambda`
- **script_8_1.py**: Using the pre-built `RetrievalQA` chain

## Key Differences

### RetrievalQA Approach (script_8_1.py)

```python
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

result = qa_chain.invoke({"query": question})
```

**Advantages:**
- **Simpler setup**: One line to create the chain
- **Built-in features**: Automatic source document return
- **Standardized**: Uses well-tested chain patterns
- **Less code**: Fewer moving parts to manage

**Disadvantages:**
- **Less flexibility**: Limited customization options
- **Fixed prompt**: Can't easily modify the prompt template
- **Black box**: Less control over document formatting and chain flow

### Custom Chain Approach (script_08.py)

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

**Advantages:**
- **Full control**: Custom prompts, document formatting, chain logic
- **Transparency**: Clear data flow through the chain
- **Extensible**: Easy to add custom processing steps
- **Debugging**: Better visibility into each component

**Disadvantages:**
- **More complex**: Requires understanding of chain composition
- **More code**: Need to handle more details manually
- **Potential errors**: More places where things can go wrong

## When to Use Each Approach

### Use RetrievalQA when:
- Building a quick prototype or proof-of-concept
- Standard RAG requirements are sufficient
- You want minimal setup time
- Team prefers battle-tested solutions

### Use Custom Chain when:
- Need custom prompt templates or document formatting
- Require specific chain logic or preprocessing
- Want to add custom retrieval strategies
- Need detailed debugging capabilities
- Building production systems with specific requirements

## Code Comparison

| Feature | RetrievalQA | Custom Chain |
|---------|-------------|--------------|
| Setup complexity | Low | Medium |
| Customization | Limited | Full |
| Document formatting | Fixed | Custom (`format_docs`) |
| Prompt control | Limited | Full |
| Source documents | Built-in | Manual |
| Chain composition | Hidden | Explicit |

## Running the Examples

Both scripts use the same PDF processing and vectorstore setup. The main differences are in the chain construction and invocation:

```bash
# RetrievalQA approach
python script_8_1.py

# Custom chain approach  
python script_08.py
```

## Best Practices

1. **Start with RetrievalQA** for initial development
2. **Switch to custom chains** when you need specific customizations
3. **Use source documents** to verify retrieval quality
4. **Test both approaches** to understand the trade-offs for your use case

## Conclusion

RetrievalQA provides a quick, reliable way to build RAG systems, while custom chains offer maximum flexibility. Choose based on your specific requirements for customization, debugging needs, and development timeline.