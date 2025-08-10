# 🎯 Activity 01: Building Your Agent's Toolbox - Master Solution Guide

## 📋 Activity Overview

**Topic:** Creating a Wikipedia search tool using LangChain's `@tool` decorator  
**Duration:** 30-45 minutes  
**Difficulty:** Beginner to Intermediate  
**Prerequisites:** Basic Python, understanding of APIs

## 🏆 Complete Solution

### Step 1: Environment Setup

```python
# Install required libraries
!pip install --quiet wikipedia==1.4.0 langchain-core==0.3.59
```

### Step 2: Import Dependencies

```python
from typing import Annotated
import wikipedia
from langchain_core.tools import tool
```

**Explanation:**
- `Annotated`: Provides metadata for type hints, crucial for LangChain tool parameter descriptions
- `wikipedia`: Python library that interfaces with Wikipedia's API
- `tool`: Decorator that converts Python functions into LangChain tools

### Step 3: Complete Wikipedia Tool Implementation

```python
@tool
def wikipedia_tool(
    query: Annotated[str, "The Wikipedia search to execute to find key summary information."],
):
    """Use this to search Wikipedia for factual information."""
    try:
        # Step 1: Search using query
        results = wikipedia.search(query)
        
        if not results:
            return "No results found on Wikipedia."
        
        # Step 2: Retrieve page title (most relevant result)
        title = results[0]

        # Step 3: Fetch summary with controlled parameters
        summary = wikipedia.summary(
            title, 
            sentences=8, 
            auto_suggest=False, 
            redirect=True
        )
        
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    
    return f"Successfully executed:\\nWikipedia summary: {summary}"
```

### Step 4: Testing Implementation

```python
# Test with Apple Inc.
company_name = "Apple Inc."
wiki_summary = wikipedia_tool.invoke(company_name)
print(wiki_summary)
```

## 🧠 Code Breakdown & Best Practices

### 1. Tool Decorator Deep Dive

```python
@tool
def wikipedia_tool(
    query: Annotated[str, "The Wikipedia search to execute to find key summary information."],
):
```

**Key Points:**
- **`@tool` decorator:** Automatically registers the function as a LangChain tool
- **Parameter annotation:** `Annotated[str, "description"]` helps AI agents understand when and how to use the tool
- **Clear description:** Essential for agent decision-making

### 2. Wikipedia API Integration

```python
# Search for relevant pages
results = wikipedia.search(query)

# Get the most relevant page title
title = results[0]

# Fetch controlled summary
summary = wikipedia.summary(
    title, 
    sentences=8,        # Limit to 8 sentences for concise info
    auto_suggest=False, # Prevent automatic query suggestions
    redirect=True       # Follow redirects to actual pages
)
```

**Parameter Explanations:**
- **`sentences=8`:** Balances information detail with response length
- **`auto_suggest=False`:** Prevents disambiguation issues
- **`redirect=True`:** Handles page redirects (e.g., "Apple" → "Apple Inc.")

### 3. Robust Error Handling

```python
try:
    # Wikipedia API operations
except BaseException as e:
    return f"Failed to execute. Error: {repr(e)}"
```

**Error Scenarios Handled:**
- **`wikipedia.exceptions.DisambiguationError`:** Multiple pages match query
- **`wikipedia.exceptions.PageError`:** Page doesn't exist
- **Network errors:** Connection timeouts or API failures
- **General exceptions:** Unexpected errors

### 4. Response Formatting

```python
return f"Successfully executed:\\nWikipedia summary: {summary}"
```

**Best Practice Features:**
- **Status indication:** "Successfully executed" confirms operation completion
- **Clear structure:** Separates status from content
- **Formatted output:** Easy to parse for downstream processing

## 🧪 Advanced Testing Suite

### Test Suite 1: Basic Functionality

```python
def test_basic_functionality():
    """Test the tool with common company names"""
    test_cases = [
        "Apple Inc.",
        "Microsoft Corporation", 
        "Tesla Inc.",
        "Amazon.com",
        "Meta Platforms"
    ]
    
    for company in test_cases:
        print(f"\\n--- Testing: {company} ---")
        result = wikipedia_tool.invoke(company)
        
        # Assertions for testing
        assert "Successfully executed" in result
        assert "Wikipedia summary:" in result
        assert len(result) > 100  # Ensure substantial content
        
        print("✅ Test passed")
        print(result[:150] + "..." if len(result) > 150 else result)
```

### Test Suite 2: Edge Cases

```python
def test_edge_cases():
    """Test tool with challenging inputs"""
    edge_cases = [
        ("", "Empty string"),
        ("XYZABC Fake Company", "Non-existent company"),
        ("Apple", "Ambiguous term"),
        ("AI", "Very broad term"),
        ("123456", "Numeric input")
    ]
    
    for query, description in edge_cases:
        print(f"\\n--- Testing: {description} ('{query}') ---")
        result = wikipedia_tool.invoke(query)
        print(result)
        
        # Should not crash, should return informative response
        assert isinstance(result, str)
        assert len(result) > 0
```

### Test Suite 3: Performance Testing

```python
import time

def test_performance():
    """Test tool response times"""
    test_queries = ["Apple Inc.", "Microsoft", "Tesla"]
    
    for query in test_queries:
        start_time = time.time()
        result = wikipedia_tool.invoke(query)
        end_time = time.time()
        
        execution_time = end_time - start_time
        print(f"Query: '{query}' - Execution time: {execution_time:.2f}s")
        
        # Should complete within reasonable time
        assert execution_time < 10.0  # Max 10 seconds
```

## 🎓 Educational Insights

### Why This Implementation is Production-Ready

1. **Comprehensive Error Handling**
   - Catches all possible Wikipedia API exceptions
   - Returns informative error messages instead of crashing
   - Maintains tool stability in unpredictable environments

2. **Configurable Parameters**
   - `sentences=8` provides balanced information density
   - `auto_suggest=False` prevents unexpected query modifications
   - `redirect=True` handles page redirects seamlessly

3. **Clear Interface Design**
   - Descriptive parameter annotations for AI agents
   - Consistent response format
   - User-friendly success/error messages

4. **Performance Considerations**
   - Limited sentence count prevents excessive data retrieval
   - Single API call per tool invocation
   - Efficient error handling without retries

### Common Student Mistakes & Solutions

#### ❌ Mistake 1: Missing Error Handling
```python
# Wrong - no error handling
def wikipedia_tool(query):
    results = wikipedia.search(query)  # Can throw exceptions
    return wikipedia.summary(results[0])
```

**✅ Solution:** Always wrap external API calls in try-except blocks

#### ❌ Mistake 2: Poor Type Annotations
```python
# Wrong - missing annotations
def wikipedia_tool(query):
    pass
```

**✅ Solution:** Use `Annotated[type, "description"]` for agent compatibility

#### ❌ Mistake 3: Uncontrolled Summary Length
```python
# Wrong - no sentence limit
summary = wikipedia.summary(title)  # Could return very long text
```

**✅ Solution:** Always specify `sentences` parameter for consistent output length

#### ❌ Mistake 4: Not Handling Empty Results
```python
# Wrong - assumes results exist
title = results[0]  # IndexError if results is empty
```

**✅ Solution:** Check if results exist before accessing them

## 🔧 Tool Variations & Extensions

### Variation 1: Customizable Summary Length

```python
@tool
def flexible_wikipedia_tool(
    query: Annotated[str, "The Wikipedia search query"],
    sentences: Annotated[int, "Number of sentences in summary"] = 8,
):
    """Wikipedia tool with configurable summary length."""
    try:
        results = wikipedia.search(query)
        if not results:
            return "No results found on Wikipedia."
        
        title = results[0]
        summary = wikipedia.summary(title, sentences=sentences, auto_suggest=False, redirect=True)
        
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    
    return f"Successfully executed:\\nWikipedia summary: {summary}"
```

### Variation 2: Multi-Result Tool

```python
@tool
def multi_result_wikipedia_tool(
    query: Annotated[str, "The Wikipedia search query"],
    max_results: Annotated[int, "Maximum number of results to return"] = 3,
):
    """Wikipedia tool that returns multiple search results."""
    try:
        results = wikipedia.search(query)[:max_results]
        
        if not results:
            return "No results found on Wikipedia."
        
        summaries = []
        for title in results:
            try:
                summary = wikipedia.summary(title, sentences=3, auto_suggest=False, redirect=True)
                summaries.append(f"**{title}**: {summary}")
            except:
                continue  # Skip problematic pages
        
        return f"Successfully executed:\\n\\n" + "\\n\\n".join(summaries)
        
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
```

### Variation 3: Enhanced Metadata Tool

```python
@tool
def detailed_wikipedia_tool(
    query: Annotated[str, "The Wikipedia search query"],
):
    """Wikipedia tool with additional metadata."""
    try:
        results = wikipedia.search(query)
        if not results:
            return "No results found on Wikipedia."
        
        title = results[0]
        page = wikipedia.page(title)
        
        # Get additional metadata
        summary = wikipedia.summary(title, sentences=8, auto_suggest=False, redirect=True)
        url = page.url
        categories = page.categories[:5]  # First 5 categories
        
        result = f"""Successfully executed:
Wikipedia summary: {summary}

Additional Information:
- Page URL: {url}
- Categories: {', '.join(categories)}
- Page Title: {title}"""
        
        return result
        
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
```

## 📊 Assessment Rubric

### Functionality (40 points)
- **Complete implementation:** All required functions work (20 pts)
- **Error handling:** Properly catches and handles exceptions (10 pts)
- **Tool invocation:** Successfully calls tool with `.invoke()` (10 pts)

### Code Quality (30 points)
- **Type annotations:** Proper use of `Annotated` (10 pts)
- **Documentation:** Clear docstrings and comments (10 pts)
- **Code structure:** Clean, readable implementation (10 pts)

### Understanding (30 points)
- **Concept explanation:** Can explain how tools work (15 pts)
- **Parameter understanding:** Knows why each parameter is used (15 pts)

### Total: 100 points

**Grading Scale:**
- 90-100: Excellent understanding
- 80-89: Good implementation with minor issues
- 70-79: Basic functionality, needs improvement
- Below 70: Requires additional practice

## 🚀 Next Steps

After completing this activity:

1. **Immediate:** Review and understand all code components
2. **Practice:** Create variations with different APIs
3. **Integration:** Combine with other tools in agent workflows
4. **Advanced:** Add caching, rate limiting, or result filtering

## 🔗 Related Resources

- **LangChain Tools Documentation:** Understanding tool patterns
- **Wikipedia API Documentation:** Advanced search parameters
- **Error Handling Best Practices:** Robust tool development
- **Multi-Agent Systems:** Using tools in complex workflows

## 💡 Pro Tips for Instructors

1. **Encourage Experimentation:** Let students modify sentence limits and test different companies
2. **Debug Together:** Walk through common errors with the class
3. **Real-World Applications:** Discuss how this tool fits into larger systems
4. **Performance Awareness:** Explain why we limit API calls and response sizes
5. **Error Scenarios:** Intentionally cause errors to show how handling works

This activity establishes fundamental tool-building skills that are essential for advanced LangChain development! 🎓