# 🛠️ Activity 01: Building Your Agent's Toolbox - Student Practice Guide

## 🎯 Learning Objectives

By the end of this activity, you will:
- Understand how to create custom tools using the `@tool` decorator
- Learn to integrate external APIs (Wikipedia) into LangChain tools
- Master error handling and tool invocation patterns
- Build the foundation for multi-agent systems

## 📚 Background Context

**What are LangChain Tools?**
Tools are functions that agents can call to perform specific actions like:
- 🔍 Searching the internet (Wikipedia, Google)
- 📊 Retrieving data from databases or files
- 🐍 Executing Python code
- 🧮 Performing calculations

**Why Wikipedia Tool?**
We're building a financial analysis system that needs to:
1. **Get company information** → Wikipedia search tool
2. **Retrieve stock data** → Stock data tool (next activity)
3. **Create visualizations** → Python execution tool (next activity)

## 🔧 Setup Instructions

### Step 1: Install Required Libraries
```bash
pip install --quiet wikipedia==1.4.0 langchain-core==0.3.59
```

### Step 2: Import Dependencies
```python
# TODO: Import the required modules
# Hint: You need typing.Annotated, wikipedia, and langchain_core.tools.tool
from typing import ________
import ________
from langchain_core.tools import ________
```

<details>
<summary>💡 Hint for Step 2</summary>

You need to import:
- `Annotated` from typing (for type hints)
- `wikipedia` library (for Wikipedia API calls)  
- `tool` decorator from langchain_core.tools
</details>

## 🏗️ Building the Wikipedia Tool

### Step 3: Create the Tool Function Structure

**Your task:** Complete the `wikipedia_tool` function below. You need to write about **65%** of the implementation.

```python
@tool
def wikipedia_tool(
    query: Annotated[str, "The Wikipedia search to execute to find key summary information."],
):
    """Use this to search Wikipedia for factual information."""
    try:
        # TODO: Step 3a - Search Wikipedia using the query
        # Hint: Use wikipedia.search() function
        results = wikipedia.________(query)
        
        # TODO: Step 3b - Handle the case when no results are found
        # Hint: Check if results is empty and return appropriate message
        if not ________:
            return "No results found on Wikipedia."
        
        # TODO: Step 3c - Get the first (most relevant) result
        # Hint: Access the first element of the results list
        title = ________[0]

        # TODO: Step 3d - Fetch the summary for the title
        # Hint: Use wikipedia.summary() with parameters:
        # - title: the page title
        # - sentences: limit to 8 sentences
        # - auto_suggest: set to False
        # - redirect: set to True
        summary = wikipedia.________(
            ________, 
            sentences=________, 
            auto_suggest=________, 
            redirect=________
        )
        
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    
    # TODO: Step 3e - Return the formatted result
    # Hint: Return a string that starts with "Successfully executed:"
    return f"Successfully executed:\\nWikipedia summary: ________"
```

<details>
<summary>🔍 Step-by-Step Hints</summary>

**Step 3a:** `wikipedia.search(query)` returns a list of page titles
**Step 3b:** Check if `results` is empty using `not results`
**Step 3c:** Get the first result with `results[0]`
**Step 3d:** Use `wikipedia.summary(title, sentences=8, auto_suggest=False, redirect=True)`
**Step 3e:** Include the `summary` variable in your return statement
</details>

### Step 4: Test Your Tool

Now test your Wikipedia tool with a company search:

```python
# TODO: Step 4a - Define a test company
# Hint: Use "Apple Inc." as the company name
company_name = "________"

# TODO: Step 4b - Invoke the tool
# Hint: Use the .invoke() method and pass the company_name
wiki_summary = wikipedia_tool.________(company_name)

# TODO: Step 4c - Print the results
# Hint: Use print() to display the wiki_summary
________(wiki_summary)
```

<details>
<summary>💡 Testing Hints</summary>

- Use `"Apple Inc."` as your test company
- Tools are invoked with `.invoke(input_parameter)`
- Use `print(wiki_summary)` to see the results
</details>

## ✅ Expected Output

Your tool should return something like:

```
Successfully executed:
Wikipedia summary: Apple Inc. is an American multinational corporation and technology company headquartered in Cupertino, California, in Silicon Valley. It is best known for its consumer electronics, software, and services. Founded in 1976 as Apple Computer Company by Steve Jobs, Steve Wozniak and Ronald Wayne, the company was incorporated by Jobs and Wozniak as Apple Computer, Inc. the following year. It was renamed Apple Inc. in 2007 as the company had expanded its focus from computers to consumer electronics. Apple is the largest technology company by revenue, with US$391.04 billion in the 2024 fiscal year.
```

## 🎓 Understanding Your Code

### Key Concepts Explained:

**1. The `@tool` Decorator:**
```python
@tool
def my_function():
    pass
```
- Converts a regular Python function into a LangChain tool
- Allows agents to discover and call this function
- Automatically handles serialization and metadata

**2. Type Annotations:**
```python
query: Annotated[str, "Description of parameter"]
```
- `Annotated` adds metadata to type hints
- The description helps AI agents understand when to use this tool

**3. Error Handling:**
```python
try:
    # risky operations
except BaseException as e:
    return f"Failed to execute. Error: {repr(e)}"
```
- Catches all exceptions to prevent crashes
- Returns informative error messages instead of failing

## 🔧 Troubleshooting Guide

### Common Issues & Solutions:

**❌ "No module named 'wikipedia'"**
```bash
# Solution: Install the wikipedia library
pip install wikipedia==1.4.0
```

**❌ "DisambiguationError"**  
This happens when Wikipedia finds multiple pages with similar names.
- **Solution:** Our tool handles this with `auto_suggest=False` and takes the first result

**❌ "PageError"**  
This happens when a page doesn't exist.
- **Solution:** The try/except block catches this and returns an error message

**❌ Tool returns "Failed to execute"**
- **Check:** Your search query - try simpler terms
- **Debug:** Print the `results` list to see what Wikipedia found

## 🧪 Testing Challenges

Once you've completed the basic implementation, try these additional tests:

### Challenge 1: Test Different Companies
```python
test_companies = ["Microsoft", "Tesla Inc.", "Amazon.com", "Meta Platforms"]

for company in test_companies:
    result = wikipedia_tool.invoke(company)
    print(f"\\n--- {company} ---")
    print(result[:200] + "..." if len(result) > 200 else result)
```

### Challenge 2: Handle Edge Cases
```python
# Test with a non-existent company
weird_queries = ["XYZABC Fake Company", "AAAAA", ""]

for query in weird_queries:
    result = wikipedia_tool.invoke(query)
    print(f"\\nQuery: '{query}'")
    print(result)
```

## 🚀 Next Steps

After completing this activity:

1. **Activity 02:** Build a stock data retrieval tool
2. **Activity 03:** Create a Python code execution tool  
3. **Activity 04:** Combine all tools into a single agent
4. **Activity 05:** Build conditional workflows with multiple tools

## 📝 Self-Assessment

**Check your understanding:**

□ I can explain what the `@tool` decorator does  
□ I understand how to use `Annotated` type hints  
□ I can handle errors in tool functions  
□ I know how to invoke tools with `.invoke()`  
□ I understand why we limit summary sentences to 8  
□ I can modify the tool to get different amounts of information  

## 🎉 Congratulations!

You've successfully created your first LangChain tool! This Wikipedia search tool will be a crucial component in your multi-agent financial analysis system.

**Key Takeaways:**
- Tools bridge the gap between AI agents and external APIs
- Proper error handling makes tools robust and reliable
- Type annotations help AI agents understand tool parameters
- Tools are building blocks for complex agent workflows

Ready for the next challenge? Let's build that stock data tool! 🚀