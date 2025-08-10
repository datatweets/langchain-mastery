# 🔨 Activity 05: Integrating Tools with LangGraph - Student Practice Guide

## 🎯 Learning Objectives

By the end of this activity, you will:
- Integrate your previously built tools (Wikipedia, Stock Data, Python REPL) into LangGraph workflows
- Learn how LLMs select and use tools through tool binding
- Master the ToolNode pattern for graph-based tool execution
- Understand tool descriptions and their role in LLM decision-making
- Build your first multi-tool agent system with linear workflow

## 📚 Background Context

**The Power of Tool Integration**
You've built three powerful tools:
1. **✅ Wikipedia Tool** - Company research and information retrieval
2. **✅ Stock Data Tool** - Historical stock performance from CSV files
3. **✅ Python REPL Tool** - Dynamic code execution and visualization

Now you'll learn how to integrate these tools into a LangGraph workflow, enabling the LLM to:
- **Automatically select** the right tool for each task
- **Chain tools together** for complex multi-step workflows
- **Make intelligent decisions** about when tools are needed

**Tool Integration Architecture:**

```
User Input → LLM Node → Tool Selection → Tool Execution → Response
     ↑                     ↓              ↓            ↓
     └─────── State ←───── State ←───── State ←───── State
```

## 🛠️ Tool Integration Concepts

### How LLMs Choose Tools

**Tool Binding Process:**
1. **Tool Descriptions**: LLM reads tool docstrings and parameter annotations
2. **Intent Analysis**: LLM analyzes user request to determine required actions
3. **Tool Selection**: LLM chooses appropriate tool(s) based on capabilities
4. **Parameter Extraction**: LLM extracts required parameters from user input
5. **Execution**: Graph routes to tool node for actual tool execution

**Example Decision Flow:**
```python
User: "Tell me about Apple Inc."
LLM Thinking: "This requires factual information → Wikipedia Tool"

User: "Show AAPL stock prices"  
LLM Thinking: "This needs stock data → Stock Data Tool"

User: "Create a price chart"
LLM Thinking: "This needs visualization → Python REPL Tool"
```

## 🔧 Setup Instructions

### Step 1: Install Required Libraries
```bash
pip install --quiet wikipedia==1.4.0 langchain-core==0.3.59 langgraph==0.5.3 langchain-openai==0.3.16 langchain-experimental==0.3.4
```

### Step 2: Import Dependencies and Define Tools
```python
# The tools you've already created are pre-defined
# Let's examine their structure first
```

## 🔍 Understanding Tool Descriptions

### Step 3: Examine Tool Metadata

**Your task:** Explore how tools expose their capabilities to LLMs.

```python
# TODO: Step 3 - Loop through tools and extract their descriptions
# Hint: Use .name and .description attributes
for tool in [wikipedia_tool, stock_data_tool, python_repl_tool]:
    print(f"The description for {tool.________}:\n> {tool.________}\n")
```

<details>
<summary>💡 Step 3 Hint</summary>

Tools have `.name` and `.description` attributes that come from their docstrings and function names. These are what the LLM uses to understand tool capabilities.
</details>

### Tool Description Analysis

**What makes a good tool description?**
- **Clear Purpose**: What the tool does
- **When to Use**: Specific use cases
- **Input Requirements**: What parameters are needed
- **Output Format**: What the tool returns

**Example Analysis:**
```python
@tool
def wikipedia_tool(query: Annotated[str, "The Wikipedia search..."]):
    """Use this to search Wikipedia for factual information."""
    # The docstring becomes the description!
```

## 🏗️ Building the Tool-Integrated Graph

### Step 4: Set Up Graph Components

**Your task:** Complete the graph setup with tool binding. You need to write about **70%** of the implementation.

```python
# TODO: Import required modules
from typing import ________
from typing_extensions import ________
from langgraph.graph import ________, ________, ________
from langgraph.graph.message import ________
from langchain_openai import ________
from langgraph.prebuilt import ________

# TODO: Step 4a - Define the State class
class ________(TypedDict):
    messages: Annotated[________, ________]

# TODO: Step 4b - Create the graph builder
graph_builder = ________(________)

# TODO: Step 4c - Create tools list with all three tools
# Hint: Include wikipedia_tool, stock_data_tool, and python_repl_tool
tools = [________, ________, ________]

# TODO: Step 4d - Create LLM instance
llm = ________(model="________")

# TODO: Step 4e - Bind tools to the LLM
# Hint: Use bind_tools() method
llm_with_tools = llm.________(________)
```

<details>
<summary>🔍 Step 4 Hints</summary>

**Step 4a:** `State(TypedDict)` with `messages: Annotated[list, add_messages]`
**Step 4b:** `StateGraph(State)`  
**Step 4c:** List all three tool functions
**Step 4d:** `ChatOpenAI(model="gpt-4o-mini")`
**Step 4e:** `llm.bind_tools(tools)`
</details>

### Step 5: Create Node Functions

**Your task:** Define the LLM node that uses bound tools.

```python
# TODO: Step 5 - Create the llm_node function
# Hint: Invoke the llm_with_tools with state messages
def llm_node(state: State):
    return {"messages": [________.________(state["messages"])]}
```

<details>
<summary>💡 Step 5 Hint</summary>

Use `llm_with_tools.invoke(state["messages"])` to process messages with tool awareness.
</details>

### Step 6: Build the Graph Structure

**Your task:** Create the complete graph with LLM and tool nodes.

```python
# TODO: Step 6a - Add the LLM node
# Hint: Use add_node() with "llm" name and llm_node function
graph_builder.________(________, ________)

# TODO: Step 6b - Create tool node using ToolNode class
# Hint: ToolNode takes the tools list as parameter
tool_node = ________(________)

# TODO: Step 6c - Add the tool node to the graph
# Hint: Use add_node() with "tools" name and tool_node
graph_builder.________(________, ________)

# TODO: Step 6d - Create linear workflow edges
# Hint: Connect START → llm → tools → END
graph_builder.________(________, ________)
graph_builder.________(________, ________)
graph_builder.________(________, ________)

# TODO: Step 6e - Compile the graph
graph = graph_builder.________()
```

<details>
<summary>🔍 Step 6 Hints</summary>

**Step 6a:** `graph_builder.add_node("llm", llm_node)`
**Step 6b:** `ToolNode(tools)`
**Step 6c:** `graph_builder.add_node("tools", tool_node)`
**Step 6d:** `START → "llm" → "tools" → END`
**Step 6e:** `graph_builder.compile()`
</details>

### Step 7: Visualize and Test

```python
# TODO: Step 7a - Display graph visualization
________

# TODO: Step 7b - Test with different types of requests
from course_helper_functions import pretty_print_messages

# Test 1: Wikipedia request
for chunk in graph.stream(
    {"messages": [{"role": "user", "content": "Tell me about Apple Inc."}]}
):
    pretty_print_messages(chunk)
```

## ✅ Expected Behavior

Your integrated agent should handle different request types:

### Test Scenarios

**1. Company Information Request:**
```
Input: "Tell me about Apple Inc."
Expected: Uses Wikipedia tool → Returns company information
```

**2. Stock Data Request:**
```  
Input: "Show me AAPL stock prices for the last 5 days"
Expected: Uses Stock Data tool → Returns price table
```

**3. Visualization Request:**
```
Input: "Create a chart of AAPL stock prices"  
Expected: Uses Python REPL tool → Generates visualization
```

**4. General Conversation:**
```
Input: "Hello, my name is John"
Expected: Direct LLM response (no tools needed)
```

## 🎓 Understanding Your Code

### Key Concepts Explained:

**1. Tool Binding Mechanism:**
```python
llm_with_tools = llm.bind_tools(tools)
```
- **Capability Advertisement**: LLM learns about available tools
- **Parameter Mapping**: LLM understands tool input requirements
- **Decision Framework**: LLM can choose when and how to use tools

**2. ToolNode Architecture:**
```python
tool_node = ToolNode(tools)
```
- **Unified Interface**: Single node handles all tool executions
- **Dynamic Routing**: Automatically calls the requested tool
- **Error Handling**: Manages tool execution failures
- **Result Integration**: Returns tool results in proper format

**3. Linear Workflow Pattern:**
```python
START → "llm" → "tools" → END
```
- **Sequential Processing**: Every request goes through both nodes
- **Tool-First Design**: Always attempts tool usage
- **Predictable Flow**: Easy to debug and understand

**4. Message Flow with Tools:**
```python
# Initial state
{"messages": [{"role": "user", "content": "Tell me about Apple"}]}

# After LLM node (tool call generated)
{"messages": [
    {"role": "user", "content": "Tell me about Apple"},
    {"role": "assistant", "tool_calls": [{"name": "wikipedia_tool", "args": {"query": "Apple Inc."}}]}
]}

# After tool node (tool result added)
{"messages": [
    {"role": "user", "content": "Tell me about Apple"},
    {"role": "assistant", "tool_calls": [...]},
    {"role": "tool", "content": "Apple Inc. is a multinational..."}
]}
```

## 🔧 Troubleshooting Guide

### Common Issues & Solutions:

**❌ "Tools not being called"**
- **Check:** Tool descriptions are clear and specific
- **Verify:** Tools are properly bound with `bind_tools()`
- **Debug:** Test with explicit tool-requiring requests

**❌ "Tool execution fails"**
```python
# Problem: Missing data files or incorrect paths
# Solution: Ensure CSV files are in data/ directory
```

**❌ "Graph compilation error"**
- **Check:** All nodes are properly added
- **Verify:** Edges connect valid node names
- **Ensure:** State class is correctly defined

**❌ "Tool results not formatted properly"**
- **Check:** Tools return strings (not objects)
- **Verify:** Tool outputs are properly formatted
- **Ensure:** Python REPL includes `plt.show()` for charts

## 🧪 Testing Challenges

### Challenge 1: Tool Selection Testing
```python
# Test different types of requests to verify tool selection
test_requests = [
    "What is Tesla's business model?",  # Should use Wikipedia
    "Show Tesla stock data for 10 days",  # Should use Stock Data
    "Plot Tesla stock prices",  # Should use Python REPL
    "Hello, how are you today?",  # Should use direct LLM
]

for request in test_requests:
    print(f"\n--- Testing: {request} ---")
    for chunk in graph.stream({"messages": [{"role": "user", "content": request}]}):
        pretty_print_messages(chunk)
```

### Challenge 2: Multi-Step Workflow Testing
```python
# Test complex requests that might require multiple steps
complex_requests = [
    "Research Microsoft and show me their recent stock performance",
    "Find information about Amazon and create a visualization of their stock data",
    "Tell me about Netflix and plot their stock price trend"
]

for request in complex_requests:
    print(f"\n--- Complex Test: {request} ---")
    for chunk in graph.stream({"messages": [{"role": "user", "content": request}]}):
        pretty_print_messages(chunk)
```

### Challenge 3: Error Handling Testing
```python
# Test with requests that might cause errors
error_tests = [
    "Show me stock data for XYZ123",  # Invalid ticker
    "Plot data for a company that doesn't exist",  # Missing data
    "Execute broken Python code: print(undefined_variable)"  # Code error
]

for test in error_tests:
    print(f"\n--- Error Test: {test} ---")
    try:
        for chunk in graph.stream({"messages": [{"role": "user", "content": test}]}):
            pretty_print_messages(chunk)
    except Exception as e:
        print(f"Caught error: {e}")
```

## 🔍 Workflow Analysis

### Understanding Linear vs Conditional Flow

**Current Linear Pattern:**
```
User Input → LLM (always) → Tools (always) → END
```

**Limitations:**
- ✅ **Simple**: Easy to understand and debug
- ❌ **Inefficient**: Always goes to tools, even for simple questions
- ❌ **Rigid**: Cannot adapt flow based on needs

**Future Conditional Pattern:**
```
User Input → LLM → Decision Point → Tools (if needed) → END
                                 → Direct Response (if no tools)
```

### Tool Interaction Patterns

**1. Single Tool Usage:**
```
"Tell me about Apple" → Wikipedia Tool → Response
```

**2. Sequential Tool Usage (Future):**
```
"Research Apple and show stock data" → Wikipedia → Stock Data → Response
```

**3. Tool Chaining (Advanced):**
```
"Plot Apple's performance" → Stock Data → Python REPL → Response
```

## 🚀 Extension Experiments

### Experiment 1: Custom Tool Integration
```python
# Add a simple calculation tool
@tool
def calculator_tool(
    expression: Annotated[str, "Mathematical expression to evaluate"]
):
    """Use this to perform mathematical calculations."""
    try:
        # Safe evaluation of basic math expressions
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Calculation result: {expression} = {result}"
    except:
        return "Invalid mathematical expression"

# Create extended tools list
extended_tools = [wikipedia_tool, stock_data_tool, python_repl_tool, calculator_tool]

# Rebuild graph with additional tool
extended_llm_with_tools = llm.bind_tools(extended_tools)
# ... rebuild graph with extended_llm_with_tools
```

### Experiment 2: Tool Usage Analytics
```python
# Track which tools are being used
tool_usage_stats = {"wikipedia": 0, "stock_data": 0, "python_repl": 0}

def analytics_llm_node(state: State):
    """LLM node with usage tracking"""
    response = llm_with_tools.invoke(state["messages"])
    
    # Track tool calls (simplified tracking)
    if hasattr(response, 'tool_calls') and response.tool_calls:
        for tool_call in response.tool_calls:
            tool_name = tool_call.get('name', '').replace('_tool', '')
            if tool_name in tool_usage_stats:
                tool_usage_stats[tool_name] += 1
    
    return {"messages": [response]}

# Use analytics_llm_node instead of llm_node
# After testing, print tool_usage_stats to see usage patterns
```

### Experiment 3: Tool Result Enhancement
```python
def enhanced_tool_node(state: State):
    """Enhanced tool node with result formatting"""
    # Use standard ToolNode for execution
    standard_result = tool_node.invoke(state)
    
    # Enhance tool results with metadata
    if standard_result.get("messages"):
        for message in standard_result["messages"]:
            if message.get("role") == "tool":
                # Add timestamp and formatting
                enhanced_content = f"🔧 **Tool Result** (Generated: {datetime.now().strftime('%H:%M:%S')})\n\n{message['content']}"
                message["content"] = enhanced_content
    
    return standard_result
```

## 📝 Self-Assessment

**Check your understanding:**

□ I can bind tools to LLMs using `bind_tools()`  
□ I understand how tool descriptions guide LLM decisions  
□ I can create ToolNode instances for graph integration  
□ I know how to build linear workflows with tools  
□ I understand the message flow in tool-enabled graphs  
□ I can debug tool selection and execution issues  
□ I can test different types of tool-requiring requests  

## 💡 Real-World Applications

**Where tool integration is used:**
- **Customer Support**: Knowledge base search + ticket creation + status updates
- **Financial Analysis**: Data retrieval + calculations + report generation
- **Content Creation**: Research + writing + fact-checking + publishing
- **Software Development**: Code search + execution + testing + deployment
- **Data Science**: Data loading + analysis + visualization + reporting

## 🎉 Congratulations!

You've successfully created your first multi-tool agent! This system can:

- ✅ **Automatically select tools** based on user requests
- ✅ **Execute multiple tool types** (research, data, code)
- ✅ **Handle tool errors gracefully** with proper error messages
- ✅ **Maintain conversation state** across tool interactions
- ✅ **Provide rich responses** combining tool results with natural language

**Key Takeaways:**
- Tool binding enables LLMs to understand and use external capabilities
- ToolNode provides unified interface for multiple tools
- Linear workflows ensure predictable tool execution
- Tool descriptions are crucial for proper LLM decision-making

## 🚀 Next Steps

After completing this activity:

1. **Exercise 1.8:** Implement conditional routing for smarter tool usage
2. **Exercise 1.9:** Build multi-agent systems with specialized tools
3. **Exercise 2.1:** Add human-in-the-loop workflows
4. **Advanced:** Create custom tool orchestration patterns

## 🔧 Workflow Limitations & Future Improvements

**Current Linear Workflow Issues:**
1. **Always Executes Tools**: Even simple greetings go through tool node
2. **No Tool Dependencies**: Can't chain tools that depend on each other
3. **Limited Error Recovery**: No fallback strategies for tool failures

**Future Conditional Workflow Benefits:**
1. **Smart Routing**: Only use tools when needed
2. **Tool Chaining**: Sequential tool execution for complex tasks
3. **Error Handling**: Alternative paths for tool failures
4. **Efficiency**: Faster responses for non-tool requests

You're now ready to build more sophisticated conditional workflows and multi-agent systems! 🚀🔨📈

## 🏗️ Architecture Evolution

**Your Learning Journey:**
- **Exercises 1.3-1.5**: Individual tool creation
- **Exercise 1.6**: Graph-based architecture basics
- **Exercise 1.7**: Tool integration with linear flow ← You are here
- **Future**: Conditional routing, multi-agent systems, advanced orchestration

The foundation is solid - now let's make it intelligent! 🧠✨