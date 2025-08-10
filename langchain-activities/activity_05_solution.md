# 🔨 Activity 05: Integrating Tools with LangGraph - Master Solution Guide

## 📋 Activity Overview

**Topic:** Integrating multiple tools into LangGraph workflows with tool binding  
**Duration:** 60-75 minutes  
**Difficulty:** Advanced  
**Prerequisites:** LangGraph basics, tool creation experience, graph state management

## 🏆 Complete Solution

### Step 1: Environment Setup

```python
# Install required libraries
!pip install --quiet wikipedia==1.4.0 langchain-core==0.3.59 langgraph==0.5.3 langchain-openai==0.3.16 langchain-experimental==0.3.4
```

### Step 2: Complete Tool Definitions

```python
from typing import Annotated
import wikipedia
from langchain_core.tools import tool
import pandas as pd
import os
from langchain_experimental.utilities import PythonREPL

# Wikipedia Tool
@tool
def wikipedia_tool(
    query: Annotated[str, "The Wikipedia search to execute to find key summary information."],
):
    """Use this to search Wikipedia for factual information."""
    try:
        results = wikipedia.search(query)
        if not results:
            return "No results found on Wikipedia."
        title = results[0]
        summary = wikipedia.summary(title, sentences=8, auto_suggest=False, redirect=True)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    return f"Successfully executed:\nWikipedia summary: {summary}"

# Stock Data Tool
@tool
def stock_data_tool(
    company_ticker: Annotated[str, "The ticker symbol of the company to retrieve their stock performance data."], 
    num_days: Annotated[int, "The number of business days of stock data required to respond to the user query."]
) -> str:
    """Use this to look-up stock performance data for companies to retrieve a table from a CSV. 
    You may need to convert company names into ticker symbols to call this function, 
    e.g, Apple Inc. -> AAPL, and you may need to convert weeks, months, and years, into days."""
    
    file_path = f"data/{company_ticker}.csv"
    if os.path.exists(file_path) is False:
        return f"Sorry, but data for company {company_ticker} is not available. Please try Apple, Amazon, Meta, Microsoft, Netflix, or Tesla."
    
    stock_df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
    stock_df.index = stock_df.index.date
    
    max_num_days = (stock_df.index.max() - stock_df.index.min()).days
    if num_days > max_num_days:
        return "Sorry, but this time period exceeds the data available. Please reduce it to continue."
    
    final_date = stock_df.index.max()
    filtered_df = stock_df[stock_df.index > (final_date - pd.Timedelta(days=num_days))]
    
    return f"Successfully executed the stock performance data retrieval tool to retrieve the last *{num_days} days* of data for company **{company_ticker}**:\n\n{filtered_df.to_markdown()}"

# Python REPL Tool
repl = PythonREPL()

@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user. 
    The chart should be displayed using `plt.show()`."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    return f"Successfully executed the Python REPL tool.\n\nPython code executed:\n```python\n{code}\n```\n\nCode output:\n```\n{result}\n```"
```

### Step 3: Complete Graph Integration

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode

# Define State
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Create graph builder
graph_builder = StateGraph(State)

# Add three tools to the list
tools = [wikipedia_tool, stock_data_tool, python_repl_tool]

# Create LLM and bind tools
llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)

# Define LLM node
def llm_node(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Build graph structure
graph_builder.add_node("llm", llm_node)

# Create and add tool node
tool_node = ToolNode(tools)
graph_builder.add_node("tools", tool_node)

# Create linear workflow edges
graph_builder.add_edge(START, "llm")
graph_builder.add_edge("llm", "tools")
graph_builder.add_edge("tools", END)

# Compile graph
graph = graph_builder.compile()
```

### Step 4: Tool Description Analysis

```python
# Loop through tools and extract descriptions
for tool in [wikipedia_tool, stock_data_tool, python_repl_tool]:
    print(f"The description for {tool.name}:\n> {tool.description}\n")
```

### Step 5: Testing Implementation

```python
# Visualize graph
graph

# Test with different request types
from course_helper_functions import pretty_print_messages

test_cases = [
    "Tell me about Apple Inc.",
    "Show AAPL stock data for 5 days",
    "Hello, my name is John"
]

for test_input in test_cases:
    print(f"\n--- Testing: {test_input} ---")
    for chunk in graph.stream(
        {"messages": [{"role": "user", "content": test_input}]}
    ):
        pretty_print_messages(chunk)
```

## 🧠 Code Breakdown & Best Practices

### 1. Tool Binding Architecture Deep Dive

```python
llm_with_tools = llm.bind_tools(tools)
```

**How Tool Binding Works:**

1. **Schema Generation**: Each tool's signature is converted to a JSON schema
2. **Description Extraction**: Docstrings become tool descriptions for the LLM
3. **Parameter Mapping**: `Annotated` types become parameter descriptions
4. **Function Registration**: LLM learns when and how to call each tool

**Tool Schema Example:**
```json
{
  "name": "wikipedia_tool",
  "description": "Use this to search Wikipedia for factual information.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "The Wikipedia search to execute to find key summary information."
      }
    },
    "required": ["query"]
  }
}
```

### 2. ToolNode Implementation Analysis

```python
tool_node = ToolNode(tools)
```

**ToolNode Capabilities:**
- **Automatic Routing**: Identifies which tool to call from LLM tool requests
- **Parameter Extraction**: Parses tool arguments from LLM responses
- **Error Handling**: Manages tool execution failures gracefully
- **Result Formatting**: Converts tool outputs to proper message format

**Tool Call Flow:**
```python
# 1. LLM generates tool call
{"role": "assistant", "tool_calls": [
    {"name": "wikipedia_tool", "args": {"query": "Apple Inc."}}
]}

# 2. ToolNode processes the call
tool_result = wikipedia_tool.invoke({"query": "Apple Inc."})

# 3. ToolNode formats result
{"role": "tool", "content": "Successfully executed...", "tool_call_id": "xyz"}
```

### 3. Linear Workflow Pattern Analysis

```python
START → "llm" → "tools" → END
```

**Flow Characteristics:**
- **Predictable**: Every request follows the same path
- **Simple**: Easy to understand and debug
- **Tool-Centric**: Assumes every request needs tools
- **Sequential**: No parallel or conditional processing

**Message Flow Example:**
```python
# Initial state
{"messages": [{"role": "user", "content": "Tell me about Apple"}]}

# After LLM node
{"messages": [
    {"role": "user", "content": "Tell me about Apple"},
    {"role": "assistant", "tool_calls": [{"name": "wikipedia_tool", "args": {"query": "Apple Inc."}}]}
]}

# After tool node
{"messages": [
    {"role": "user", "content": "Tell me about Apple"},
    {"role": "assistant", "tool_calls": [...]},
    {"role": "tool", "content": "Apple Inc. is a multinational corporation..."}
]}
```

### 4. Tool Description Impact on LLM Decision Making

**Critical Design Elements:**

1. **Clear Purpose Statements**:
   ```python
   """Use this to search Wikipedia for factual information."""  # Good
   """A tool that does things."""  # Poor
   ```

2. **Specific Use Cases**:
   ```python
   """You may need to convert company names into ticker symbols to call this function, e.g, Apple Inc. -> AAPL"""
   ```

3. **Parameter Guidance**:
   ```python
   code: Annotated[str, "The python code to execute to generate your chart."]
   ```

4. **Output Expectations**:
   ```python
   """The chart should be displayed using `plt.show()`."""
   ```

## 🧪 Comprehensive Testing Suite

### Test Suite 1: Tool Selection Validation

```python
def test_tool_selection():
    """Test that LLM selects appropriate tools for different request types"""
    
    test_scenarios = [
        # Wikipedia tool scenarios
        ("Tell me about Tesla's business model", "wikipedia_tool"),
        ("What is Microsoft Corporation?", "wikipedia_tool"),
        ("Research Amazon's history", "wikipedia_tool"),
        
        # Stock data tool scenarios  
        ("Show me AAPL stock prices", "stock_data_tool"),
        ("Get Tesla stock data for 10 days", "stock_data_tool"),
        ("What's the recent performance of META stock?", "stock_data_tool"),
        
        # Python REPL tool scenarios
        ("Create a chart of stock prices", "python_repl_tool"),
        ("Plot a graph showing trends", "python_repl_tool"),
        ("Calculate the average of [1,2,3,4,5]", "python_repl_tool"),
        
        # Direct LLM scenarios (should not call tools)
        ("Hello, how are you?", None),
        ("What's your favorite color?", None),
        ("Thank you for your help", None),
    ]
    
    for request, expected_tool in test_scenarios:
        print(f"\n--- Testing: {request} ---")
        print(f"Expected tool: {expected_tool}")
        
        # Stream response and analyze tool calls
        chunks = list(graph.stream({
            "messages": [{"role": "user", "content": request}]
        }))
        
        # Find LLM response chunk
        llm_response = None
        for chunk in chunks:
            if "messages" in chunk:
                for message in chunk["messages"]:
                    if message.get("role") == "assistant" and "tool_calls" in message:
                        llm_response = message
                        break
        
        if expected_tool is None:
            # Should not call any tools
            if llm_response and llm_response.get("tool_calls"):
                print(f"❌ Unexpected tool call: {llm_response['tool_calls'][0]['name']}")
            else:
                print("✅ Correctly responded without tools")
        else:
            # Should call expected tool
            if llm_response and llm_response.get("tool_calls"):
                actual_tool = llm_response["tool_calls"][0]["name"]
                if actual_tool == expected_tool:
                    print(f"✅ Correctly selected {actual_tool}")
                else:
                    print(f"❌ Expected {expected_tool}, got {actual_tool}")
            else:
                print(f"❌ Expected {expected_tool}, but no tool was called")

# Run tool selection tests
test_tool_selection()
```

### Test Suite 2: Tool Integration and Chaining

```python
def test_tool_integration():
    """Test tool integration and multi-step workflows"""
    
    integration_tests = [
        {
            "request": "Research Apple Inc. and then show me their stock data",
            "expected_tools": ["wikipedia_tool", "stock_data_tool"],
            "description": "Sequential tool usage"
        },
        {
            "request": "Get Microsoft's stock performance and create a visualization",
            "expected_tools": ["stock_data_tool", "python_repl_tool"],
            "description": "Stock data to visualization workflow"
        },
        {
            "request": "Tell me about Tesla, get their recent stock prices, and plot them",
            "expected_tools": ["wikipedia_tool", "stock_data_tool", "python_repl_tool"],
            "description": "Complete three-tool workflow"
        }
    ]
    
    for test in integration_tests:
        print(f"\n--- Integration Test: {test['description']} ---")
        print(f"Request: {test['request']}")
        print(f"Expected tools: {test['expected_tools']}")
        
        # Note: Current linear workflow has limitations for tool chaining
        # This test documents expected behavior for future conditional workflows
        
        try:
            chunks = list(graph.stream({
                "messages": [{"role": "user", "content": test['request']}]
            }))
            
            # Analyze tool usage
            tools_used = []
            for chunk in chunks:
                if "messages" in chunk:
                    for message in chunk["messages"]:
                        if message.get("role") == "assistant" and "tool_calls" in message:
                            for tool_call in message["tool_calls"]:
                                tools_used.append(tool_call["name"])
            
            print(f"Tools actually used: {tools_used}")
            
            # Current linear workflow limitation
            if len(tools_used) <= 1:
                print("ℹ️ Linear workflow limitation: Only first tool executed")
            else:
                print("✅ Multiple tools executed successfully")
                
        except Exception as e:
            print(f"❌ Integration test failed: {e}")

# Run integration tests
test_tool_integration()
```

### Test Suite 3: Error Handling and Edge Cases

```python
def test_error_handling():
    """Test tool error handling and edge cases"""
    
    error_scenarios = [
        # Invalid stock ticker
        {
            "request": "Show me stock data for INVALID ticker",
            "expected_error": "not available",
            "tool": "stock_data_tool"
        },
        
        # Excessive date range
        {
            "request": "Get AAPL stock data for 10000 days",
            "expected_error": "exceeds the data available",
            "tool": "stock_data_tool"
        },
        
        # Invalid Python code
        {
            "request": "Execute this code: print(undefined_variable)",
            "expected_error": "Failed to execute",
            "tool": "python_repl_tool"
        },
        
        # Non-existent Wikipedia topic
        {
            "request": "Tell me about XYZABC123 fake company",
            "expected_behavior": "graceful handling",
            "tool": "wikipedia_tool"
        }
    ]
    
    for scenario in error_scenarios:
        print(f"\n--- Error Test: {scenario['tool']} ---")
        print(f"Request: {scenario['request']}")
        
        try:
            chunks = list(graph.stream({
                "messages": [{"role": "user", "content": scenario['request']}]
            }))
            
            # Find tool response
            tool_response = None
            for chunk in chunks:
                if "messages" in chunk:
                    for message in chunk["messages"]:
                        if message.get("role") == "tool":
                            tool_response = message["content"]
                            break
            
            if tool_response:
                if "expected_error" in scenario and scenario["expected_error"] in tool_response:
                    print(f"✅ Error handled correctly: {scenario['expected_error']}")
                else:
                    print(f"ℹ️ Tool response: {tool_response[:100]}...")
            else:
                print("❌ No tool response received")
                
        except Exception as e:
            print(f"❌ Unexpected exception: {e}")

# Run error handling tests
test_error_handling()
```

### Test Suite 4: Performance and Resource Management

```python
import time
import sys

def test_performance():
    """Test system performance and resource usage"""
    
    performance_tests = [
        {
            "name": "Wikipedia Query",
            "request": "Tell me about Apple Inc.",
            "expected_max_time": 10.0
        },
        {
            "name": "Stock Data Query",
            "request": "Show me AAPL stock data for 5 days",
            "expected_max_time": 5.0
        },
        {
            "name": "Simple Python Execution",
            "request": "Calculate 2 + 2 using Python",
            "expected_max_time": 5.0
        }
    ]
    
    for test in performance_tests:
        print(f"\n--- Performance Test: {test['name']} ---")
        
        # Measure execution time
        start_time = time.time()
        
        try:
            chunks = list(graph.stream({
                "messages": [{"role": "user", "content": test['request']}]
            }))
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            print(f"Execution time: {execution_time:.2f} seconds")
            
            if execution_time <= test["expected_max_time"]:
                print(f"✅ Performance within limits (< {test['expected_max_time']}s)")
            else:
                print(f"⚠️ Slower than expected (> {test['expected_max_time']}s)")
            
            # Measure response size
            total_response_size = sum(
                len(str(chunk)) for chunk in chunks
            )
            print(f"Total response size: {total_response_size} characters")
            
        except Exception as e:
            print(f"❌ Performance test failed: {e}")

# Run performance tests
test_performance()
```

### Test Suite 5: State Management and Conversation Flow

```python
def test_conversation_flow():
    """Test state management across multiple interactions"""
    
    # Multi-turn conversation scenarios
    conversation_scenarios = [
        {
            "name": "Research and Follow-up",
            "turns": [
                "Tell me about Microsoft Corporation",
                "Now show me their stock performance",
                "Can you plot that data?"
            ]
        },
        {
            "name": "Context Retention",
            "turns": [
                "My favorite company is Apple",
                "Show me stock data for my favorite company",
                "Create a chart of their performance"
            ]
        }
    ]
    
    for scenario in conversation_scenarios:
        print(f"\n--- Conversation Test: {scenario['name']} ---")
        
        conversation_state = {"messages": []}
        
        for turn_num, user_input in enumerate(scenario['turns'], 1):
            print(f"\nTurn {turn_num}: {user_input}")
            
            # Add user message to conversation
            conversation_state["messages"].append({
                "role": "user", 
                "content": user_input
            })
            
            try:
                # Stream response
                chunks = list(graph.stream(conversation_state))
                
                # Update conversation state with all new messages
                for chunk in chunks:
                    if "messages" in chunk:
                        conversation_state["messages"].extend(chunk["messages"])
                
                # Show final response
                if conversation_state["messages"]:
                    last_message = conversation_state["messages"][-1]
                    if last_message.get("role") == "tool":
                        print(f"Tool result: {last_message['content'][:100]}...")
                    elif last_message.get("role") == "assistant":
                        print(f"Assistant: {last_message.get('content', 'Tool call made')}")
                
                print(f"Total messages in conversation: {len(conversation_state['messages'])}")
                
            except Exception as e:
                print(f"❌ Conversation turn failed: {e}")
                break

# Run conversation flow tests
test_conversation_flow()
```

## 🎓 Educational Insights

### Why Tool Integration Represents a Breakthrough

1. **Autonomous Tool Selection**
   - LLMs can now choose appropriate tools based on context
   - Eliminates need for explicit tool specification
   - Enables natural language interfaces to complex capabilities

2. **Unified Tool Interface**
   - Single ToolNode handles all tool types
   - Consistent error handling across tools
   - Simplified graph architecture

3. **Context-Aware Execution**
   - Tools receive full conversation context
   - Enables complex multi-step reasoning
   - Maintains state across tool interactions

### Tool Integration Design Patterns

#### Pattern 1: Description-Driven Selection
```python
@tool
def specialized_tool(param: Annotated[str, "Very specific parameter description"]):
    """Very specific use case description that guides LLM selection."""
    pass
```

#### Pattern 2: Multi-Purpose Tool Design
```python
@tool
def flexible_tool(
    action: Annotated[str, "Action to perform: 'search', 'analyze', or 'report'"],
    data: Annotated[str, "Data to process based on action"]
):
    """Flexible tool that can perform multiple related actions."""
    if action == "search":
        return search_function(data)
    elif action == "analyze":
        return analyze_function(data)
    # etc.
```

#### Pattern 3: Tool Composition
```python
@tool
def composite_tool(request: Annotated[str, "Complex request requiring multiple steps"]):
    """High-level tool that orchestrates multiple sub-tools."""
    # Can internally call other tools or services
    results = []
    if "research" in request.lower():
        results.append(wikipedia_tool.invoke({"query": extract_query(request)}))
    if "data" in request.lower():
        results.append(stock_data_tool.invoke(extract_stock_params(request)))
    return combine_results(results)
```

## 🔧 Advanced Implementation Variations

### Variation 1: Enhanced Tool Node with Logging

```python
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LoggingToolNode:
    """Enhanced ToolNode with comprehensive logging"""
    
    def __init__(self, tools):
        self.tool_node = ToolNode(tools)
        self.execution_stats = {
            "total_calls": 0,
            "tool_usage": {},
            "errors": 0,
            "avg_execution_time": 0
        }
    
    def invoke(self, state):
        """Enhanced invoke with logging and statistics"""
        start_time = time.time()
        
        # Extract tool calls from state
        tool_calls = []
        for message in state.get("messages", []):
            if message.get("role") == "assistant" and "tool_calls" in message:
                tool_calls.extend(message["tool_calls"])
        
        # Log tool calls
        for tool_call in tool_calls:
            tool_name = tool_call.get("name", "unknown")
            logger.info(f"Executing tool: {tool_name}")
            
            # Update statistics
            self.execution_stats["total_calls"] += 1
            self.execution_stats["tool_usage"][tool_name] = \
                self.execution_stats["tool_usage"].get(tool_name, 0) + 1
        
        try:
            # Execute tools
            result = self.tool_node.invoke(state)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            self._update_avg_time(execution_time)
            
            logger.info(f"Tool execution completed in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            self.execution_stats["errors"] += 1
            logger.error(f"Tool execution failed: {e}")
            raise
    
    def _update_avg_time(self, execution_time):
        """Update running average of execution time"""
        current_avg = self.execution_stats["avg_execution_time"]
        total_calls = self.execution_stats["total_calls"]
        
        new_avg = ((current_avg * (total_calls - 1)) + execution_time) / total_calls
        self.execution_stats["avg_execution_time"] = new_avg
    
    def get_statistics(self):
        """Return execution statistics"""
        return self.execution_stats

# Use enhanced tool node
enhanced_tool_node = LoggingToolNode(tools)
graph_builder.add_node("tools", enhanced_tool_node)
```

### Variation 2: Conditional Tool Routing (Preview)

```python
from langgraph.graph import END

def should_use_tools(state: State) -> str:
    """Determine if tools are needed based on LLM response"""
    messages = state.get("messages", [])
    
    if not messages:
        return "llm"
    
    last_message = messages[-1]
    
    # Check if LLM wants to use tools
    if (last_message.get("role") == "assistant" and 
        "tool_calls" in last_message and 
        last_message["tool_calls"]):
        return "tools"
    
    # Direct response - no tools needed
    return END

# Enhanced LLM node that doesn't automatically proceed to tools
def smart_llm_node(state: State):
    """LLM node that can choose to use tools or respond directly"""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# Build conditional graph (preview for future activities)
conditional_graph_builder = StateGraph(State)
conditional_graph_builder.add_node("llm", smart_llm_node)
conditional_graph_builder.add_node("tools", tool_node)

# Add conditional edges
conditional_graph_builder.add_edge(START, "llm")
conditional_graph_builder.add_conditional_edges(
    "llm", 
    should_use_tools,
    {
        "tools": "tools",
        END: END
    }
)
conditional_graph_builder.add_edge("tools", END)

# This would be compiled and used in future exercises
# conditional_graph = conditional_graph_builder.compile()
```

### Variation 3: Tool Result Enhancement and Formatting

```python
class EnhancedToolNode:
    """Tool node with result enhancement and formatting"""
    
    def __init__(self, tools):
        self.tool_node = ToolNode(tools)
    
    def invoke(self, state):
        """Enhanced invoke with result formatting"""
        # Execute standard tool node
        result = self.tool_node.invoke(state)
        
        # Enhance tool results
        if "messages" in result:
            enhanced_messages = []
            for message in result["messages"]:
                if message.get("role") == "tool":
                    enhanced_message = self._enhance_tool_result(message)
                    enhanced_messages.append(enhanced_message)
                else:
                    enhanced_messages.append(message)
            
            result["messages"] = enhanced_messages
        
        return result
    
    def _enhance_tool_result(self, tool_message):
        """Enhance tool result with formatting and metadata"""
        content = tool_message.get("content", "")
        tool_call_id = tool_message.get("tool_call_id", "")
        
        # Determine tool type from content
        if "Wikipedia summary:" in content:
            icon = "📚"
            tool_type = "Research"
        elif "stock performance data" in content:
            icon = "📈"
            tool_type = "Financial Data"
        elif "Python REPL tool" in content:
            icon = "🐍"
            tool_type = "Code Execution"
        else:
            icon = "🔧"
            tool_type = "Tool"
        
        # Format enhanced content
        enhanced_content = f"""
{icon} **{tool_type} Result**

{content}

---
*Generated at: {datetime.now().strftime('%H:%M:%S')}*
        """.strip()
        
        # Return enhanced message
        enhanced_message = tool_message.copy()
        enhanced_message["content"] = enhanced_content
        
        return enhanced_message

# Use enhanced tool node
enhanced_tool_node = EnhancedToolNode(tools)
```

## 📊 Performance Optimization Strategies

### 1. Tool Selection Optimization

```python
def optimize_tool_descriptions():
    """Guidelines for optimizing tool descriptions for better LLM selection"""
    
    optimization_tips = {
        "Specificity": "Use specific keywords that match user intent",
        "Examples": "Include concrete examples of when to use the tool", 
        "Parameters": "Clearly describe all required parameters",
        "Output": "Specify what the tool returns",
        "Constraints": "Mention any limitations or requirements"
    }
    
    # Example of optimized tool description
    @tool
    def optimized_stock_tool(
        ticker: Annotated[str, "Stock ticker symbol (e.g., AAPL, MSFT, GOOGL)"],
        days: Annotated[int, "Number of trading days (1-365, recommend 5-30 for trends)"]
    ):
        """
        Retrieve historical stock price data from CSV files.
        
        Use this when users ask for:
        - Stock prices, performance, or trading data
        - Historical market information for specific companies
        - Data to be used in financial analysis or charts
        
        Available companies: Apple (AAPL), Microsoft (MSFT), Google (GOOGL), 
        Amazon (AMZN), Tesla (TSLA), Netflix (NFLX), Meta (META).
        
        Returns: Formatted table with Date, Open, High, Low, Close, Volume data.
        """
        # Implementation here
        pass
    
    return optimization_tips
```

### 2. Tool Execution Caching

```python
import hashlib
import json
from functools import wraps

def tool_cache(max_size=100):
    """Decorator for caching tool results"""
    cache = {}
    
    def decorator(tool_func):
        @wraps(tool_func)
        def wrapper(*args, **kwargs):
            # Create cache key from arguments
            key_data = {"args": args, "kwargs": kwargs}
            cache_key = hashlib.md5(
                json.dumps(key_data, sort_keys=True, default=str).encode()
            ).hexdigest()
            
            # Check cache
            if cache_key in cache:
                print(f"🎯 Cache hit for {tool_func.name}")
                return cache[cache_key]
            
            # Execute tool
            result = tool_func(*args, **kwargs)
            
            # Cache result (with size limit)
            if len(cache) >= max_size:
                # Remove oldest entry
                oldest_key = next(iter(cache))
                del cache[oldest_key]
            
            cache[cache_key] = result
            return result
        
        return wrapper
    return decorator

# Apply caching to tools
@tool_cache(max_size=50)
@tool
def cached_wikipedia_tool(query: Annotated[str, "Wikipedia search query"]):
    """Wikipedia tool with caching for repeated queries"""
    # Same implementation as before
    pass
```

### 3. Parallel Tool Execution (Advanced)

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class ParallelToolNode:
    """Tool node that can execute multiple tools in parallel"""
    
    def __init__(self, tools, max_workers=3):
        self.tools = {tool.name: tool for tool in tools}
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    def invoke(self, state):
        """Execute tools in parallel when possible"""
        messages = state.get("messages", [])
        
        # Find tool calls
        tool_calls = []
        for message in messages:
            if message.get("role") == "assistant" and "tool_calls" in message:
                tool_calls.extend(message["tool_calls"])
        
        if not tool_calls:
            return {"messages": []}
        
        # Execute tools in parallel
        futures = []
        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})
            
            if tool_name in self.tools:
                future = self.executor.submit(
                    self._execute_single_tool,
                    tool_name, tool_args, tool_call.get("id")
                )
                futures.append(future)
        
        # Collect results
        results = []
        for future in futures:
            try:
                result = future.result(timeout=30)  # 30 second timeout
                results.append(result)
            except Exception as e:
                error_message = {
                    "role": "tool",
                    "content": f"Tool execution failed: {str(e)}"
                }
                results.append(error_message)
        
        return {"messages": results}
    
    def _execute_single_tool(self, tool_name, tool_args, tool_call_id):
        """Execute a single tool"""
        tool = self.tools[tool_name]
        
        try:
            result = tool.invoke(tool_args)
            return {
                "role": "tool",
                "content": result,
                "tool_call_id": tool_call_id
            }
        except Exception as e:
            return {
                "role": "tool", 
                "content": f"Error executing {tool_name}: {str(e)}",
                "tool_call_id": tool_call_id
            }
```

## 📝 Assessment Rubric

### Functionality (45 points)
- **Tool binding:** Correct use of `bind_tools()` and tool integration (15 pts)
- **Graph construction:** Proper node and edge setup with ToolNode (15 pts)
- **Tool execution:** Tools are called correctly and produce expected results (15 pts)

### Code Quality (30 points)
- **Tool descriptions:** Clear, informative docstrings and annotations (10 pts)
- **Error handling:** Robust tool error management (10 pts)  
- **Code organization:** Clean structure and proper imports (10 pts)

### Understanding (25 points)
- **Tool selection:** Understands how LLMs choose tools based on descriptions (15 pts)
- **Workflow patterns:** Explains linear vs conditional routing trade-offs (10 pts)

### Total: 100 points

**Grading Scale:**
- 90-100: Excellent tool integration with deep understanding of selection mechanisms
- 80-89: Good implementation with minor issues in tool binding or description quality
- 70-79: Basic functionality working, needs improvement in advanced concepts
- Below 70: Requires additional practice with tool integration patterns

## 🚀 Real-World Applications

### Enterprise Tool Orchestration

```python
# Example: Customer service automation
enterprise_tools = [
    knowledge_base_tool,  # Search company knowledge base
    ticket_system_tool,   # Create/update support tickets  
    email_tool,          # Send automated emails
    escalation_tool,     # Route to human agents
    analytics_tool       # Log interaction metrics
]

# Multi-step customer service workflow
customer_service_graph = build_conditional_graph(
    tools=enterprise_tools,
    routing_strategy="intent_based",
    fallback_to_human=True
)
```

### Research and Analysis Pipeline

```python
# Example: Investment research automation  
research_tools = [
    market_data_tool,     # Get market data
    news_sentiment_tool,  # Analyze news sentiment
    financial_analysis_tool, # Calculate metrics
    report_generation_tool,  # Create reports
    compliance_check_tool    # Verify compliance
]

# Automated investment research workflow
research_graph = build_sequential_graph(
    tools=research_tools,
    quality_gates=True,
    human_review_required=True
)
```

## 💡 Pro Tips for Instructors

1. **Tool Description Workshop**: Have students rewrite tool descriptions and test selection accuracy
2. **Error Simulation**: Intentionally break tools to demonstrate error handling
3. **Performance Comparison**: Compare linear vs direct LLM calls to show tool overhead
4. **Real Tool Integration**: Connect to actual APIs instead of CSV files for production examples
5. **Tool Design Principles**: Teach single responsibility and clear interfaces

## 🏁 Conclusion

This exercise establishes the foundation for intelligent tool orchestration in AI agent systems. Students learn:

- **Automatic Tool Selection**: How LLMs choose appropriate tools based on descriptions and context
- **Unified Tool Interface**: Using ToolNode for consistent tool execution across different tool types
- **Linear Workflow Patterns**: Building predictable, debuggable tool-enabled workflows
- **Integration Architecture**: Combining multiple specialized tools into cohesive agent systems

**Key Architectural Insights:**
- Tool descriptions are critical for proper LLM decision-making
- ToolNode provides elegant abstraction for tool execution
- Linear workflows ensure predictability but limit flexibility
- State management enables complex multi-step tool interactions

Students are now ready to build sophisticated conditional workflows and multi-agent systems with intelligent tool orchestration! 🚀🔨🧠