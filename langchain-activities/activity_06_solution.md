# 🔀 Activity 06: Conditional Routing with LangGraph - Master Solution Guide

## 📋 Activity Overview

**Topic:** Implementing conditional edges and intelligent workflow routing in LangGraph  
**Duration:** 45-60 minutes  
**Difficulty:** Advanced  
**Prerequisites:** LangGraph basics, tool integration experience, state management

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

### Step 3: Complete Conditional Graph Implementation

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode, tools_condition

# Define State
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Create graph builder
graph_builder = StateGraph(State)

# Define tools and bind to LLM
tools = [wikipedia_tool, stock_data_tool, python_repl_tool]
llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)

# Define LLM node
def llm_node(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Build graph with conditional routing
graph_builder.add_node("llm", llm_node)
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

# Add edges
graph_builder.add_edge(START, "llm")
graph_builder.add_conditional_edges(
    "llm",
    tools_condition,
    {
        "tools": "tools",
        "__end__": END
    }
)
graph_builder.add_edge("tools", "llm")

# Compile graph
graph = graph_builder.compile()
```

### Step 4: Testing Implementation

```python
# Visualize graph
graph

# Test different request types
from course_helper_functions import pretty_print_messages

test_cases = [
    "Hello, how are you?",                    # Direct response
    "Tell me about Apple Inc.",               # Wikipedia tool
    "Show me AAPL stock data for 5 days",    # Stock data tool
    "What's your favorite color?",            # Direct response
    "Plot Apple stock prices"                 # Python REPL tool
]

for test_input in test_cases:
    print(f"\n--- Testing: {test_input} ---")
    for chunk in graph.stream(
        {"messages": [{"role": "user", "content": test_input}]}
    ):
        pretty_print_messages(chunk)
```

## 🧠 Code Breakdown & Best Practices

### 1. Conditional Edge Architecture Deep Dive

```python
graph_builder.add_conditional_edges(
    "llm",              # Source node
    tools_condition,    # Condition function
    {
        "tools": "tools",    # If tools needed
        "__end__": END       # If direct response
    }
)
```

**How Conditional Edges Work:**

1. **State Evaluation**: Condition function receives current state
2. **Decision Making**: Function analyzes LLM output for tool calls
3. **Route Selection**: Returns key that maps to destination node
4. **Dynamic Flow**: Graph follows the selected path

**Built-in `tools_condition` Logic:**
```python
def tools_condition(state):
    """
    Examines the last message in state
    
    Returns:
    - "tools" if assistant message contains tool_calls
    - "__end__" if no tool_calls (direct response)
    """
    messages = state.get("messages", [])
    if not messages:
        return "__end__"
    
    last_message = messages[-1]
    if (last_message.get("role") == "assistant" and 
        "tool_calls" in last_message and 
        last_message["tool_calls"]):
        return "tools"
    
    return "__end__"
```

### 2. Cyclic Workflow Pattern Analysis

```python
START → LLM → (Decision) → Tools → LLM → (Decision) → END
                    ↓                      ↑
                    → END                  └─ (Possible loop back)
```

**Cyclic Flow Benefits:**
- **Tool Result Processing**: LLM can interpret and format tool outputs
- **Multi-step Workflows**: Enables chaining multiple tool calls
- **User-friendly Responses**: Raw tool data becomes conversational
- **Decision Points**: LLM can decide when task is complete

**Flow Examples:**

1. **Direct Response Flow:**
   ```
   "Hello" → LLM → (no tools needed) → END
   ```

2. **Single Tool Flow:**
   ```
   "About Apple" → LLM → (needs Wikipedia) → Tools → LLM → (format response) → END
   ```

3. **Multi-tool Flow:**
   ```
   "Research and plot Apple data" → LLM → Tools → LLM → (need more tools) → Tools → LLM → END
   ```

### 3. Message Flow State Management

**State Evolution in Conditional Workflow:**

```python
# Initial state
{
    "messages": [
        {"role": "user", "content": "Tell me about Apple Inc."}
    ]
}

# After first LLM decision
{
    "messages": [
        {"role": "user", "content": "Tell me about Apple Inc."},
        {"role": "assistant", "tool_calls": [
            {"name": "wikipedia_tool", "args": {"query": "Apple Inc."}}
        ]}
    ]
}

# After tool execution
{
    "messages": [
        {"role": "user", "content": "Tell me about Apple Inc."},
        {"role": "assistant", "tool_calls": [...]},
        {"role": "tool", "content": "Successfully executed:\nWikipedia summary: Apple Inc. is..."}
    ]
}

# After LLM processes tool result
{
    "messages": [
        {"role": "user", "content": "Tell me about Apple Inc."},
        {"role": "assistant", "tool_calls": [...]},
        {"role": "tool", "content": "..."},
        {"role": "assistant", "content": "Based on the information I found, Apple Inc. is..."}
    ]
}
```

### 4. Decision Logic and Route Mapping

**Route Dictionary Structure:**
```python
{
    "condition_result_1": "destination_node_1",
    "condition_result_2": "destination_node_2",
    "__end__": END  # Special END node
}
```

**Advanced Routing Patterns:**
```python
# Multiple tool categories
{
    "research_tools": "research_node",
    "analysis_tools": "analysis_node", 
    "visualization_tools": "viz_node",
    "__end__": END
}

# Priority-based routing
{
    "high_priority": "priority_tools",
    "normal": "standard_tools",
    "low_priority": "background_tools",
    "__end__": END
}
```

## 🧪 Comprehensive Testing Suite

### Test Suite 1: Routing Decision Validation

```python
def test_routing_decisions():
    """Test that conditional edges route correctly for different request types"""
    
    routing_test_cases = [
        # Format: (request, should_use_tools, expected_tool_type)
        ("Hello, how are you?", False, None),
        ("Good morning!", False, None),
        ("Thank you", False, None),
        ("What's your name?", False, None),
        
        ("Tell me about Apple Inc.", True, "wikipedia_tool"),
        ("Research Microsoft Corporation", True, "wikipedia_tool"),
        ("What is Tesla's business model?", True, "wikipedia_tool"),
        
        ("Show AAPL stock prices", True, "stock_data_tool"),
        ("Get Tesla stock data for 10 days", True, "stock_data_tool"),
        ("MSFT stock performance", True, "stock_data_tool"),
        
        ("Plot stock prices", True, "python_repl_tool"),
        ("Create a chart", True, "python_repl_tool"),
        ("Calculate average of [1,2,3]", True, "python_repl_tool"),
    ]
    
    correct_routing = 0
    total_tests = len(routing_test_cases)
    
    for request, should_use_tools, expected_tool in routing_test_cases:
        print(f"\n--- Routing Test: {request} ---")
        
        # Execute workflow
        chunks = list(graph.stream({
            "messages": [{"role": "user", "content": request}]
        }))
        
        # Analyze routing decision
        tools_used = []
        direct_response = False
        
        for chunk in chunks:
            if "messages" in chunk:
                for message in chunk["messages"]:
                    if message.get("role") == "assistant":
                        if "tool_calls" in message and message["tool_calls"]:
                            for tool_call in message["tool_calls"]:
                                tools_used.append(tool_call["name"])
                        elif "content" in message and not message.get("tool_calls"):
                            direct_response = True
        
        # Validate routing
        actual_tool_usage = len(tools_used) > 0
        
        if should_use_tools == actual_tool_usage:
            print(f"✅ Routing correct - Tools used: {actual_tool_usage}")
            if should_use_tools and expected_tool:
                if expected_tool in tools_used:
                    print(f"✅ Correct tool selected: {expected_tool}")
                    correct_routing += 1
                else:
                    print(f"❌ Wrong tool selected. Expected: {expected_tool}, Got: {tools_used}")
            else:
                correct_routing += 1
        else:
            print(f"❌ Routing incorrect - Expected tools: {should_use_tools}, Actual: {actual_tool_usage}")
    
    print(f"\n=== Routing Test Summary ===")
    print(f"Correct routing: {correct_routing}/{total_tests} ({(correct_routing/total_tests)*100:.1f}%)")
    
    return correct_routing / total_tests

# Run routing validation tests
routing_accuracy = test_routing_decisions()
```

### Test Suite 2: Response Quality and Processing

```python
def test_response_quality():
    """Test response quality for both direct and tool-assisted responses"""
    
    response_scenarios = [
        {
            "request": "Hello, how are you?",
            "type": "direct",
            "min_length": 20,
            "should_be_conversational": True
        },
        {
            "request": "Tell me about Apple Inc.",
            "type": "tool_assisted", 
            "min_length": 100,
            "should_contain": ["Apple", "company", "technology"]
        },
        {
            "request": "Show me AAPL stock data for 5 days",
            "type": "tool_assisted",
            "min_length": 50,
            "should_contain": ["AAPL", "stock", "data"]
        }
    ]
    
    for scenario in response_scenarios:
        print(f"\n--- Response Quality Test: {scenario['request']} ---")
        
        # Execute workflow
        chunks = list(graph.stream({
            "messages": [{"role": "user", "content": scenario['request']}]
        }))
        
        # Extract final response
        final_response = None
        tool_responses = []
        
        for chunk in chunks:
            if "messages" in chunk:
                for message in chunk["messages"]:
                    if message.get("role") == "assistant" and "content" in message:
                        # Skip tool call messages, only get final content
                        if not message.get("tool_calls"):
                            final_response = message["content"]
                    elif message.get("role") == "tool":
                        tool_responses.append(message["content"])
        
        # Analyze response quality
        if final_response:
            response_length = len(final_response)
            print(f"Response type: {scenario['type']}")
            print(f"Response length: {response_length} characters")
            
            # Length validation
            if response_length >= scenario["min_length"]:
                print("✅ Response length adequate")
            else:
                print(f"❌ Response too short (minimum {scenario['min_length']})")
            
            # Content validation
            if "should_contain" in scenario:
                missing_terms = []
                for term in scenario["should_contain"]:
                    if term.lower() not in final_response.lower():
                        missing_terms.append(term)
                
                if not missing_terms:
                    print("✅ Response contains expected terms")
                else:
                    print(f"❌ Missing terms: {missing_terms}")
            
            # Conversational quality for direct responses
            if scenario.get("should_be_conversational"):
                conversational_indicators = ["I", "you", "!", "?", "Hello", "Hi"]
                if any(indicator in final_response for indicator in conversational_indicators):
                    print("✅ Response is conversational")
                else:
                    print("⚠️ Response may lack conversational tone")
            
            print(f"Response preview: {final_response[:150]}...")
            
        else:
            print("❌ No final response found")

# Run response quality tests
test_response_quality()
```

### Test Suite 3: Multi-Step Workflow Testing

```python
def test_multi_step_workflows():
    """Test complex workflows that may require multiple tool interactions"""
    
    multi_step_scenarios = [
        {
            "request": "Research Apple Inc. and then show me their stock performance",
            "expected_tools": ["wikipedia_tool", "stock_data_tool"],
            "description": "Research followed by data retrieval"
        },
        {
            "request": "Tell me about Tesla and create a chart of their stock prices",
            "expected_tools": ["wikipedia_tool", "python_repl_tool"],
            "description": "Research followed by visualization"
        },
        {
            "request": "Get Microsoft stock data and plot it",
            "expected_tools": ["stock_data_tool", "python_repl_tool"],
            "description": "Data retrieval followed by visualization"
        }
    ]
    
    for scenario in multi_step_scenarios:
        print(f"\n--- Multi-Step Test: {scenario['description']} ---")
        print(f"Request: {scenario['request']}")
        
        # Execute workflow
        chunks = list(graph.stream({
            "messages": [{"role": "user", "content": scenario['request']}]
        }))
        
        # Track tool execution sequence
        tool_sequence = []
        llm_iterations = 0
        
        for chunk in chunks:
            if "messages" in chunk:
                for message in chunk["messages"]:
                    if message.get("role") == "assistant":
                        if "tool_calls" in message and message["tool_calls"]:
                            llm_iterations += 1
                            for tool_call in message["tool_calls"]:
                                tool_sequence.append(tool_call["name"])
        
        print(f"Tool sequence: {tool_sequence}")
        print(f"LLM decision iterations: {llm_iterations}")
        
        # Analyze workflow completeness
        expected_tools_used = all(tool in tool_sequence for tool in scenario["expected_tools"])
        
        if expected_tools_used:
            print("✅ All expected tools were used")
        else:
            missing_tools = [tool for tool in scenario["expected_tools"] if tool not in tool_sequence]
            print(f"❌ Missing tools: {missing_tools}")
        
        # Note: Current implementation limitations
        if len(set(tool_sequence)) == 1:
            print("ℹ️ Note: Current workflow may not support true multi-step coordination")
        elif len(set(tool_sequence)) > 1:
            print("✅ Multi-tool workflow executed")

# Run multi-step workflow tests
test_multi_step_workflows()
```

### Test Suite 4: Error Handling and Recovery

```python
def test_error_handling():
    """Test error handling in conditional workflows"""
    
    error_scenarios = [
        {
            "request": "Show me stock data for INVALIDTICKER",
            "expected_error_type": "invalid_ticker",
            "should_recover": True
        },
        {
            "request": "Get AAPL data for 10000 days",
            "expected_error_type": "date_range_exceeded",
            "should_recover": True
        },
        {
            "request": "Execute this code: print(undefined_variable)",
            "expected_error_type": "code_execution",
            "should_recover": True
        },
        {
            "request": "Research XYZFAKECOMPANY123",
            "expected_error_type": "no_wikipedia_results", 
            "should_recover": True
        }
    ]
    
    for scenario in error_scenarios:
        print(f"\n--- Error Handling Test: {scenario['expected_error_type']} ---")
        print(f"Request: {scenario['request']}")
        
        try:
            # Execute workflow
            chunks = list(graph.stream({
                "messages": [{"role": "user", "content": scenario['request']}]
            }))
            
            # Analyze error handling
            error_occurred = False
            error_handled_gracefully = False
            final_response = None
            
            for chunk in chunks:
                if "messages" in chunk:
                    for message in chunk["messages"]:
                        if message.get("role") == "tool":
                            content = message.get("content", "")
                            if any(error_indicator in content.lower() for error_indicator in 
                                  ["error", "failed", "sorry", "not available", "exceeds"]):
                                error_occurred = True
                        elif message.get("role") == "assistant" and "content" in message:
                            final_response = message["content"]
                            # Check if LLM provided helpful response despite tool error
                            if final_response and len(final_response) > 20:
                                error_handled_gracefully = True
            
            print(f"Error detected: {error_occurred}")
            print(f"Graceful handling: {error_handled_gracefully}")
            
            if error_occurred and error_handled_gracefully:
                print("✅ Error handled gracefully")
            elif error_occurred and not error_handled_gracefully:
                print("⚠️ Error occurred but handling could be improved")
            elif not error_occurred:
                print("ℹ️ No error detected - may indicate robust input handling")
            
            if final_response:
                print(f"Final response: {final_response[:100]}...")
                
        except Exception as e:
            print(f"❌ Unhandled exception: {e}")
            print("❌ Workflow failed catastrophically")

# Run error handling tests
test_error_handling()
```

### Test Suite 5: Performance and Efficiency Analysis

```python
import time

def test_performance_efficiency():
    """Test performance differences between conditional and linear workflows"""
    
    performance_scenarios = [
        {
            "request": "Hello there!",
            "type": "direct_response",
            "expected_max_time": 3.0
        },
        {
            "request": "Tell me about Apple Inc.",
            "type": "tool_assisted",
            "expected_max_time": 15.0
        },
        {
            "request": "What's your favorite color?",
            "type": "direct_response", 
            "expected_max_time": 3.0
        }
    ]
    
    performance_results = []
    
    for scenario in performance_scenarios:
        print(f"\n--- Performance Test: {scenario['type']} ---")
        print(f"Request: {scenario['request']}")
        
        # Measure execution time
        start_time = time.time()
        
        try:
            chunks = list(graph.stream({
                "messages": [{"role": "user", "content": scenario['request']}]
            }))
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Analyze performance
            print(f"Execution time: {execution_time:.2f} seconds")
            
            if execution_time <= scenario["expected_max_time"]:
                print(f"✅ Performance within expected range (≤{scenario['expected_max_time']}s)")
                performance_result = "pass"
            else:
                print(f"⚠️ Slower than expected (>{scenario['expected_max_time']}s)")
                performance_result = "slow"
            
            # Count tool executions
            tool_executions = 0
            for chunk in chunks:
                if "messages" in chunk:
                    for message in chunk["messages"]:
                        if message.get("role") == "tool":
                            tool_executions += 1
            
            print(f"Tool executions: {tool_executions}")
            
            # Efficiency analysis
            if scenario["type"] == "direct_response" and tool_executions == 0:
                print("✅ Efficient - no unnecessary tool calls")
                efficiency = "excellent"
            elif scenario["type"] == "tool_assisted" and tool_executions > 0:
                print("✅ Appropriate - tools used as needed")
                efficiency = "good"
            else:
                print("⚠️ Efficiency issue - check routing logic")
                efficiency = "poor"
            
            performance_results.append({
                "scenario": scenario["request"],
                "time": execution_time,
                "performance": performance_result,
                "efficiency": efficiency,
                "tool_calls": tool_executions
            })
            
        except Exception as e:
            print(f"❌ Performance test failed: {e}")
            performance_results.append({
                "scenario": scenario["request"],
                "time": float('inf'),
                "performance": "error",
                "efficiency": "error",
                "tool_calls": -1
            })
    
    # Summary
    print(f"\n=== Performance Summary ===")
    total_tests = len(performance_results)
    passed_tests = sum(1 for r in performance_results if r["performance"] == "pass")
    efficient_tests = sum(1 for r in performance_results if r["efficiency"] in ["excellent", "good"])
    
    print(f"Performance tests passed: {passed_tests}/{total_tests}")
    print(f"Efficient routing: {efficient_tests}/{total_tests}")
    
    return performance_results

# Run performance tests
performance_data = test_performance_efficiency()
```

## 🎓 Educational Insights

### Why Conditional Routing is a Game-Changer

1. **Intelligent Resource Management**
   - Tools only execute when genuinely needed
   - Computational resources are optimized
   - Response times improve for simple queries

2. **Enhanced User Experience**
   - Natural conversation for casual interactions
   - Tool-assisted responses for complex queries
   - Contextually appropriate communication style

3. **Scalable Architecture**
   - Easy to add new routing conditions
   - Modular decision-making logic
   - Supports complex multi-agent scenarios

4. **Production-Ready Patterns**
   - Error handling at decision points
   - Performance monitoring capabilities
   - Flexible routing for different use cases

### Advanced Conditional Routing Patterns

#### Pattern 1: Multi-Tier Decision Making
```python
def hierarchical_routing_condition(state):
    """Multi-level routing based on complexity analysis"""
    content = get_last_user_message(state)
    
    # Tier 1: Simple responses
    if is_greeting_or_casual(content):
        return "__end__"
    
    # Tier 2: Single tool needed
    elif requires_single_tool(content):
        return determine_single_tool(content)
    
    # Tier 3: Complex workflow
    elif requires_workflow(content):
        return "workflow_coordinator"
    
    # Default: escalate to human
    else:
        return "human_handoff"
```

#### Pattern 2: Context-Aware Routing
```python
def context_aware_routing(state):
    """Route based on conversation history and context"""
    messages = state.get("messages", [])
    
    # Analyze conversation context
    recent_tools_used = get_recent_tool_usage(messages)
    user_expertise_level = infer_user_expertise(messages)
    conversation_topic = extract_topic(messages)
    
    # Make routing decision based on context
    if conversation_topic == "financial" and "stock_data_tool" not in recent_tools_used:
        return "financial_tools"
    elif user_expertise_level == "beginner":
        return "guided_tools"
    else:
        return "advanced_tools"
```

#### Pattern 3: Load-Balancing Routing
```python
class LoadBalancingRouter:
    """Route to different tool nodes based on current load"""
    
    def __init__(self):
        self.node_loads = {
            "tools_primary": 0,
            "tools_secondary": 0,
            "tools_backup": 0
        }
    
    def route_with_load_balancing(self, state):
        """Route to least loaded tool node"""
        if not needs_tools(state):
            return "__end__"
        
        # Find least loaded node
        min_load_node = min(self.node_loads.items(), key=lambda x: x[1])[0]
        
        # Update load counter
        self.node_loads[min_load_node] += 1
        
        return min_load_node
```

## 🔧 Advanced Implementation Variations

### Variation 1: Enhanced Conditional Logic with Confidence Scoring

```python
import re
from typing import Dict, Any

class AdvancedConditionalRouter:
    """Advanced routing with confidence scoring and fallback strategies"""
    
    def __init__(self):
        self.routing_patterns = {
            'wikipedia': {
                'keywords': ['tell me about', 'what is', 'who is', 'research', 'information about'],
                'weight': 1.0
            },
            'stock_data': {
                'keywords': ['stock', 'price', 'share', 'ticker', 'financial data', 'market'],
                'weight': 1.0
            },
            'python_repl': {
                'keywords': ['plot', 'chart', 'graph', 'visualize', 'calculate', 'code'],
                'weight': 1.0
            }
        }
        self.confidence_threshold = 0.6
    
    def enhanced_routing_condition(self, state: Dict[str, Any]) -> str:
        """Advanced routing with confidence scoring"""
        messages = state.get("messages", [])
        if not messages:
            return "__end__"
        
        last_message = messages[-1]
        
        # Check if this is a tool call response
        if (last_message.get("role") == "assistant" and 
            "tool_calls" in last_message and 
            last_message["tool_calls"]):
            return "tools"
        
        # For user messages, analyze intent with confidence
        if last_message.get("role") == "user":
            content = last_message.get("content", "").lower()
            
            # Calculate confidence scores for each tool category
            tool_scores = {}
            for tool_name, pattern_info in self.routing_patterns.items():
                score = self._calculate_confidence(content, pattern_info)
                tool_scores[tool_name] = score
            
            # Find highest confidence tool
            max_score = max(tool_scores.values())
            best_tool = max(tool_scores.items(), key=lambda x: x[1])[0]
            
            # Route based on confidence threshold
            if max_score >= self.confidence_threshold:
                return f"specialized_{best_tool}_tools"
            elif max_score > 0.3:  # Low confidence - use general tools
                return "tools"
            else:  # Very low confidence - direct response
                return "__end__"
        
        return "__end__"
    
    def _calculate_confidence(self, content: str, pattern_info: Dict) -> float:
        """Calculate confidence score for routing decision"""
        keywords = pattern_info['keywords']
        weight = pattern_info['weight']
        
        # Count keyword matches
        matches = sum(1 for keyword in keywords if keyword in content)
        
        # Calculate base confidence
        base_confidence = matches / len(keywords)
        
        # Apply weight
        confidence = base_confidence * weight
        
        return min(confidence, 1.0)  # Cap at 1.0

# Usage example
advanced_router = AdvancedConditionalRouter()

# Build graph with advanced routing
graph_builder.add_conditional_edges(
    "llm",
    advanced_router.enhanced_routing_condition,
    {
        "specialized_wikipedia_tools": "wikipedia_only_node",
        "specialized_stock_data_tools": "stock_only_node", 
        "specialized_python_repl_tools": "python_only_node",
        "tools": "general_tools_node",
        "__end__": END
    }
)
```

### Variation 2: Dynamic Routing with Machine Learning

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np

class MLBasedRouter:
    """Machine learning-based routing for more sophisticated decisions"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.classifier = MultinomialNB()
        self.trained = False
        
        # Training data examples
        self.training_data = [
            ("tell me about apple company", "wikipedia"),
            ("what is tesla", "wikipedia"),
            ("research microsoft", "wikipedia"),
            ("show apple stock price", "stock_data"),
            ("get tesla financial data", "stock_data"),
            ("plot apple stock chart", "python_repl"),
            ("create visualization", "python_repl"),
            ("hello how are you", "direct"),
            ("thank you", "direct"),
            ("what's your name", "direct"),
        ]
    
    def train_router(self):
        """Train the routing classifier"""
        texts = [example[0] for example in self.training_data]
        labels = [example[1] for example in self.training_data]
        
        # Vectorize training data
        X = self.vectorizer.fit_transform(texts)
        
        # Train classifier
        self.classifier.fit(X, labels)
        self.trained = True
    
    def ml_routing_condition(self, state: Dict[str, Any]) -> str:
        """ML-based routing condition"""
        if not self.trained:
            self.train_router()
        
        messages = state.get("messages", [])
        if not messages:
            return "__end__"
        
        last_message = messages[-1]
        
        # Check for tool call responses
        if (last_message.get("role") == "assistant" and 
            "tool_calls" in last_message):
            return "tools"
        
        # For user messages, use ML prediction
        if last_message.get("role") == "user":
            content = last_message.get("content", "")
            
            # Vectorize input
            X = self.vectorizer.transform([content])
            
            # Get prediction and confidence
            prediction = self.classifier.predict(X)[0]
            confidence = max(self.classifier.predict_proba(X)[0])
            
            # Route based on prediction and confidence
            if confidence > 0.7:
                if prediction == "direct":
                    return "__end__"
                else:
                    return "tools"
            else:
                # Low confidence - use fallback logic
                return "tools" if any(keyword in content.lower() 
                                    for keyword in ["show", "tell", "get", "plot"]) else "__end__"
        
        return "__end__"

# Usage
ml_router = MLBasedRouter()
```

### Variation 3: Stateful Routing with Memory

```python
class StatefulRouter:
    """Router that maintains state across conversations"""
    
    def __init__(self):
        self.conversation_memory = {}
        self.user_preferences = {}
        self.tool_usage_history = []
    
    def stateful_routing_condition(self, state: Dict[str, Any]) -> str:
        """Routing with conversation memory"""
        messages = state.get("messages", [])
        if not messages:
            return "__end__"
        
        # Extract conversation ID (in real implementation, from session)
        conversation_id = "default"  # Would be dynamic
        
        # Initialize conversation memory
        if conversation_id not in self.conversation_memory:
            self.conversation_memory[conversation_id] = {
                "topics": [],
                "tool_preferences": {},
                "expertise_level": "unknown"
            }
        
        conv_memory = self.conversation_memory[conversation_id]
        last_message = messages[-1]
        
        # Handle tool call responses
        if (last_message.get("role") == "assistant" and 
            "tool_calls" in last_message):
            # Record tool usage
            for tool_call in last_message["tool_calls"]:
                tool_name = tool_call["name"]
                self.tool_usage_history.append(tool_name)
                conv_memory["tool_preferences"][tool_name] = \
                    conv_memory["tool_preferences"].get(tool_name, 0) + 1
            
            return "tools"
        
        # For user messages, use memory-enhanced routing
        if last_message.get("role") == "user":
            content = last_message.get("content", "")
            
            # Update conversation memory
            self._update_conversation_memory(conv_memory, content)
            
            # Make routing decision based on memory
            routing_decision = self._memory_based_routing(conv_memory, content)
            
            return routing_decision
        
        return "__end__"
    
    def _update_conversation_memory(self, conv_memory: Dict, content: str):
        """Update conversation memory with new message"""
        # Extract topics
        if "apple" in content.lower():
            conv_memory["topics"].append("apple")
        elif "tesla" in content.lower():
            conv_memory["topics"].append("tesla")
        
        # Infer expertise level
        if any(term in content.lower() for term in ["api", "technical", "advanced"]):
            conv_memory["expertise_level"] = "advanced"
        elif any(term in content.lower() for term in ["beginner", "help", "how"]):
            conv_memory["expertise_level"] = "beginner"
    
    def _memory_based_routing(self, conv_memory: Dict, content: str) -> str:
        """Make routing decision based on conversation memory"""
        # If user frequently uses certain tools, prefer them
        preferred_tool = max(conv_memory["tool_preferences"].items(), 
                           key=lambda x: x[1], default=(None, 0))[0]
        
        # Route based on content and preferences
        if "stock" in content.lower() and preferred_tool == "stock_data_tool":
            return "preferred_stock_tools"
        elif preferred_tool and self._content_matches_tool(content, preferred_tool):
            return "tools"
        else:
            # Standard routing logic
            return "tools" if self._needs_tools(content) else "__end__"
    
    def _content_matches_tool(self, content: str, tool_name: str) -> bool:
        """Check if content matches tool capabilities"""
        tool_keywords = {
            "wikipedia_tool": ["tell", "about", "what", "research"],
            "stock_data_tool": ["stock", "price", "financial"],
            "python_repl_tool": ["plot", "calculate", "code"]
        }
        
        keywords = tool_keywords.get(tool_name, [])
        return any(keyword in content.lower() for keyword in keywords)
    
    def _needs_tools(self, content: str) -> bool:
        """Determine if content needs tools"""
        tool_indicators = ["show", "tell", "get", "find", "plot", "calculate"]
        return any(indicator in content.lower() for indicator in tool_indicators)

# Usage
stateful_router = StatefulRouter()
```

## 📊 Performance Analysis and Optimization

### Routing Decision Metrics

```python
class RoutingAnalyzer:
    """Analyze and optimize routing decisions"""
    
    def __init__(self):
        self.routing_stats = {
            "total_requests": 0,
            "direct_responses": 0,
            "tool_assisted": 0,
            "routing_accuracy": [],
            "response_times": {"direct": [], "tool_assisted": []}
        }
    
    def analyze_routing_performance(self, test_cases: List[Dict]) -> Dict:
        """Comprehensive routing performance analysis"""
        results = {
            "efficiency_score": 0.0,
            "accuracy_score": 0.0,
            "performance_metrics": {},
            "optimization_suggestions": []
        }
        
        for test_case in test_cases:
            # Execute and measure
            start_time = time.time()
            chunks = list(graph.stream({"messages": [{"role": "user", "content": test_case["request"]}]}))
            execution_time = time.time() - start_time
            
            # Analyze routing
            tools_used = self._extract_tools_used(chunks)
            route_taken = "tool_assisted" if tools_used else "direct"
            
            # Update stats
            self.routing_stats["total_requests"] += 1
            self.routing_stats[route_taken] += 1
            self.routing_stats["response_times"][route_taken].append(execution_time)
            
            # Check accuracy
            expected_route = test_case.get("expected_route", "unknown")
            if expected_route != "unknown":
                accuracy = 1.0 if route_taken == expected_route else 0.0
                self.routing_stats["routing_accuracy"].append(accuracy)
        
        # Calculate scores
        if self.routing_stats["routing_accuracy"]:
            results["accuracy_score"] = np.mean(self.routing_stats["routing_accuracy"])
        
        # Efficiency analysis
        direct_avg = np.mean(self.routing_stats["response_times"]["direct"]) if self.routing_stats["response_times"]["direct"] else 0
        tool_avg = np.mean(self.routing_stats["response_times"]["tool_assisted"]) if self.routing_stats["response_times"]["tool_assisted"] else 0
        
        results["performance_metrics"] = {
            "avg_direct_response_time": direct_avg,
            "avg_tool_assisted_time": tool_avg,
            "efficiency_ratio": direct_avg / tool_avg if tool_avg > 0 else float('inf')
        }
        
        # Generate optimization suggestions
        results["optimization_suggestions"] = self._generate_optimization_suggestions()
        
        return results
    
    def _extract_tools_used(self, chunks: List[Dict]) -> List[str]:
        """Extract tools used from execution chunks"""
        tools = []
        for chunk in chunks:
            if "messages" in chunk:
                for message in chunk["messages"]:
                    if message.get("role") == "assistant" and "tool_calls" in message:
                        for tool_call in message["tool_calls"]:
                            tools.append(tool_call["name"])
        return tools
    
    def _generate_optimization_suggestions(self) -> List[str]:
        """Generate suggestions for routing optimization"""
        suggestions = []
        
        # Efficiency suggestions
        direct_ratio = self.routing_stats["direct_responses"] / self.routing_stats["total_requests"]
        if direct_ratio < 0.3:
            suggestions.append("Consider improving direct response routing for simple queries")
        
        # Accuracy suggestions  
        if self.routing_stats["routing_accuracy"]:
            accuracy = np.mean(self.routing_stats["routing_accuracy"])
            if accuracy < 0.8:
                suggestions.append("Routing accuracy below 80% - review condition logic")
        
        return suggestions

# Usage
analyzer = RoutingAnalyzer()
```

## 📝 Assessment Rubric

### Functionality (45 points)
- **Conditional edge implementation:** Correct use of `add_conditional_edges()` (15 pts)
- **Routing logic:** Proper implementation with `tools_condition` (15 pts)
- **Graph structure:** Complete graph with all nodes and edges (15 pts)

### Code Quality (30 points)
- **Route mapping:** Correct dictionary format for conditional routing (10 pts)
- **Error handling:** Robust handling of routing edge cases (10 pts)
- **Code organization:** Clean structure and proper imports (10 pts)

### Understanding (25 points)
- **Concept explanation:** Can explain conditional vs linear workflows (15 pts)
- **Performance implications:** Understands efficiency benefits of conditional routing (10 pts)

### Total: 100 points

**Grading Scale:**
- 90-100: Excellent understanding of conditional routing with optimization insights
- 80-89: Good implementation with minor issues in routing logic or efficiency
- 70-79: Basic conditional routing working, needs improvement in advanced concepts
- Below 70: Requires additional practice with conditional edge patterns

## 🚀 Real-World Applications

### Enterprise Conditional Routing Examples

#### Customer Service Automation
```python
def customer_service_routing(state):
    """Route customer queries based on urgency and complexity"""
    content = get_last_user_message(state)
    
    if contains_urgency_indicators(content):
        return "priority_support"
    elif requires_technical_expertise(content):
        return "technical_support_tools"
    elif is_billing_related(content):
        return "billing_tools"
    else:
        return "general_support_tools"
```

#### Content Moderation Pipeline
```python
def content_moderation_routing(state):
    """Route content through appropriate moderation tools"""
    content = extract_user_content(state)
    
    risk_score = calculate_risk_score(content)
    
    if risk_score > 0.8:
        return "human_review"
    elif risk_score > 0.5:
        return "advanced_ai_moderation"
    elif contains_personal_info(content):
        return "privacy_check_tools"
    else:
        return "standard_processing"
```

#### Financial Services Routing
```python
def financial_services_routing(state):
    """Route financial queries based on complexity and compliance"""
    query = get_user_query(state)
    
    if is_investment_advice(query):
        return "compliance_review"
    elif requires_calculation(query):
        return "financial_calculation_tools"
    elif is_account_inquiry(query):
        return "account_management_tools"
    else:
        return "general_financial_info"
```

## 💡 Pro Tips for Instructors

1. **Visual Learning**: Show graph visualizations before and after adding conditional edges
2. **Performance Comparison**: Demonstrate efficiency gains with timing comparisons
3. **Error Scenarios**: Test routing with edge cases to show robustness
4. **Real Examples**: Use business scenarios where conditional routing adds value
5. **Debugging Practice**: Show how to trace routing decisions through state analysis

## 🏁 Conclusion

This activity introduces intelligent workflow routing that adapts to user needs and optimizes resource usage. Students learn:

- **Conditional Edge Implementation**: Using `add_conditional_edges()` for dynamic routing
- **Decision Logic**: How `tools_condition` analyzes LLM output for routing decisions
- **Cyclic Workflows**: Building flows where LLM can process tool results
- **Performance Optimization**: Creating efficient pathways for different request types
- **Production Patterns**: Scalable routing architectures for enterprise applications

**Key Architectural Benefits:**
- Efficiency gains through intelligent resource allocation
- Better user experience with contextually appropriate responses
- Scalable decision-making frameworks
- Foundation for multi-agent and specialized workflows

Students are now equipped to build sophisticated, production-ready AI agents with intelligent routing capabilities! 🚀🔀🧠