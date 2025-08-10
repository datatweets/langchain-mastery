# 🔀 Activity 06: Conditional Routing with LangGraph - Student Practice Guide

## 🎯 Learning Objectives

By the end of this activity, you will:
- Implement conditional edges for intelligent workflow routing
- Learn to use the built-in `tools_condition` function for tool decision-making
- Master the `add_conditional_edges()` method for dynamic flow control
- Understand how to build graphs that adapt their behavior based on LLM decisions
- Create efficient workflows that only use tools when necessary

## 📚 Background Context

**The Problem with Linear Workflows**
In Activity 05, you built a linear workflow:
```
START → LLM → Tools (always) → END
```

**Issues with Linear Approach:**
- ❌ **Inefficient**: Every request goes through tools, even simple greetings
- ❌ **Slower**: Unnecessary tool calls add latency
- ❌ **Rigid**: Cannot adapt to different types of requests
- ❌ **Resource Waste**: Tools execute even when not needed

**The Conditional Solution:**
```
START → LLM → Decision Point → Tools (if needed) → END
                           → Direct Response (if no tools needed) → END
```

**Benefits of Conditional Routing:**
- ✅ **Efficient**: Only uses tools when actually needed
- ✅ **Faster**: Direct responses for simple queries
- ✅ **Adaptive**: Different paths for different request types
- ✅ **Smart**: LLM decides the best approach

## 🧠 Conditional Edge Concepts

### How Conditional Routing Works

**1. LLM Analysis Phase:**
```python
User: "Hello, how are you?"
LLM Thinking: "This is a greeting - no tools needed"
Decision: Direct response
```

**2. Tool Decision Phase:**
```python
User: "Tell me about Apple Inc."
LLM Thinking: "This needs factual information - Wikipedia tool required"
Decision: Route to tools
```

**3. Post-Tool Processing:**
```python
After tool execution:
LLM Thinking: "Tool provided data - now I'll format a user-friendly response"
Decision: End workflow
```

### Conditional Edge Architecture

**Key Components:**
- **Condition Function**: Evaluates LLM output to determine next step
- **Route Mapping**: Dictionary mapping conditions to destination nodes
- **Dynamic Flow**: Different execution paths based on runtime decisions

**Built-in `tools_condition` Function:**
```python
def tools_condition(state):
    """
    Returns "tools" if LLM wants to call tools
    Returns "__end__" if LLM wants to respond directly
    """
    # Checks if last message contains tool_calls
    # Routes accordingly
```

## 🔧 Setup Instructions

### Step 1: Install Required Libraries
```bash
pip install --quiet wikipedia==1.4.0 langchain-core==0.3.59 langgraph==0.5.3 langchain-openai==0.3.16 langchain-experimental==0.3.4
```

### Step 2: Import Dependencies and Define Tools
```python
# Tools are pre-defined (same as Exercise 1.7)
# Focus will be on conditional routing implementation
```

## 🏗️ Building Conditional Workflows

### Step 3: Set Up Basic Graph Components

**Your task:** Complete the basic graph setup. You need to write about **65%** of the implementation.

```python
# TODO: Import required modules
from typing import ________
from typing_extensions import ________
from langgraph.graph import ________, ________, ________
from langgraph.graph.message import ________
from langchain_openai import ________
from langgraph.prebuilt import ________, ________

# TODO: Step 3a - Define the State class
class ________(TypedDict):
    messages: Annotated[________, ________]

# TODO: Step 3b - Create graph builder
graph_builder = ________(________)

# TODO: Step 3c - Define tools list
tools = [________, ________, ________]

# TODO: Step 3d - Create and bind LLM
llm = ________(model="________")
llm_with_tools = llm.________(________)
```

<details>
<summary>🔍 Step 3 Hints</summary>

**Step 3a:** `State(TypedDict)` with `messages: Annotated[list, add_messages]`
**Step 3b:** `StateGraph(State)`
**Step 3c:** Include all three tools: wikipedia_tool, stock_data_tool, python_repl_tool
**Step 3d:** `ChatOpenAI(model="gpt-4o-mini")` and `bind_tools(tools)`
</details>

### Step 4: Define the LLM Node

**Your task:** Create the LLM node that will make conditional decisions.

```python
# TODO: Step 4 - Define the llm_node function
# Hint: Use llm_with_tools to invoke with state messages
def llm_node(state: State):
    return {"messages": [________.________(state["messages"])]}
```

<details>
<summary>💡 Step 4 Hint</summary>

Use `llm_with_tools.invoke(state["messages"])` to get LLM responses that may include tool calls.
</details>

### Step 5: Build the Conditional Graph Structure

**Your task:** Create the graph with conditional routing. This is the core of the exercise!

```python
# TODO: Step 5a - Add the LLM node
graph_builder.________(________, ________)

# TODO: Step 5b - Create and add tool node
tool_node = ________(tools=________)
graph_builder.________(________, ________)

# TODO: Step 5c - Add edge from START to LLM
graph_builder.________(________, ________)

# TODO: Step 5d - Add conditional edges from LLM
# Hint: Use add_conditional_edges with tools_condition and routing dictionary
graph_builder.________(
    ________,  # Source node (llm)
    ________,  # Condition function (tools_condition) 
    {
        "tools": ________,     # If tools needed, go to tools node
        "__end__": ________    # If no tools needed, go to END
    }
)

# TODO: Step 5e - Add edge from tools back to LLM
# Hint: After tools execute, LLM should process results
graph_builder.________(________, ________)

# TODO: Step 5f - Compile the graph
graph = graph_builder.________()
```

<details>
<summary>🔍 Step 5 Hints</summary>

**Step 5a:** `graph_builder.add_node("llm", llm_node)`
**Step 5b:** `ToolNode(tools=tools)` and `add_node("tools", tool_node)`
**Step 5c:** `graph_builder.add_edge(START, "llm")`
**Step 5d:** Source: `"llm"`, Condition: `tools_condition`, Routes: `"tools"` and `END`
**Step 5e:** `graph_builder.add_edge("tools", "llm")`
**Step 5f:** `graph_builder.compile()`
</details>

### Step 6: Visualize and Test

```python
# TODO: Step 6a - Display graph visualization
________

# TODO: Step 6b - Test different request types
from course_helper_functions import pretty_print_messages

# Test cases to try:
test_cases = [
    "Hello, how are you?",                    # Should not use tools
    "Tell me about Apple Inc.",               # Should use Wikipedia
    "Show me AAPL stock data for 5 days",    # Should use Stock Data
    "What's your favorite color?",            # Should not use tools
]

for test_input in test_cases:
    print(f"\n--- Testing: {test_input} ---")
    for chunk in graph.stream(
        {"messages": [{"role": "user", "content": test_input}]}
    ):
        pretty_print_messages(chunk)
```

## ✅ Expected Behavior

Your conditional workflow should demonstrate:

### Intelligent Routing

**Direct Response Path:**
```
Input: "Hello, how are you?"
Flow: START → LLM → (no tools needed) → END
Output: Direct conversational response
```

**Tool-Assisted Path:**
```
Input: "Tell me about Apple Inc."
Flow: START → LLM → (tools needed) → Tools → LLM → END
Output: Tool result processed into user-friendly response
```

### Key Differences from Linear Workflow

**1. Efficiency:**
- Simple queries bypass tools entirely
- Faster response times for basic interactions

**2. Better User Experience:**
- Conversational queries get natural responses
- Tool results are processed and formatted nicely

**3. Resource Optimization:**
- Tools only execute when necessary
- Reduced computational overhead

## 🎓 Understanding Your Code

### Key Concepts Explained:

**1. Conditional Edge Mechanism:**
```python
graph_builder.add_conditional_edges(
    "llm",              # Source node
    tools_condition,    # Decision function
    {
        "tools": "tools",    # If condition returns "tools"
        "__end__": END       # If condition returns "__end__"
    }
)
```
- **Source Node**: Where the decision is made
- **Condition Function**: Evaluates state to determine route
- **Route Dictionary**: Maps condition results to destination nodes

**2. tools_condition Function:**
```python
# Built-in function that checks:
# - Does the last message have tool_calls?
# - If yes → return "tools"
# - If no → return "__end__"
```

**3. Cyclic Flow Pattern:**
```
LLM → Tools → LLM → (decision) → END or more tools
```
- Allows LLM to process tool results
- Enables multi-step tool workflows
- LLM can decide when to stop

**4. Message Flow in Conditional Workflow:**
```python
# Initial user message
{"messages": [{"role": "user", "content": "Tell me about Apple"}]}

# LLM decides to use tools
{"messages": [
    {"role": "user", "content": "Tell me about Apple"},
    {"role": "assistant", "tool_calls": [{"name": "wikipedia_tool", ...}]}
]}

# Tool executes and returns result
{"messages": [
    ...,
    {"role": "tool", "content": "Apple Inc. is a multinational..."}
]}

# LLM processes tool result and responds to user
{"messages": [
    ...,
    {"role": "assistant", "content": "Based on the information I found..."}
]}
```

## 🔧 Troubleshooting Guide

### Common Issues & Solutions:

**❌ "Graph compilation error with conditional edges"**
- **Check:** Route dictionary keys match condition function returns
- **Verify:** "__end__" is spelled correctly (with underscores)
- **Ensure:** Source node name matches added node name

**❌ "Tools always execute even for simple queries"**
- **Check:** `tools_condition` is imported correctly
- **Verify:** LLM is properly bound with tools
- **Debug:** Print condition function results

**❌ "Infinite loop between LLM and tools"**
- **Check:** tools → llm edge exists for result processing
- **Verify:** LLM can decide to end after processing tool results
- **Monitor:** Message flow to identify stuck points

**❌ "Direct responses don't work"**
```python
# Problem: Missing END route
{"tools": "tools"}  # Missing "__end__" route

# Solution: Include both routes
{"tools": "tools", "__end__": END}
```

## 🧪 Testing Challenges

### Challenge 1: Routing Behavior Analysis
```python
# Test routing decisions for various input types
routing_tests = [
    # Should NOT use tools
    ("Hello there!", False),
    ("How are you doing?", False), 
    ("What's your favorite food?", False),
    ("Thank you for your help", False),
    
    # SHOULD use tools
    ("Tell me about Microsoft", True),
    ("Show me Tesla stock data", True),
    ("Plot Apple stock prices", True),
    ("Research Amazon's business model", True),
]

def analyze_routing(request, should_use_tools):
    print(f"\n--- Routing Test: {request} ---")
    print(f"Expected tool usage: {should_use_tools}")
    
    chunks = list(graph.stream({
        "messages": [{"role": "user", "content": request}]
    }))
    
    # Analyze if tools were used
    tools_used = False
    for chunk in chunks:
        if "messages" in chunk:
            for message in chunk["messages"]:
                if message.get("role") == "tool":
                    tools_used = True
                    break
    
    print(f"Actual tool usage: {tools_used}")
    if should_use_tools == tools_used:
        print("✅ Routing decision correct")
    else:
        print("❌ Routing decision incorrect")

# Run routing analysis
for request, expected in routing_tests:
    analyze_routing(request, expected)
```

### Challenge 2: Multi-Step Tool Workflows
```python
# Test complex requests that might require multiple tool calls
complex_requests = [
    "Research Apple Inc. and then show me their recent stock performance",
    "Tell me about Tesla and create a visualization of their stock data",
    "Find information about Microsoft and analyze their stock trends"
]

for request in complex_requests:
    print(f"\n--- Complex Workflow Test ---")
    print(f"Request: {request}")
    
    tool_sequence = []
    chunks = list(graph.stream({
        "messages": [{"role": "user", "content": request}]
    }))
    
    # Track tool usage sequence
    for chunk in chunks:
        if "messages" in chunk:
            for message in chunk["messages"]:
                if message.get("role") == "assistant" and "tool_calls" in message:
                    for tool_call in message["tool_calls"]:
                        tool_sequence.append(tool_call["name"])
    
    print(f"Tool sequence: {tool_sequence}")
    if len(tool_sequence) > 1:
        print("✅ Multi-step workflow executed")
    else:
        print("ℹ️ Single-step or no tools used")
```

### Challenge 3: Response Quality Analysis
```python
# Compare response quality between direct and tool-assisted responses
def analyze_response_quality(request):
    print(f"\n--- Response Quality Test: {request} ---")
    
    chunks = list(graph.stream({
        "messages": [{"role": "user", "content": request}]
    }))
    
    # Extract final assistant response
    final_response = None
    tool_used = False
    
    for chunk in chunks:
        if "messages" in chunk:
            for message in chunk["messages"]:
                if message.get("role") == "assistant":
                    if "tool_calls" not in message:  # Final response, not tool call
                        final_response = message.get("content", "")
                elif message.get("role") == "tool":
                    tool_used = True
    
    print(f"Tool used: {tool_used}")
    print(f"Response length: {len(final_response) if final_response else 0} characters")
    print(f"Response preview: {(final_response[:100] + '...') if final_response and len(final_response) > 100 else final_response}")

# Test different response types
response_tests = [
    "Hello, how are you?",  # Direct response
    "Tell me about Apple Inc.",  # Tool-assisted response
    "What's the weather like?",  # Direct response (can't help)
    "Show me Tesla stock data for 7 days"  # Tool-assisted response
]

for test in response_tests:
    analyze_response_quality(test)
```

### Challenge 4: Error Handling in Conditional Flows
```python
# Test error handling with conditional routing
error_scenarios = [
    "Show me stock data for INVALID ticker",
    "Plot data for a company that doesn't exist",
    "Execute this broken Python code: print(undefined_variable)"
]

for scenario in error_scenarios:
    print(f"\n--- Error Handling Test ---")
    print(f"Request: {scenario}")
    
    try:
        chunks = list(graph.stream({
            "messages": [{"role": "user", "content": scenario}]
        }))
        
        # Check if error was handled gracefully
        error_handled = False
        final_response = None
        
        for chunk in chunks:
            if "messages" in chunk:
                for message in chunk["messages"]:
                    if message.get("role") == "assistant" and "content" in message:
                        final_response = message["content"]
                    elif message.get("role") == "tool" and ("Error" in message.get("content", "") or "Failed" in message.get("content", "")):
                        error_handled = True
        
        print(f"Error handled gracefully: {error_handled}")
        if final_response:
            print(f"Final response: {final_response[:100]}...")
            
    except Exception as e:
        print(f"❌ Unhandled exception: {e}")
```

## 🚀 Advanced Extensions

### Extension 1: Custom Condition Functions
```python
# Create custom routing logic
def smart_routing_condition(state):
    """Custom condition function for more sophisticated routing"""
    messages = state.get("messages", [])
    if not messages:
        return "__end__"
    
    last_message = messages[-1]
    
    # Check if it's an assistant message with tool calls
    if (last_message.get("role") == "assistant" and 
        "tool_calls" in last_message and 
        last_message["tool_calls"]):
        return "tools"
    
    # Custom logic: route based on message content
    content = last_message.get("content", "").lower()
    
    if any(word in content for word in ["urgent", "emergency", "critical"]):
        return "priority_tools"  # Could route to specialized high-priority tools
    elif any(word in content for word in ["complex", "detailed", "analysis"]):
        return "analysis_tools"  # Could route to analysis-specific tools
    else:
        return "__end__"

# This would require additional nodes and routing setup
```

### Extension 2: Tool Usage Statistics
```python
# Track tool usage patterns
tool_usage_stats = {
    "total_requests": 0,
    "tool_requests": 0,
    "direct_responses": 0,
    "tool_breakdown": {}
}

def track_tool_usage(state):
    """Monitor and track tool usage patterns"""
    global tool_usage_stats
    
    tool_usage_stats["total_requests"] += 1
    
    messages = state.get("messages", [])
    for message in messages:
        if message.get("role") == "assistant" and "tool_calls" in message:
            tool_usage_stats["tool_requests"] += 1
            for tool_call in message["tool_calls"]:
                tool_name = tool_call.get("name", "unknown")
                tool_usage_stats["tool_breakdown"][tool_name] = \
                    tool_usage_stats["tool_breakdown"].get(tool_name, 0) + 1
            break
    else:
        tool_usage_stats["direct_responses"] += 1
    
    # Call original condition
    return tools_condition(state)

# Replace tools_condition with track_tool_usage for monitoring
```

### Extension 3: Dynamic Tool Selection
```python
# Route to different tool sets based on request type
def dynamic_tool_routing(state):
    """Route to different tool nodes based on request analysis"""
    messages = state.get("messages", [])
    if not messages:
        return "__end__"
    
    last_message = messages[-1]
    content = last_message.get("content", "").lower()
    
    if "wikipedia" in content or "research" in content or "information" in content:
        return "research_tools"
    elif "stock" in content or "price" in content or "financial" in content:
        return "financial_tools"
    elif "plot" in content or "chart" in content or "visualize" in content:
        return "visualization_tools"
    else:
        return "__end__"

# This would require creating specialized tool nodes
# research_tools_node = ToolNode([wikipedia_tool])
# financial_tools_node = ToolNode([stock_data_tool])
# visualization_tools_node = ToolNode([python_repl_tool])
```

## 📝 Self-Assessment

**Check your understanding:**

□ I can implement conditional edges using `add_conditional_edges()`  
□ I understand how `tools_condition` evaluates LLM decisions  
□ I can create efficient workflows that only use tools when needed  
□ I know how to build cyclic flows (LLM ↔ Tools ↔ LLM)  
□ I understand the routing dictionary format for conditional edges  
□ I can debug conditional routing issues  
□ I can extend conditional logic for custom routing scenarios  

## 💡 Real-World Applications

**Where conditional routing is used:**
- **Customer Support**: Route to human agents only for complex issues
- **Content Moderation**: Flag problematic content while allowing normal posts
- **Financial Services**: Route to compliance review only for high-risk transactions
- **Healthcare**: Escalate to specialists only when AI confidence is low
- **E-commerce**: Use inventory tools only when checking availability

## 🎉 Congratulations!

You've successfully implemented intelligent conditional routing! This system can:

- ✅ **Make smart decisions** about when tools are actually needed
- ✅ **Optimize performance** by avoiding unnecessary tool calls
- ✅ **Provide better UX** with appropriate responses for different request types
- ✅ **Handle complex workflows** with multi-step tool interactions
- ✅ **Process tool results** into user-friendly responses

**Key Takeaways:**
- Conditional edges enable intelligent workflow routing
- `tools_condition` provides built-in tool decision logic
- Cyclic flows allow LLM to process and respond to tool results
- Different request types can follow optimized execution paths

## 🚀 Next Steps

After completing this exercise:

1. **Exercise 2.3:** Build multi-agent systems with specialized roles
2. **Exercise 2.4:** Implement human-in-the-loop workflows
3. **Advanced:** Create custom condition functions for complex routing logic
4. **Enterprise:** Build production-ready agents with error handling and monitoring

## 🔄 Workflow Pattern Evolution

**Your Journey:**
- **Exercise 1.6**: Basic linear graphs (START → Node → END)
- **Exercise 1.7**: Linear tool integration (START → LLM → Tools → END)
- **Exercise 2.2**: Conditional routing (Smart decisions!) ← You are here
- **Future**: Multi-agent coordination, specialized workflows, enterprise patterns

## 🧠 Architecture Benefits

**Conditional vs Linear Workflows:**

| Aspect | Linear Workflow | Conditional Workflow |
|--------|-----------------|---------------------|
| **Efficiency** | Always uses tools | Only when needed |
| **Response Time** | Consistent (slower) | Variable (optimized) |
| **Resource Usage** | High | Optimized |
| **User Experience** | Tool-centric | Context-appropriate |
| **Flexibility** | Rigid | Adaptive |
| **Debugging** | Simple | More complex |

You've now mastered intelligent agent workflows that can adapt their behavior based on user needs and LLM decisions! 🚀🔀🧠