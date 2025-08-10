# 🕸️ Activity 04: Introduction to LangGraph - Student Practice Guide

## 🎯 Learning Objectives

By the end of this activity, you will:
- Understand the graph-based approach to building AI agents
- Learn to create and manage graph state with message history
- Build your first LangGraph workflow with nodes and edges
- Master the fundamentals of graph compilation and execution
- Visualize and test graph-based agent workflows

## 📚 Background Context

**What is LangGraph?**
LangGraph is a powerful framework that uses graph-based thinking to build complex AI agent workflows:

- **🔗 Nodes**: Individual components (LLMs, tools, decision points)
- **➡️ Edges**: Connections that define workflow flow
- **📊 State**: Persistent data that flows between nodes
- **🎛️ Control**: Precise workflow management and branching

**Why Use LangGraph?**
1. **Better Control**: Define exact workflow paths and decision points
2. **State Management**: Maintain conversation history and context
3. **Scalability**: Build complex multi-agent systems
4. **Debugging**: Visualize and trace workflow execution
5. **Reliability**: Predictable agent behavior with explicit flow control

## 🏗️ Graph Architecture Concepts

**Traditional vs Graph-Based Agents:**

```
Traditional Approach:
LLM → Tool → LLM → Tool → Response

LangGraph Approach:
START → Node₁ → Node₂ → Decision → Node₃ → END
  ↑      ↓       ↓        ↓        ↓      ↑
  └─── State ←── State ←── State ←── State ─┘
```

**Core Components:**
- **State**: Data container that persists across the workflow
- **Nodes**: Processing units (LLM calls, tool executions, decisions)
- **Edges**: Connections defining workflow progression
- **START/END**: Entry and exit points for the workflow

## 🔧 Setup Instructions

### Step 1: Install Required Libraries
```bash
pip install --quiet langgraph==0.5.3 langchain-openai==0.3.16
```

### Step 2: Import Dependencies
```python
# TODO: Import the required modules
# Hint: You need typing.Annotated, TypedDict, StateGraph, START, END, add_messages, and ChatOpenAI
from typing import ________
from typing_extensions import ________
from langgraph.graph import ________, ________, ________
from langgraph.graph.message import ________
from langchain_openai import ________
```

<details>
<summary>💡 Hint for Step 2</summary>

You need to import:
- `Annotated` from typing for type annotations
- `TypedDict` from typing_extensions for state definition
- `StateGraph`, `START`, `END` from langgraph.graph for graph creation
- `add_messages` from langgraph.graph.message for message handling
- `ChatOpenAI` from langchain_openai for LLM integration
</details>

## 🏗️ Building Your First Graph

### Step 3: Create the State Class

**Your task:** Complete the State class that will manage conversation messages. You need to write about **65%** of the implementation.

```python
# TODO: Step 3a - Create the State class using TypedDict
# Hint: The class should inherit from TypedDict
class ________(________):
    # TODO: Step 3b - Define messages attribute with proper annotation
    # Hint: Use Annotated with list type and add_messages function
    messages: Annotated[________, ________]
```

<details>
<summary>🔍 Step 3 Hints</summary>

**Step 3a:** Use `State` as class name and inherit from `TypedDict`
**Step 3b:** Use `list` for the type and `add_messages` for the annotation
</details>

### Step 4: Initialize the Graph Builder

```python
# TODO: Step 4 - Create the graph state builder
# Hint: Initialize StateGraph with the State class
graph_builder = ________(________)
```

<details>
<summary>💡 Step 4 Hint</summary>

Use `StateGraph(State)` to create the graph builder with your State class.
</details>

### Step 5: Define the LLM and Node Function

**Your task:** Set up the LLM and create the node function.

```python
# TODO: Step 5a - Define the OpenAI LLM
# Hint: Use ChatOpenAI with model "gpt-4o-mini"
llm = ________(model="________")

# TODO: Step 5b - Create the llm_node function
# Hint: Function takes state parameter and returns dictionary with messages
def llm_node(state: State):
    # TODO: Step 5c - Invoke the LLM with state messages and return result
    # Hint: Use llm.invoke() with state["messages"] and wrap result in list
    return {"messages": [llm.________(state["________"])]}
```

<details>
<summary>🔍 Step 5 Hints</summary>

**Step 5a:** `ChatOpenAI(model="gpt-4o-mini")`
**Step 5b:** Function signature: `def llm_node(state: State):`
**Step 5c:** `llm.invoke(state["messages"])` and return as dictionary
</details>

### Step 6: Build the Graph Structure

**Your task:** Add nodes and edges to create the workflow.

```python
# TODO: Step 6a - Add the LLM node to the graph
# Hint: Use add_node() method with node name "llm" and llm_node function
graph_builder.________(________, ________)

# TODO: Step 6b - Add edge from START to llm node
# Hint: Use add_edge() method to connect START to "llm"
graph_builder.________(________, ________)

# TODO: Step 6c - Add edge from llm node to END
# Hint: Use add_edge() method to connect "llm" to END
graph_builder.________(________, ________)

# TODO: Step 6d - Compile the graph
# Hint: Use compile() method to create the executable graph
graph = graph_builder.________()
```

<details>
<summary>🔍 Step 6 Hints</summary>

**Step 6a:** `graph_builder.add_node("llm", llm_node)`
**Step 6b:** `graph_builder.add_edge(START, "llm")`
**Step 6c:** `graph_builder.add_edge("llm", END)`
**Step 6d:** `graph_builder.compile()`
</details>

### Step 7: Visualize Your Graph

```python
# TODO: Step 7 - Display the graph visualization
# Hint: Simply call the graph object to display it
________
```

<details>
<summary>💡 Step 7 Hint</summary>

Just call `graph` to display the visual representation of your workflow.
</details>

### Step 8: Test Your Graph

**Your task:** Run your graph with a test message.

```python
# TODO: Step 8a - Import the helper function for pretty printing
from course_helper_functions import pretty_print_messages

# TODO: Step 8b - Stream the graph execution with a test message
# Hint: Use graph.stream() with a messages dictionary containing user message
for chunk in graph.stream(
    {"messages": [{"role": "________", "content": "________"}]}
):
    # TODO: Step 8c - Print the results using the helper function
    # Hint: Use pretty_print_messages() to display the chunk
    ________(chunk)
```

<details>
<summary>💡 Testing Hints</summary>

- Use `"user"` as the role
- Try a message like `"Tell me about Apple Inc."`
- Use `pretty_print_messages(chunk)` to display results
</details>

## ✅ Expected Output

Your graph should:

1. **Display Visualization**: Show a graph with START → llm → END
2. **Process Messages**: Accept user input and generate AI responses
3. **Maintain State**: Keep conversation context in the messages list
4. **Stream Results**: Output responses in real-time chunks

Example output format:
```
User: Tell me about Apple Inc.
Assistant: Apple Inc. is a multinational technology company headquartered in Cupertino, California...
```

## 🎓 Understanding Your Code

### Key Concepts Explained:

**1. State Management:**
```python
class State(TypedDict):
    messages: Annotated[list, add_messages]
```
- **TypedDict**: Provides type safety for state structure
- **Annotated**: Adds metadata to specify how messages are handled
- **add_messages**: Function that appends new messages rather than replacing them

**2. Graph Builder Pattern:**
```python
graph_builder = StateGraph(State)
```
- **Builder Pattern**: Construct the graph step by step
- **State Integration**: Graph knows how to handle your State class
- **Compilation**: Convert builder instructions into executable graph

**3. Node Functions:**
```python
def llm_node(state: State):
    return {"messages": [llm.invoke(state["messages"])]}
```
- **State Input**: Receives current state with message history
- **LLM Invocation**: Processes all messages for context
- **State Update**: Returns new messages to add to state

**4. Graph Flow Control:**
```python
graph_builder.add_edge(START, "llm")
graph_builder.add_edge("llm", END)
```
- **Linear Flow**: Simple START → Node → END pattern
- **Entry Point**: START determines where user input enters
- **Exit Point**: END determines where final response exits

## 🔧 Troubleshooting Guide

### Common Issues & Solutions:

**❌ "No module named 'langgraph'"**
```bash
# Solution: Install LangGraph
pip install langgraph==0.5.3
```

**❌ "StateGraph requires a state class"**
- **Check**: Make sure State class is defined correctly
- **Verify**: State inherits from TypedDict
- **Confirm**: Passed State class to StateGraph constructor

**❌ "Node function error"**
```python
# Problem: Incorrect return format
return llm.invoke(state["messages"])  # Wrong

# Solution: Return dictionary with messages key
return {"messages": [llm.invoke(state["messages"])]}  # Correct
```

**❌ "Graph compilation fails"**
- **Check**: All nodes are connected properly
- **Verify**: START and END edges are defined
- **Ensure**: Node functions exist and are correctly named

**❌ "OpenAI API errors"**
- **Note**: Activity environment provides API access automatically
- **Check**: Using correct model name "gpt-4o-mini"
- **Verify**: ChatOpenAI import is correct

## 🧪 Testing Challenges

### Challenge 1: Different Message Types
```python
# Test with various message types
test_messages = [
    {"role": "user", "content": "What is machine learning?"},
    {"role": "user", "content": "Explain quantum computing in simple terms"},
    {"role": "user", "content": "Write a haiku about programming"},
    {"role": "user", "content": "What are the benefits of renewable energy?"}
]

for test_msg in test_messages:
    print(f"\n--- Testing: {test_msg['content'][:30]}... ---")
    for chunk in graph.stream({"messages": [test_msg]}):
        pretty_print_messages(chunk)
```

### Challenge 2: Multi-turn Conversations
```python
# Test conversation continuity
conversation_history = [
    {"role": "user", "content": "Tell me about Python programming"},
    {"role": "assistant", "content": "Python is a high-level programming language..."},
    {"role": "user", "content": "What makes it different from Java?"}
]

print("--- Multi-turn Conversation Test ---")
for chunk in graph.stream({"messages": conversation_history}):
    pretty_print_messages(chunk)
```

### Challenge 3: Graph Analysis
```python
# Analyze the graph structure
print("=== Graph Analysis ===")
print(f"Graph type: {type(graph)}")
print(f"Graph has nodes: {hasattr(graph, 'nodes')}")

# Try to inspect graph properties (if available)
try:
    print(f"Available methods: {[method for method in dir(graph) if not method.startswith('_')][:10]}")
except:
    print("Graph methods inspection not available")

# Test graph visualization
print("\n--- Graph Visualization ---")
graph  # Display the graph structure
```

### Challenge 4: State Inspection
```python
# Understand state flow
initial_state = {"messages": [{"role": "user", "content": "Hello, how does state work?"}]}

print("=== State Flow Analysis ===")
print(f"Initial state: {initial_state}")

# Execute and capture final state
result_generator = graph.stream(initial_state)
final_chunk = None
for chunk in result_generator:
    final_chunk = chunk
    print(f"Chunk received: {list(chunk.keys())}")

print(f"Final chunk structure: {final_chunk}")
```

## 🚀 Extension Experiments

### Experiment 1: Custom Response Processing
```python
# Create a modified node function with custom processing
def enhanced_llm_node(state: State):
    """Enhanced LLM node with response preprocessing"""
    response = llm.invoke(state["messages"])
    
    # Add custom metadata to the response
    enhanced_response = f"🤖 AI Assistant: {response.content}"
    
    # Return in the expected format
    from langchain_core.messages import AIMessage
    return {"messages": [AIMessage(content=enhanced_response)]}

# Create a new graph with enhanced node
enhanced_graph_builder = StateGraph(State)
enhanced_graph_builder.add_node("enhanced_llm", enhanced_llm_node)
enhanced_graph_builder.add_edge(START, "enhanced_llm")
enhanced_graph_builder.add_edge("enhanced_llm", END)
enhanced_graph = enhanced_graph_builder.compile()

# Test the enhanced graph
print("=== Enhanced Graph Test ===")
for chunk in enhanced_graph.stream({"messages": [{"role": "user", "content": "What is LangGraph?"}]}):
    pretty_print_messages(chunk)
```

### Experiment 2: Error Handling Node
```python
# Create a graph with error handling
def safe_llm_node(state: State):
    """LLM node with error handling"""
    try:
        response = llm.invoke(state["messages"])
        return {"messages": [response]}
    except Exception as e:
        from langchain_core.messages import AIMessage
        error_msg = AIMessage(content=f"I encountered an error: {str(e)}. Please try again.")
        return {"messages": [error_msg]}

# Build error-safe graph
safe_graph_builder = StateGraph(State)
safe_graph_builder.add_node("safe_llm", safe_llm_node)
safe_graph_builder.add_edge(START, "safe_llm")
safe_graph_builder.add_edge("safe_llm", END)
safe_graph = safe_graph_builder.compile()

print("=== Error-Safe Graph Test ===")
# This should handle errors gracefully
for chunk in safe_graph.stream({"messages": [{"role": "user", "content": "Test error handling"}]}):
    pretty_print_messages(chunk)
```

## 📝 Self-Assessment

**Check your understanding:**

□ I can explain the difference between traditional and graph-based agents  
□ I understand how TypedDict and Annotated work for state management  
□ I can create nodes and edges in a LangGraph workflow  
□ I know how to compile and execute a graph  
□ I understand message flow and state persistence  
□ I can visualize and debug graph structures  
□ I can modify node functions to customize behavior  

## 🎓 Conceptual Deep Dive

### Why Graphs Matter for AI Agents

**1. Explicit Control Flow:**
- Traditional agents: Implicit, unpredictable paths
- Graph agents: Explicit, traceable, controllable paths

**2. State Management:**
- Traditional: Limited context handling
- Graph: Persistent, structured state across workflow

**3. Debugging & Monitoring:**
- Traditional: Black box execution
- Graph: Visual representation, step-by-step inspection

**4. Scalability:**
- Traditional: Difficult to extend complex workflows
- Graph: Modular nodes, easy to add/remove/modify

### Graph Theory Applied to AI

**Nodes (Vertices):**
- Processing units: LLMs, tools, decision points
- Stateful: Can access and modify workflow state
- Reusable: Same node type can appear multiple times

**Edges (Connections):**
- Control flow: Define execution order
- Conditional: Can route based on state or results
- Weighted: Can have priorities or conditions

**State (Graph Data):**
- Persistent: Flows through the entire workflow
- Mutable: Nodes can read and modify state
- Typed: Structure enforced by TypedDict

## 🔄 Workflow Patterns

### Linear Pattern (This Exercise)
```
START → Node → END
```
- Simple, predictable flow
- Good for basic LLM interactions
- Foundation for more complex patterns

### Branching Pattern (Future Exercises)
```
START → Decision → Node_A → END
              ├→ Node_B → END
              └→ Node_C → END
```

### Loop Pattern (Advanced)
```
START → Node_A → Decision → Node_B → Decision → END
           ↑                  ↓         ↑
           └──────────────────┘         ↓
                                    Continue?
```

## 🚀 Next Steps

After completing this exercise:

1. **Exercise 1.7:** Add tools to your LangGraph workflow
2. **Exercise 1.8:** Build conditional branching and decision nodes
3. **Exercise 1.9:** Create multi-agent graph systems
4. **Advanced:** Implement human-in-the-loop workflows

## 💡 Real-World Applications

**Where LangGraph is used:**
- **Customer Service**: Multi-step support workflows
- **Content Creation**: Research → Draft → Review → Publish pipelines
- **Data Analysis**: Query → Process → Validate → Report workflows
- **Software Development**: Code → Test → Review → Deploy pipelines
- **Financial Analysis**: Research → Analysis → Risk Assessment → Reporting

## 🎉 Congratulations!

You've successfully created your first LangGraph workflow! This foundation enables:

- ✅ **Graph-based thinking** for agent design
- ✅ **State management** for conversation continuity
- ✅ **Visual workflows** for better understanding
- ✅ **Modular architecture** for scalable systems
- ✅ **Precise control** over agent behavior

**Key Takeaways:**
- Graphs provide explicit control over agent workflows
- State management enables context-aware conversations
- Nodes and edges create flexible, reusable architectures
- Visualization helps debug and understand complex systems

Ready to add tools and build more sophisticated graph workflows? Let's create some intelligent agent systems! 🚀🕸️

## 🏗️ Architecture Benefits

**Compared to Traditional Approaches:**

| Aspect | Traditional Chain | LangGraph |
|--------|------------------|-----------|
| **Flow Control** | Implicit, rigid | Explicit, flexible |
| **State Management** | Limited | Comprehensive |
| **Debugging** | Difficult | Visual, traceable |
| **Scalability** | Complex to extend | Modular, composable |
| **Error Handling** | Global try/catch | Node-level control |
| **Testing** | End-to-end only | Individual node testing |

This graph-based foundation will serve you well as you build increasingly sophisticated AI agent systems! 🎓