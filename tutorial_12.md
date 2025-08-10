# Tutorial 12: Building Graph-Based AI Chatbots with LangGraph

## 🎯 What You'll Learn

In this tutorial, you'll learn how to:

- Understand LangGraph - a new way to build AI applications using graphs
- Create conversational AI systems with automatic state management
- Build streaming chatbots that respond in real-time
- Implement multi-turn conversations with persistent memory
- Visualize AI workflows as interactive diagrams

Think of this as upgrading from linear AI conversations to a visual, flowchart-based approach where you can see exactly how your AI thinks and flows!

## 🤔 Traditional LangChain vs LangGraph

### Traditional LangChain (Previous tutorials):
```
Input → Prompt → LLM → Output
(Linear, step-by-step processing)
```

### LangGraph (This tutorial):
```
     START
       ↓
   [Chatbot Node]
       ↓
      END
```

**Visual workflow with:**
- **Nodes**: Individual processing steps
- **Edges**: Connections between steps  
- **State**: Shared memory across the entire flow
- **Streaming**: Real-time responses

## 🧠 Why Use LangGraph?

### Advantages Over Traditional Approaches:

**Visual Understanding:**
- **See your AI workflow** as a diagram
- **Debug easily** by following the flow
- **Share designs** with non-technical people

**Better State Management:**
- **Automatic message handling** - no manual memory management
- **Persistent conversations** - remembers everything automatically
- **Type safety** - prevents common errors

**Scalability:**
- **Add complex workflows** easily
- **Multiple decision points** and branches
- **Parallel processing** capabilities
- **Easy to modify** and extend

## 🔍 Understanding the Code: Line by Line

Let's examine `script_12.py` step by step:

### Step 1: New Imports for LangGraph

```python
import os
from dotenv import load_dotenv

# --- Load environment (expects OPENAI_API_KEY in .env) ---
load_dotenv()

# --- LangChain / LangGraph imports ---
from langchain_openai import ChatOpenAI
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
```

**What's NEW here?**

1. **`from typing import Annotated`** - NEW!
   - **Purpose**: Adds extra information to data types
   - **Why needed**: Tells LangGraph how to handle different types of data
   - **Think of it as**: Labels that explain how to use variables

2. **`from typing_extensions import TypedDict`** - NEW!
   - **Purpose**: Creates structured data types with named fields
   - **Why needed**: Defines what our conversation state looks like
   - **Think of it as**: A template for organizing information

3. **`from langgraph.graph import StateGraph, START, END`** - THE CORE!
   - **`StateGraph`**: The main class for building AI workflows
   - **`START`**: Special marker for where the workflow begins
   - **`END`**: Special marker for where the workflow ends
   - **Think of it as**: Traffic signals for AI workflow

4. **`from langgraph.graph.message import add_messages`** - AUTOMATIC MEMORY!
   - **Purpose**: Automatically handles adding new messages to conversation history
   - **Why important**: No manual memory management needed
   - **Think of it as**: An automatic secretary that keeps perfect notes

### Step 2: Optional Display Imports

```python
# (Optional) inline display if running in a notebook; safe to import lazily later
try:
    from IPython.display import Image, display  # noqa: F401
    _HAVE_IPY = True
except Exception:
    _HAVE_IPY = False
```

**What's happening here?**

1. **Conditional import** - Smart error handling
   - **Purpose**: Imports visualization tools if available
   - **Why try/except**: Not everyone has Jupyter notebook installed
   - **Graceful degradation**: Works with or without visualization

2. **`_HAVE_IPY`** flag:
   - **Purpose**: Remembers if visualization is available
   - **Used later**: To decide whether to show diagrams inline
   - **Best practice**: Check capabilities before using them

### Step 3: Setting Up the AI Model

```python
# -----------------------------
# 1) Define the LLM
# -----------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.2,
)
```

**What's happening here?**

1. **`model="gpt-4o-mini"`** - Different model!
   - **vs previous tutorials**: Used "gpt-4o" (full version)
   - **gpt-4o-mini**: Faster, cheaper, still very capable
   - **Good for**: Chatbots, conversations, quick responses

2. **`temperature=0.2`** - Balanced creativity
   - **Not 0**: Allows some creativity for natural conversation
   - **Not too high**: Still focused and reliable
   - **Perfect for**: Chatbot personalities

### Step 4: Defining the Conversation State

```python
# -----------------------------
# 2) Define the State schema
# -----------------------------
class State(TypedDict):
    # An accumulating list of chat messages; LangGraph will merge/append for us
    messages: Annotated[list, add_messages]
```

**What's NEW here?** (This is the heart of LangGraph!)

1. **`class State(TypedDict)`**:
   - **Purpose**: Defines the structure of conversation memory
   - **TypedDict**: Like a dictionary but with guaranteed structure
   - **Think of it as**: A blueprint for organizing conversation data

2. **`messages: Annotated[list, add_messages]`** - THE MAGIC!
   - **`messages`**: The name of our conversation history field
   - **`list`**: It will store a list of conversation messages
   - **`Annotated[..., add_messages]`**: Tells LangGraph "automatically manage this list by adding new messages"
   - **Result**: We never have to manually manage conversation history!

**What `add_messages` does automatically:**
```python
# Without add_messages (manual work):
state["messages"] = state["messages"] + [new_message]

# With add_messages (automatic):
# LangGraph handles this for us completely!
```

### Step 5: Building the Graph Structure

```python
# -----------------------------
# 3) Build the graph
# -----------------------------
graph_builder = StateGraph(State)
```

**What's happening here?**

1. **`StateGraph(State)`** - Creates the workflow builder
   - **`State`**: Uses our conversation structure we defined above
   - **`StateGraph`**: The main class for building AI workflows
   - **Think of it as**: A blank flowchart that we'll fill in

### Step 6: Creating the Chatbot Node

```python
def chatbot(state: State):
    """
    Single node: call the LLM with running 'messages' and return the AI's reply.
    """
    ai_message = llm.invoke(state["messages"])
    return {"messages": [ai_message]}
```

**What's happening here?** (This is a graph node!)

1. **Function definition**: `def chatbot(state: State)`
   - **Parameter**: Takes the current conversation state
   - **Return**: Returns new data to add to the state
   - **Think of it as**: One step in a flowchart

2. **`ai_message = llm.invoke(state["messages"])`**:
   - **Gets**: The full conversation history from state
   - **Sends**: Everything to the AI model
   - **Receives**: AI's response based on full context

3. **`return {"messages": [ai_message]}`**:
   - **Returns**: The AI's new message
   - **LangGraph magic**: Automatically merges this with existing messages
   - **Result**: Conversation history grows automatically

**How it works:**
```
Input State: {"messages": [user_msg1, ai_msg1, user_msg2]}
↓
AI processes all messages and generates response
↓
Output: {"messages": [ai_msg2]}
↓
LangGraph merges: {"messages": [user_msg1, ai_msg1, user_msg2, ai_msg2]}
```

### Step 7: Building the Graph Flow

```python
# Add node and edges
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# Compile the graph
graph = graph_builder.compile()
```

**What's happening here?** (Building the flowchart!)

1. **`graph_builder.add_node("chatbot", chatbot)`**:
   - **"chatbot"**: Name of this step in the workflow
   - **`chatbot`**: The function to run for this step
   - **Think of it as**: Adding a labeled box to a flowchart

2. **`graph_builder.add_edge(START, "chatbot")`**:
   - **Creates**: An arrow from START to the chatbot step
   - **Meaning**: "When the workflow begins, go to the chatbot"
   - **Think of it as**: Drawing an arrow in a flowchart

3. **`graph_builder.add_edge("chatbot", END)`**:
   - **Creates**: An arrow from chatbot to END
   - **Meaning**: "After the chatbot responds, finish"
   - **Think of it as**: Drawing the final arrow to completion

4. **`graph = graph_builder.compile()`**:
   - **Compiles**: Turns the blueprint into a working system
   - **Optimization**: LangGraph prepares everything for efficient execution
   - **Think of it as**: Building a machine from blueprints

**Visual representation:**
```
START → [Chatbot Node] → END
```

### Step 8: Setting Up Conversation Memory

```python
# -----------------------------
# 4) Multi-turn conversation with state
# -----------------------------
# Seed the conversation with a concise, high-school-friendly style
system_style = ("system", "Answer concisely (2–4 sentences) for high-school students.")

state: State = {"messages": [system_style]}  # running memory
```

**What's happening here?**

1. **`system_style = ("system", "...")`**:
   - **Message type**: "system" (gives AI instructions)
   - **Content**: Instructions for how to behave
   - **Purpose**: Sets the AI's personality and response style

2. **`state: State = {"messages": [system_style]}`**:
   - **Initializes**: The conversation memory with system instructions
   - **Type annotation**: `: State` ensures we use the right structure
   - **Starting point**: Every conversation begins with these instructions

### Step 9: The Ask Function (Conversation Interface)

```python
def ask(user_input: str):
    """
    Stream a turn through the graph, print the latest agent message,
    and update the running state with full message history.
    """
    print(f"\nUser: {user_input}")
    # Merge prior state + new user message; `add_messages` will append properly
    for event in graph.stream({**state, "messages": [("user", user_input)]}):
        for item in event.values():
            msgs = item.get("messages", [])
            if msgs:
                last = msgs[-1]
                # Handle either LangChain Message objects or raw tuples
                content = getattr(last, "content", last[1] if isinstance(last, (tuple, list)) else str(last))
                print("Agent:", content)
                # Update state so next turn has full history
                state["messages"] = msgs
```

**What's happening here?** (This is the conversation engine!)

1. **Function purpose**:
   - **Takes**: User's question as input
   - **Does**: Runs it through the graph workflow
   - **Prints**: Real-time response from AI
   - **Updates**: Conversation memory for next turn

2. **`print(f"\\nUser: {user_input}")`**:
   - **Shows**: What the user asked
   - **Format**: Clean, readable conversation display

3. **`{**state, "messages": [("user", user_input)]}`** - SMART MERGING!
   - **`**state`**: Takes all existing conversation history
   - **`"messages": [("user", user_input)]`**: Adds the new user message
   - **LangGraph magic**: Automatically merges everything correctly
   - **Result**: Full conversation context sent to AI

4. **`for event in graph.stream(...)`** - STREAMING!
   - **Purpose**: Runs the workflow and gets real-time updates
   - **Streaming**: Responses appear as soon as available
   - **vs invoke**: More responsive for users

5. **Response handling**:
   ```python
   content = getattr(last, "content", last[1] if isinstance(last, (tuple, list)) else str(last))
   ```
   - **Flexible**: Handles different message formats
   - **`getattr`**: Safely gets content from message objects
   - **Fallback**: Works with simple tuples too

6. **`state["messages"] = msgs`**:
   - **Updates**: Global conversation memory
   - **Next turn**: Will have full history including this exchange
   - **Persistent**: Memory survives between function calls

### Step 10: Demo Conversations

```python
# -----------------------------
# 5) Demo runs
# -----------------------------
if __name__ == "__main__":
    ask("Who is Ada Lovelace?")
    ask("Name her collaborator and what machine it was.")  # demonstrates memory
```

**What's happening here?**

1. **First question**: `"Who is Ada Lovelace?"`
   - **Tests**: Basic AI knowledge
   - **Establishes**: Conversation context about Ada Lovelace

2. **Second question**: `"Name her collaborator and what machine it was."`
   - **Memory test**: Uses "her" referring to Ada Lovelace from previous question
   - **Context dependency**: Only works if AI remembers first question
   - **Demonstrates**: LangGraph's automatic memory management

### Step 11: Graph Visualization

```python
# -------------------------
# 6) Generate and save the graph diagram
# -------------------------
try:
    png_bytes = graph.get_graph().draw_mermaid_png()
    out_path = "chatbot_graph.png"
    with open(out_path, "wb") as f:
        f.write(png_bytes)
    print(f"\nDiagram saved to {out_path}")
    if _HAVE_IPY:
        display(Image(png_bytes))  # inline display if in a notebook
except Exception:
    print("Diagram generation requires additional dependencies.")
```

**What's happening here?** (Visualization magic!)

1. **`graph.get_graph().draw_mermaid_png()`**:
   - **Creates**: A visual diagram of the workflow
   - **Format**: PNG image showing nodes and connections
   - **Technology**: Uses Mermaid diagram format
   - **Automatic**: LangGraph generates this from your code

2. **Saving the diagram**:
   ```python
   with open(out_path, "wb") as f:
       f.write(png_bytes)
   ```
   - **Saves**: Diagram as "chatbot_graph.png" file
   - **Binary mode**: `"wb"` for image data
   - **Result**: You can open and view your AI workflow

3. **Conditional display**:
   ```python
   if _HAVE_IPY:
       display(Image(png_bytes))
   ```
   - **Smart**: Only tries to display if in Jupyter notebook
   - **Graceful**: Doesn't crash if not available

## 🧠 What Happens Behind the Scenes?

### Complete Workflow Execution:

**Initialization:**
```
state = {"messages": [("system", "Answer concisely...")]}
graph = START → [Chatbot Node] → END
```

**First Question: "Who is Ada Lovelace?"**
```
1. User input arrives
2. State merges: existing messages + ("user", "Who is Ada Lovelace?")
3. Graph streams through: START → Chatbot Node → END
4. Chatbot node: 
   - Receives full message history
   - AI generates response about Ada Lovelace
   - Returns new AI message
5. LangGraph automatically merges response into state
6. Updated state: [system_msg, user_msg1, ai_msg1]
```

**Second Question: "Name her collaborator..."**
```
1. User input arrives
2. State merges: [system_msg, user_msg1, ai_msg1] + ("user", "Name her collaborator...")
3. AI sees "her" and understands it refers to Ada Lovelace from context
4. AI responds with information about Charles Babbage and the Analytical Engine
5. Updated state: [system_msg, user_msg1, ai_msg1, user_msg2, ai_msg2]
```

**Visual Diagram Generation:**
```
1. LangGraph analyzes the compiled graph structure
2. Generates Mermaid diagram code
3. Renders as PNG image
4. Saves to chatbot_graph.png
```

## 🚀 How to Run This Code

### Prerequisites
1. **Install LangGraph**: `pip install langgraph`
2. **API key**: Set up your OpenAI API key in `.env`
3. **Optional**: Install visualization dependencies

### Steps
1. **Run the script**:
   ```bash
   python script_12.py
   ```

2. **What you'll see**:
   ```
   User: Who is Ada Lovelace?
   Agent: Ada Lovelace was a British mathematician and the world's first computer programmer. She wrote the first algorithm for Charles Babbage's Analytical Engine in 1843. She's considered a pioneer of computing and programming.
   
   User: Name her collaborator and what machine it was.
   Agent: Her collaborator was Charles Babbage, and the machine was the Analytical Engine. This was an early mechanical general-purpose computer designed in the 1830s.
   
   Diagram saved to chatbot_graph.png
   ```

3. **Generated files**:
   - **`chatbot_graph.png`**: Visual diagram of your AI workflow

## 🎓 Key Concepts You've Learned

### LangGraph Fundamentals
- **What**: Graph-based framework for building AI workflows
- **Why**: Visual, scalable, automatic state management
- **How**: Nodes (functions) + Edges (connections) + State (memory)

### State Management
- **TypedDict**: Structured data with guaranteed fields
- **Annotated types**: Extra information about how to handle data
- **add_messages**: Automatic conversation history management

### Graph Components
- **Nodes**: Individual processing steps (functions)
- **Edges**: Connections between steps
- **START/END**: Special markers for workflow boundaries
- **StateGraph**: The main builder class

### Streaming vs Invoke
- **Streaming**: Real-time responses as they're generated
- **Invoke**: Wait for complete response before returning
- **User experience**: Streaming feels more natural

### Automatic Memory
- **Manual (previous tutorials)**: Manually manage chat_history lists
- **Automatic (LangGraph)**: State management handled transparently
- **Benefits**: Less error-prone, cleaner code

## 🔧 Common Issues and Solutions

**Problem: "ModuleNotFoundError: No module named 'langgraph'"**
```bash
# Solution: Install LangGraph
pip install langgraph
```

**Problem: "Diagram generation fails"**
```bash
# Solution: Install visualization dependencies
pip install "langgraph[dev]"
# or
pip install pygraphviz pillow
```

**Problem: "Graph doesn't remember conversations"**
```python
# Wrong - not updating global state
def ask(user_input: str):
    for event in graph.stream({...}):
        # Missing: state["messages"] = msgs

# Right - update global state
def ask(user_input: str):
    for event in graph.stream({...}):
        state["messages"] = msgs  # This line is crucial!
```

**Problem: "Messages format errors"**
```python
# Solution: Check message format consistency
print("Current state:", state["messages"])
for msg in state["messages"]:
    print(f"Type: {type(msg)}, Content: {msg}")
```

## 🎯 Try These Experiments

### 1. Add Multiple Nodes
```python
def analyzer(state: State):
    """Analyze user sentiment before responding"""
    last_user_msg = [msg for msg in state["messages"] if msg[0] == "user"][-1][1]
    sentiment = "positive" if "thank" in last_user_msg.lower() else "neutral"
    return {"sentiment": sentiment}

def responder(state: State):
    """Respond based on sentiment analysis"""
    sentiment = state.get("sentiment", "neutral")
    style = "enthusiastic" if sentiment == "positive" else "helpful"
    # ... generate response based on style
    
# Add to graph
graph_builder.add_node("analyzer", analyzer)
graph_builder.add_node("responder", responder)
graph_builder.add_edge(START, "analyzer")
graph_builder.add_edge("analyzer", "responder")
graph_builder.add_edge("responder", END)
```

### 2. Conditional Workflows
```python
def router(state: State):
    """Route to different nodes based on question type"""
    last_msg = state["messages"][-1][1].lower()
    if "math" in last_msg:
        return "math_solver"
    elif "history" in last_msg:
        return "history_expert"
    else:
        return "general_chat"

# Add conditional edges
from langgraph.graph import END
graph_builder.add_conditional_edges(
    "router",
    router,
    {
        "math_solver": "math_node",
        "history_expert": "history_node", 
        "general_chat": "general_node"
    }
)
```

### 3. Persistent State Across Sessions
```python
import json
from pathlib import Path

def save_conversation(state: State, filename: str):
    """Save conversation to file"""
    with open(filename, 'w') as f:
        # Convert messages to serializable format
        serializable_messages = []
        for msg in state["messages"]:
            if hasattr(msg, 'content'):
                serializable_messages.append({"role": msg.__class__.__name__, "content": msg.content})
            else:
                serializable_messages.append({"role": msg[0], "content": msg[1]})
        json.dump(serializable_messages, f, indent=2)

def load_conversation(filename: str) -> State:
    """Load conversation from file"""
    if not Path(filename).exists():
        return {"messages": []}
    
    with open(filename, 'r') as f:
        messages = json.load(f)
    
    # Convert back to tuple format
    state_messages = [(msg["role"], msg["content"]) for msg in messages]
    return {"messages": state_messages}

# Usage
state = load_conversation("conversation.json")
# ... have conversation ...
save_conversation(state, "conversation.json")
```

### 4. Streaming with Progress Updates
```python
def ask_with_progress(user_input: str):
    """Show progress as the AI thinks"""
    print(f"\nUser: {user_input}")
    print("🤔 Thinking...")
    
    response_parts = []
    for event in graph.stream({**state, "messages": [("user", user_input)]}):
        for item in event.values():
            msgs = item.get("messages", [])
            if msgs:
                last = msgs[-1]
                content = getattr(last, "content", last[1] if isinstance(last, (tuple, list)) else str(last))
                
                # Simulate streaming effect
                if content not in response_parts:
                    response_parts.append(content)
                    print(f"💭 Generating... ({len(content)} characters)")
                
                state["messages"] = msgs
    
    print(f"✅ Complete!")
    print(f"Agent: {response_parts[-1]}")
```

## 🌟 Advanced LangGraph Features

### 1. Custom State Types
```python
from datetime import datetime

class AdvancedState(TypedDict):
    messages: Annotated[list, add_messages]
    user_info: dict  # Store user preferences
    conversation_id: str
    timestamp: datetime
    mood: str  # Track conversation mood

def enhanced_chatbot(state: AdvancedState):
    """Chatbot that uses all state information"""
    user_info = state.get("user_info", {})
    mood = state.get("mood", "neutral")
    
    # Customize response based on full state
    system_prompt = f"User prefers {user_info.get('style', 'normal')} responses. Current mood: {mood}"
    
    # ... generate response with context
```

### 2. Parallel Processing
```python
from langgraph.graph import StateGraph

def fact_checker(state: State):
    """Check facts in parallel"""
    # Run fact-checking
    return {"fact_status": "verified"}

def sentiment_analyzer(state: State):
    """Analyze sentiment in parallel"""
    # Run sentiment analysis
    return {"sentiment": "positive"}

# Run both in parallel before main response
graph_builder.add_node("fact_checker", fact_checker)
graph_builder.add_node("sentiment_analyzer", sentiment_analyzer)
graph_builder.add_node("main_response", main_responder)

# Parallel edges
graph_builder.add_edge(START, "fact_checker")
graph_builder.add_edge(START, "sentiment_analyzer")
graph_builder.add_edge("fact_checker", "main_response")
graph_builder.add_edge("sentiment_analyzer", "main_response")
```

### 3. Error Handling and Recovery
```python
def safe_chatbot(state: State):
    """Chatbot with error handling"""
    try:
        ai_message = llm.invoke(state["messages"])
        return {"messages": [ai_message], "error": None}
    except Exception as e:
        error_msg = ("assistant", f"I'm having trouble right now. Error: {str(e)}")
        return {"messages": [error_msg], "error": str(e)}

def error_handler(state: State):
    """Handle errors gracefully"""
    if state.get("error"):
        return {"messages": [("assistant", "Let me try a different approach...")]}
    return {}
```

## 📚 LangGraph Best Practices

### 1. State Design
```python
# Good - Clear, typed state
class ChatState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    session_id: str
    preferences: dict

# Bad - Unclear state
class BadState(TypedDict):
    data: dict  # Too vague
    stuff: list  # What kind of stuff?
```

### 2. Node Functions
```python
# Good - Clear, single responsibility
def generate_response(state: ChatState):
    """Generate AI response based on conversation"""
    # Single, clear purpose
    return {"messages": [llm.invoke(state["messages"])]}

# Bad - Doing too much
def do_everything(state: ChatState):
    """Do everything at once"""
    # Sentiment analysis, fact checking, response generation...
    # Too many responsibilities
```

### 3. Error Handling
```python
# Good - Graceful error handling
def robust_node(state: State):
    try:
        result = risky_operation(state)
        return {"data": result, "error": None}
    except Exception as e:
        return {"data": None, "error": str(e)}

# Bad - No error handling
def fragile_node(state: State):
    return {"data": risky_operation(state)}  # Might crash!
```

## 🌟 What's Next?

Now that you understand LangGraph, you can:

- **Build complex AI workflows** with multiple decision points
- **Create visual AI applications** that are easy to understand and debug
- **Implement sophisticated chatbots** with automatic memory management
- **Scale to enterprise applications** with robust state management
- **Combine with RAG systems** for knowledge-aware conversational AI

Congratulations! You've mastered graph-based AI development! 🎉

## 💡 Real-World Applications

### Customer Service:
- **Multi-step support flows**: Route → Analyze → Solve → Follow-up
- **Escalation paths**: Human handoff when AI reaches limits
- **Context preservation**: Remember entire customer interaction history

### Education:
- **Adaptive tutoring**: Assess → Teach → Test → Review
- **Learning paths**: Branch based on student understanding
- **Progress tracking**: Visual representation of learning journey

### Healthcare:
- **Symptom analysis**: Collect → Analyze → Recommend → Follow-up
- **Treatment workflows**: Diagnose → Plan → Monitor → Adjust
- **Decision support**: Multiple expert opinions in parallel

### Business Process Automation:
- **Approval workflows**: Request → Review → Approve → Execute
- **Data processing pipelines**: Collect → Clean → Analyze → Report
- **Quality assurance**: Multiple validation steps in sequence

The visual, graph-based approach makes complex AI systems manageable and maintainable! 🚀