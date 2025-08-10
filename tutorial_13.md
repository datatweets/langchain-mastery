# Tutorial 13: Building Smart Research Assistants with LangGraph + Wikipedia

## 🎯 What You'll Learn

In this tutorial, you'll learn how to:

- Combine LangGraph workflows with external tools (Wikipedia)
- Build AI agents that can research information and answer follow-up questions
- Implement conditional routing - AI decides when to use tools vs. respond directly
- Add persistent memory that survives across multiple conversations
- Create visual diagrams of complex AI workflows with tool integration

Think of this as creating a smart research assistant that can look up information on demand, remember everything from your conversation, and make intelligent decisions about when to research vs. when to respond from memory!

## 🤔 Simple Chatbot vs. Research Assistant

### Script 12 (Simple Chatbot):
```
You: "Tell me about the Eiffel Tower"
AI: "The Eiffel Tower is a famous landmark..." (limited to training data)

You: "Who built it?" 
AI: "I believe it was Gustave Eiffel..." (might be uncertain or outdated)
```

### Script 13 (Research Assistant):
```
You: "Tell me about the Eiffel Tower"
AI: *searches Wikipedia* → "According to Wikipedia, the Eiffel Tower is a wrought-iron lattice tower located in Paris, France. Built in 1889..."

You: "Who built it?"
AI: "From our previous conversation, you asked about the Eiffel Tower. It was built by Gustave Eiffel and his company." (remembers context!)
```

**The Power**: Combines real-time research with perfect memory for the ultimate AI assistant!

## 🧠 Key Innovations in This Tutorial

### Advanced LangGraph Features:
- **Tool Integration**: AI can use Wikipedia on demand
- **Conditional Routing**: AI decides when to research vs. respond directly
- **Persistent Memory**: Conversations survive across sessions
- **Visual Workflows**: Multiple diagrams showing different configurations

### Smart Decision Making:
- **When to research**: New topics or specific facts
- **When to remember**: Follow-up questions about previous topics
- **When to combine**: Use both memory and new research

## 🔍 Understanding the Code: Line by Line

Let's examine `script_13.py` step by step:

### Step 1: New Imports for Wikipedia and Advanced Features

```python
# wikipedia_chatbot_graph.py
# Chatbot with Wikipedia tool via LangGraph, conversation memory, and PNG diagram export.

import os
from dotenv import load_dotenv

# --- Load env (expects OPENAI_API_KEY in .env) ---
load_dotenv()

# --- Core LangChain / LangGraph imports ---
from langchain_openai import ChatOpenAI
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Wikipedia tool
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper

# Tool routing in LangGraph
from langgraph.prebuilt import ToolNode, tools_condition

# Memory checkpointer
from langgraph.checkpoint.memory import MemorySaver
```

**What's NEW here?**

1. **Wikipedia Integration**:
   ```python
   from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
   from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
   ```
   - **`WikipediaQueryRun`**: The actual tool that searches Wikipedia
   - **`WikipediaAPIWrapper`**: Handles the connection to Wikipedia's API
   - **Purpose**: Gives AI access to real-time, accurate information

2. **Advanced LangGraph Features**:
   ```python
   from langgraph.prebuilt import ToolNode, tools_condition
   ```
   - **`ToolNode`**: Pre-built node that executes tools automatically
   - **`tools_condition`**: Smart routing function that decides when to use tools
   - **Benefit**: Less code, more intelligence

3. **Persistent Memory**:
   ```python
   from langgraph.checkpoint.memory import MemorySaver
   ```
   - **`MemorySaver`**: Saves conversation state between sessions
   - **Persistence**: Conversations survive program restarts
   - **Thread management**: Multiple conversation threads

### Step 2: Setting Up Wikipedia Integration

```python
# -----------------------------
# 3) Build Wikipedia tool + bind to model
# -----------------------------
# Initialize Wikipedia API wrapper to fetch only the top result
api_wrapper = WikipediaAPIWrapper(top_k_results=1)

# Create the Wikipedia query tool and make a tools list
wikipedia_tool = WikipediaQueryRun(api_wrapper=api_wrapper)
tools = [wikipedia_tool]

# Bind tools to the LLM so it can emit tool-calls in responses
llm_with_tools = llm.bind_tools(tools)
```

**What's happening here?** (Setting up the research capability!)

1. **`WikipediaAPIWrapper(top_k_results=1)`**:
   - **Purpose**: Configures how many Wikipedia results to fetch
   - **`top_k_results=1`**: Gets only the most relevant article
   - **Why limit**: Focused, relevant information without overwhelming the AI

2. **`WikipediaQueryRun(api_wrapper=api_wrapper)`**:
   - **Creates**: The actual tool that AI can use
   - **Wraps**: The Wikipedia API in a format LangChain understands
   - **Result**: A tool the AI can "call" to search Wikipedia

3. **`tools = [wikipedia_tool]`**:
   - **List format**: Allows adding more tools later
   - **Extensible**: Could add calculator, weather, news, etc.
   - **Currently**: Just Wikipedia for focused learning

4. **`llm.bind_tools(tools)`** - THE MAGIC CONNECTION!
   - **Binds**: Tools to the AI model
   - **Result**: AI can now decide to use Wikipedia
   - **How it works**: AI generates special "tool call" messages
   - **Think of it as**: Giving the AI access to a phone directory of services

### Step 3: Enhanced Graph with Tool Routing

```python
# -----------------------------
# 4) Graph: nodes and edges
# -----------------------------
graph_builder = StateGraph(State)

# Chatbot node that lets the model decide whether to call the tool
def chatbot(state: State):
    # Pass the current messages to the tool-enabled model
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Add chatbot node
graph_builder.add_node("chatbot", chatbot)

# Create a ToolNode to execute tool calls when the model requests them
tool_node = ToolNode(tools=[wikipedia_tool])
graph_builder.add_node("tools", tool_node)
```

**What's NEW here?** (Building the intelligent workflow!)

1. **Enhanced Chatbot Node**:
   ```python
   def chatbot(state: State):
       return {"messages": [llm_with_tools.invoke(state["messages"])]}
   ```
   - **Uses**: `llm_with_tools` instead of regular `llm`
   - **Capability**: Can generate tool calls when needed
   - **Decision making**: AI decides whether to research or respond directly

2. **`ToolNode(tools=[wikipedia_tool])`** - AUTOMATIC TOOL EXECUTION!
   - **Pre-built**: LangGraph provides this ready-made node
   - **Purpose**: Executes tool calls automatically
   - **Input**: Receives tool call requests from the chatbot
   - **Output**: Returns tool results back to the conversation
   - **Think of it as**: An automatic research assistant

### Step 4: Conditional Routing (The Intelligence!)

```python
# Conditional routing:
# - If the model called a tool, route to the "tools" node
# - Otherwise route toward END
graph_builder.add_conditional_edges("chatbot", tools_condition)

# Connect tools back to chatbot (to continue after tool results)
graph_builder.add_edge("tools", "chatbot")

# Start and end of the flow
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
```

**What's happening here?** (Building smart decision-making!)

1. **`graph_builder.add_conditional_edges("chatbot", tools_condition)`** - SMART ROUTING!
   - **Conditional**: Route depends on AI's decision
   - **`tools_condition`**: Pre-built function that checks if AI wants to use tools
   - **Logic**: 
     - If AI generated tool calls → go to "tools" node
     - If AI responded normally → go to END
   - **Think of it as**: A smart traffic light that reads the AI's intentions

2. **`graph_builder.add_edge("tools", "chatbot")`**:
   - **Purpose**: After using tools, go back to chatbot
   - **Why**: AI needs to process tool results and generate final response
   - **Flow**: Tools execute → results go back to AI → AI generates final answer

3. **Complete Workflow**:
   ```
   START → [Chatbot] → Decision Point
                        ↙         ↘
                   [Tools]      [END]
                        ↓
                   [Chatbot] → [END]
   ```

### Step 5: Graph Compilation and Visualization

```python
# Compile the graph (initial, without memory)
graph = graph_builder.compile()

# -----------------------------
# 5) Save & (optionally) display the graph diagram
# -----------------------------
def save_graph_png(g, out_path: str = "wikipedia_chatbot_graph.png"):
    try:
        png_bytes = g.get_graph().draw_mermaid_png()
        with open(out_path, "wb") as f:
            f.write(png_bytes)
        print(f"Diagram saved to {out_path}")
        if _HAVE_IPY:
            display(Image(png_bytes))
    except Exception:
        print("Diagram generation requires additional dependencies.")

# Save a diagram for the initial graph (without memory)
save_graph_png(graph, "wikipedia_chatbot_graph_nomemory.png")
```

**What's happening here?**

1. **`graph_builder.compile()`**:
   - **First compilation**: Without memory features
   - **Purpose**: Creates working graph for immediate use
   - **Result**: Basic research assistant without persistence

2. **`save_graph_png()` function**:
   - **Reusable**: Can save diagrams of different graph configurations
   - **Smart naming**: Different files for different versions
   - **Visual documentation**: See exactly how your AI workflow works

3. **First diagram**: `"wikipedia_chatbot_graph_nomemory.png"`
   - **Shows**: Basic workflow without memory
   - **Useful for**: Understanding core tool integration

### Step 6: Adding Persistent Memory

```python
# -----------------------------
# 6) Add memory and recompile
# -----------------------------
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

# Save a diagram for the memory-enabled graph
save_graph_png(graph, "wikipedia_chatbot_graph_with_memory.png")
```

**What's NEW here?** (Adding perfect memory!)

1. **`memory = MemorySaver()`**:
   - **Creates**: Memory storage system
   - **Persistence**: Conversations survive between runs
   - **Thread-based**: Multiple conversation threads possible

2. **`graph_builder.compile(checkpointer=memory)`**:
   - **Recompilation**: Same graph structure, but with memory
   - **`checkpointer=memory`**: Enables persistent conversation state
   - **Result**: Research assistant with perfect memory

3. **Second diagram**: `"wikipedia_chatbot_graph_with_memory.png"`
   - **Shows**: Enhanced workflow with memory capabilities
   - **Comparison**: Side-by-side with non-memory version

### Step 7: Memory-Enabled Conversations

```python
# -----------------------------
# 7) Stream responses with memory
# -----------------------------
def stream_memory_responses(user_input: str):
    """
    Streams events from the memory-enabled graph. Uses a single thread_id so
    follow-up questions can rely on previous turns automatically.
    """
    config = {"configurable": {"thread_id": "single_session_memory"}}

    # Stream events; LangGraph yields partial states as the graph runs
    for event in graph.stream({"messages": [("user", user_input)]}, config):
        for value in event.values():
            if "messages" in value and value["messages"]:
                # Print the content of the latest message only (cleaner for terminal)
                last = value["messages"][-1]
                content = getattr(last, "content", last[1] if isinstance(last, (tuple, list)) else str(last))
                print("Agent:", content)
```

**What's happening here?** (Implementing persistent conversations!)

1. **`config = {"configurable": {"thread_id": "single_session_memory"}}`**:
   - **Thread ID**: Unique identifier for this conversation
   - **Persistence**: All messages with this ID are remembered
   - **Sessions**: Different thread_ids = different conversation histories

2. **`graph.stream({"messages": [("user", user_input)]}, config)`**:
   - **Streaming**: Real-time responses
   - **Config parameter**: Tells LangGraph which conversation thread to use
   - **Memory integration**: Automatically loads previous conversation context

3. **Response handling**:
   - **Clean output**: Shows only the latest AI response
   - **Behind the scenes**: Full conversation history is maintained
   - **Flexibility**: Handles different message formats gracefully

### Step 8: Demonstration with Memory

```python
# -----------------------------
# 8) Demo
# -----------------------------
if __name__ == "__main__":
    # First question can be answered from model or by calling Wikipedia
    stream_memory_responses("Tell me about the Eiffel Tower.")
    # Follow-up relies on memory (no need to restate context)
    stream_memory_responses("Who built it?")
```

**What's happening here?** (Showcasing intelligent memory!)

1. **First Question**: `"Tell me about the Eiffel Tower."`
   - **AI Decision**: "I need specific facts about the Eiffel Tower"
   - **Action**: Searches Wikipedia for current, accurate information
   - **Result**: Detailed, factual response with latest information
   - **Memory**: Stores both question and detailed response

2. **Second Question**: `"Who built it?"`
   - **Context awareness**: AI knows "it" refers to the Eiffel Tower
   - **AI Decision**: "I can answer this from the previous Wikipedia search"
   - **Action**: Responds directly from memory without new search
   - **Result**: Quick, contextual answer
   - **Efficiency**: No unnecessary Wikipedia calls

## 🧠 What Happens Behind the Scenes?

### First Question Flow: "Tell me about the Eiffel Tower"

**Step 1: Input Processing**
```
User message → LangGraph loads conversation (empty initially)
```

**Step 2: Chatbot Node Decision**
```
AI with tools receives: [("user", "Tell me about the Eiffel Tower")]
AI thinks: "I need current, specific information about the Eiffel Tower"
AI generates: Tool call to search Wikipedia for "Eiffel Tower"
```

**Step 3: Conditional Routing**
```
tools_condition checks AI response
Finds: Tool call present
Routes to: "tools" node
```

**Step 4: Tool Execution**
```
ToolNode receives: Wikipedia search request
Executes: Wikipedia API call
Returns: Article content about Eiffel Tower
```

**Step 5: Back to Chatbot**
```
AI receives: Tool results + original question
AI generates: Comprehensive response based on Wikipedia data
Memory saves: Full conversation including tool usage
```

### Second Question Flow: "Who built it?"

**Step 1: Input Processing**
```
User message → LangGraph loads previous conversation
Context: Previous discussion about Eiffel Tower
```

**Step 2: Chatbot Node Decision**
```
AI receives: [Previous conversation] + [("user", "Who built it?")]
AI understands: "it" = Eiffel Tower from context
AI thinks: "I have this information from the previous Wikipedia search"
AI generates: Direct response (no tool call)
```

**Step 3: Conditional Routing**
```
tools_condition checks AI response
Finds: No tool calls
Routes to: END (direct response)
```

**Result**: Fast, contextual answer without unnecessary research!

## 🚀 How to Run This Code

### Prerequisites
1. **Install dependencies**: 
   ```bash
   pip install langgraph langchain-community wikipedia
   ```
2. **API key**: Set up OpenAI API key in `.env`

### Steps
1. **Run the script**:
   ```bash
   python script_13.py
   ```

2. **What you'll see**:
   ```
   Diagram saved to wikipedia_chatbot_graph_nomemory.png
   Diagram saved to wikipedia_chatbot_graph_with_memory.png
   
   Agent: According to Wikipedia, the Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower...
   
   Agent: The Eiffel Tower was built by Gustave Eiffel and his engineering company, Compagnie des Établissements Eiffel, between 1887 and 1889.
   ```

3. **Generated files**:
   - **`wikipedia_chatbot_graph_nomemory.png`**: Basic workflow diagram
   - **`wikipedia_chatbot_graph_with_memory.png`**: Memory-enhanced workflow diagram

## 🎓 Key Concepts You've Learned

### Tool Integration with LangGraph
- **Tool binding**: `llm.bind_tools(tools)` connects AI with external services
- **Conditional routing**: AI decides when to use tools vs. respond directly
- **ToolNode**: Pre-built component for automatic tool execution
- **tools_condition**: Smart routing based on AI's tool usage decisions

### Wikipedia Integration
- **WikipediaAPIWrapper**: Handles connection to Wikipedia's API
- **WikipediaQueryRun**: LangChain tool for Wikipedia searches
- **Configuration**: `top_k_results` controls information volume
- **Real-time data**: Always current and accurate information

### Advanced Memory Management
- **MemorySaver**: Persistent conversation storage
- **Thread-based memory**: Multiple conversation contexts
- **Cross-session persistence**: Conversations survive program restarts
- **Automatic context loading**: Previous conversations automatically available

### Intelligent Decision Making
- **Context awareness**: AI understands pronoun references ("it", "that", etc.)
- **Efficiency optimization**: Avoids unnecessary tool calls
- **Research vs. memory**: Smart choice between new research and existing knowledge
- **Follow-up handling**: Natural conversation flow

## 🔧 Common Issues and Solutions

**Problem: "Wikipedia module not found"**
```bash
# Solution: Install wikipedia package
pip install wikipedia
```

**Problem: "Tool calls not working"**
```python
# Wrong - using regular LLM
def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

# Right - using tool-enabled LLM
def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}
```

**Problem: "Memory not persisting"**
```python
# Wrong - not using config
for event in graph.stream({"messages": [("user", user_input)]}):

# Right - using thread config
config = {"configurable": {"thread_id": "session_1"}}
for event in graph.stream({"messages": [("user", user_input)]}, config):
```

**Problem: "AI always searches Wikipedia"**
```python
# Solution: Improve system instructions
system_msg = ("system", """You are a helpful assistant with access to Wikipedia. 
Use Wikipedia for new topics or when you need current facts. 
For follow-up questions about topics we've already discussed, 
use information from our conversation history when possible.""")
```

## 🎯 Try These Experiments

### 1. Multiple Tool Types
```python
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

# Add multiple research tools
wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1))
search_tool = DuckDuckGoSearchRun(api_wrapper=DuckDuckGoSearchAPIWrapper())

tools = [wikipedia_tool, search_tool]
llm_with_tools = llm.bind_tools(tools)

# AI can now choose between Wikipedia and web search!
```

### 2. Conversation Branching
```python
def start_new_conversation(thread_id: str):
    """Start a conversation in a new thread"""
    def stream_with_thread(user_input: str):
        config = {"configurable": {"thread_id": thread_id}}
        for event in graph.stream({"messages": [("user", user_input)]}, config):
            # ... handle response
    return stream_with_thread

# Create separate conversation threads
work_chat = start_new_conversation("work_research")
personal_chat = start_new_conversation("personal_learning")

work_chat("Research project management tools")
personal_chat("Tell me about cooking techniques")
```

### 3. Smarter Tool Selection
```python
def enhanced_chatbot(state: State):
    """Chatbot with smarter tool selection logic"""
    messages = state["messages"]
    last_user_msg = [msg for msg in messages if hasattr(msg, 'type') and msg.type == 'human'][-1].content
    
    # Add context about when to use tools
    enhanced_messages = messages + [
        ("system", f"""Previous context: {len(messages)} messages in conversation.
        For the question '{last_user_msg}':
        - Use Wikipedia for factual, encyclopedic information
        - Use web search for current events or recent developments
        - Use memory for follow-up questions about previous topics""")
    ]
    
    return {"messages": [llm_with_tools.invoke(enhanced_messages)]}
```

### 4. Research Report Generation
```python
def research_and_report(topic: str, thread_id: str):
    """Generate comprehensive research report"""
    config = {"configurable": {"thread_id": thread_id}}
    
    # Research phase
    research_query = f"Research comprehensive information about {topic}"
    research_results = []
    
    for event in graph.stream({"messages": [("user", research_query)]}, config):
        # Collect research results
        pass
    
    # Report generation phase
    report_query = f"Based on the research about {topic}, create a structured report with: 1) Overview 2) Key facts 3) Historical context 4) Current relevance"
    
    for event in graph.stream({"messages": [("user", report_query)]}, config):
        # Generate final report
        pass

# Usage
research_and_report("Renewable Energy", "research_session_1")
```

## 🌟 Advanced Features

### 1. Tool Result Processing
```python
class EnhancedState(TypedDict):
    messages: Annotated[list, add_messages]
    tool_results: dict  # Store tool results for analysis
    research_topics: list  # Track what's been researched

def process_tool_results(state: EnhancedState):
    """Process and categorize tool results"""
    messages = state["messages"]
    tool_results = {}
    
    for msg in messages:
        if hasattr(msg, 'tool_calls'):
            # Process tool calls and results
            pass
    
    return {"tool_results": tool_results}
```

### 2. Smart Caching
```python
import hashlib
from datetime import datetime, timedelta

class SmartCache:
    def __init__(self, cache_duration_hours=24):
        self.cache = {}
        self.cache_duration = timedelta(hours=cache_duration_hours)
    
    def get_cached_result(self, query: str):
        """Get cached result if recent enough"""
        query_hash = hashlib.md5(query.encode()).hexdigest()
        if query_hash in self.cache:
            result, timestamp = self.cache[query_hash]
            if datetime.now() - timestamp < self.cache_duration:
                return result
        return None
    
    def cache_result(self, query: str, result: str):
        """Cache a search result"""
        query_hash = hashlib.md5(query.encode()).hexdigest()
        self.cache[query_hash] = (result, datetime.now())

# Use in tool node to avoid redundant searches
```

### 3. Multi-Source Research
```python
def multi_source_research(state: State):
    """Research from multiple sources and combine results"""
    messages = state["messages"]
    query = extract_research_query(messages)
    
    # Research from multiple sources in parallel
    wikipedia_result = wikipedia_tool.run(query)
    web_result = web_search_tool.run(query)
    
    # Combine and synthesize results
    combined_prompt = f"""
    Based on these research sources:
    
    Wikipedia: {wikipedia_result}
    Web Search: {web_result}
    
    Provide a comprehensive, well-sourced answer to: {query}
    """
    
    synthesis = llm.invoke([("user", combined_prompt)])
    return {"messages": [synthesis]}
```

## 📚 Best Practices for Research Assistants

### 1. Tool Selection Strategy
```python
# Good - Clear tool purposes
wikipedia_tool = WikipediaQueryRun(...)  # For encyclopedic facts
news_search = DuckDuckGoSearchRun(...)   # For current events
calculator = CalculatorTool(...)          # For calculations

# Bad - Redundant tools
web_search1 = GoogleSearchRun(...)
web_search2 = BingSearchRun(...)
web_search3 = DuckDuckGoSearchRun(...)   # Too many similar tools
```

### 2. Memory Management
```python
# Good - Organized thread management
def get_research_thread(topic: str) -> str:
    """Get consistent thread ID for topic"""
    return f"research_{topic.replace(' ', '_').lower()}"

# Bad - Random thread IDs
thread_id = f"thread_{random.randint(1, 1000)}"  # Inconsistent
```

### 3. Error Handling
```python
def robust_research(state: State):
    """Research with fallback options"""
    try:
        result = primary_research_tool.run(query)
        if not result or len(result) < 100:  # Too short
            result = fallback_research_tool.run(query)
        return {"messages": [("assistant", result)]}
    except Exception as e:
        return {"messages": [("assistant", f"Research temporarily unavailable: {str(e)}")]}
```

## 🌟 What's Next?

Now that you understand intelligent research assistants, you can:

- **Build domain-specific research tools** for specialized fields
- **Create multi-modal assistants** that can research text, images, and videos
- **Develop fact-checking systems** that verify information across sources
- **Build educational platforms** with adaptive research capabilities
- **Create business intelligence tools** with automated research workflows

Congratulations! You've mastered building intelligent AI research assistants! 🎉

## 💡 Real-World Applications

### Academic Research:
- **Literature reviews**: Automated research across academic databases
- **Fact verification**: Cross-reference information across sources
- **Citation management**: Track sources and generate bibliographies

### Business Intelligence:
- **Market research**: Gather competitive intelligence automatically
- **Industry analysis**: Track trends and developments
- **Due diligence**: Research companies and business opportunities

### Journalism & Content:
- **Story research**: Gather background information quickly
- **Fact-checking**: Verify claims and statements
- **Source discovery**: Find relevant experts and studies

### Education:
- **Homework assistance**: Help students research topics
- **Curriculum development**: Research teaching materials
- **Knowledge exploration**: Support curiosity-driven learning

### Personal Use:
- **Travel planning**: Research destinations and activities
- **Health information**: Research medical conditions and treatments
- **Hobby exploration**: Learn about new interests and skills

The combination of intelligent decision-making, tool integration, and persistent memory creates powerful research assistants for any domain! 🚀