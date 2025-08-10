# Tutorial 14: Multi-Tool AI Assistant with LangGraph and Memory

## 🎯 What You'll Learn

In this tutorial, you'll learn how to:

- Create a sophisticated AI assistant with multiple specialized tools
- Build custom tools that use AI models internally for complex reasoning
- Combine different types of tools (Wikipedia, text processing, AI-powered tools) in one system
- Implement advanced LangGraph workflows with conditional routing and memory
- Create streaming conversational interfaces with real-time responses
- Build production-ready AI assistants that remember context across conversations

Think of this as creating your ultimate AI Swiss Army knife - one assistant that can check palindromes, look up historical events, search Wikipedia, and remember everything from your conversation!

## 🤔 Why Multi-Tool AI Assistants?

### The Problem with Single-Purpose AI:
```
Traditional AI: "I can only do one thing"
User: "Check if 'racecar' is a palindrome"
AI: ✅ "Yes, it's a palindrome"

User: "What happened on July 20, 1969?"
AI: ❌ "I can't look up specific historical events"
```

### Multi-Tool AI Assistant:
```
Smart Assistant: "I can do many things!"
User: "Check if 'racecar' is a palindrome"
AI: ✅ "Yes, it's a palindrome" (uses palindrome tool)

User: "What happened on July 20, 1969?"  
AI: ✅ "Apollo 11 landed on the moon!" (uses historical events tool)

User: "Tell me more about Neil Armstrong"
AI: ✅ "Neil Armstrong was..." (uses Wikipedia tool)
```

**The Power**: One AI that intelligently chooses the right tool for each task!

## 🧠 Script 14: The Ultimate Multi-Tool Assistant

### New Features in This Tutorial:
1. **Three Different Tool Types**:
   - **Wikipedia Tool**: Real-world information lookup
   - **Custom Text Processing Tool**: Palindrome checker
   - **AI-Powered Tool**: Historical events using LLM internally

2. **Advanced LangGraph Features**:
   - **MessagesState**: More sophisticated state management
   - **Smart routing**: AI decides which tool to use
   - **Streaming responses**: Real-time conversation experience

3. **Production Features**:
   - **Error handling**: Graceful failures
   - **Memory persistence**: Conversations survive restarts
   - **Multiple interaction modes**: Single queries and multi-turn conversations

## 🔍 Understanding the Code: Line by Line

Let's examine `script_14.py` step by step:

### Step 1: Import and Environment Setup

```python
# multi_tool_graph_chatbot.py
# Full lesson demo: multiple tools + LangGraph + memory + PNG diagram export

import os
from dotenv import load_dotenv

load_dotenv()
```

**What's happening here?**

1. **Clear naming**: `multi_tool_graph_chatbot.py`
   - **Purpose**: Tells us exactly what this script does
   - **Components**: Multiple tools + Graph workflow + Chatbot functionality
   - **Memory**: Persistent conversation memory

2. **Standard setup**: Environment and API keys
   - **Same pattern**: Consistent with previous tutorials
   - **Best practice**: Keep sensitive information in `.env` files

### Step 2: LLM Configuration

```python
# ---- LLM ----
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.2,
)
```

**What's consistent here?**

1. **Same model choice**: `gpt-4o-mini`
   - **Performance**: Fast responses for interactive conversations
   - **Cost-effective**: Cheaper than full GPT-4o
   - **Capable**: Handles complex multi-tool reasoning

2. **Balanced temperature**: `0.2`
   - **Reliability**: Consistent tool usage decisions
   - **Creativity**: Some flexibility in responses
   - **Perfect for**: Interactive assistants

### Step 3: Tool Definitions - Three Different Types

#### Type 1: External API Tool (Wikipedia)

```python
# ---- Tools ----
from langchain.tools import tool
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun

# Wikipedia tool (top-1 result)
api_wrapper = WikipediaAPIWrapper(top_k_results=1)
wikipedia_tool = WikipediaQueryRun(api_wrapper=api_wrapper)
```

**What's happening here?**

1. **External API integration**: Connects to Wikipedia's API
   - **Real-time data**: Always current information
   - **Focused results**: `top_k_results=1` for relevance
   - **Pre-built tool**: LangChain provides this ready-made

#### Type 2: Custom Processing Tool (Palindrome Checker)

```python
# Palindrome checker tool
@tool
def palindrome_checker(text: str) -> str:
    """Check if a word or phrase is a palindrome."""
    cleaned_text = "".join(ch.lower() for ch in text if ch.isalnum())
    if cleaned_text == cleaned_text[::-1]:
        return f"The phrase or word '{text}' is a palindrome."
    else:
        return f"The phrase or word '{text}' is not a palindrome."
```

**What's NEW here?** (Custom text processing logic!)

1. **@tool decorator**: Converts regular function to LangChain tool
   - **Automatic**: AI can discover and use this tool
   - **Type hints**: `text: str` tells AI what input is expected
   - **Docstring**: Helps AI understand when to use this tool

2. **Text cleaning**: `"".join(ch.lower() for ch in text if ch.isalnum())`
   - **`ch.lower()`**: Converts to lowercase for comparison
   - **`ch.isalnum()`**: Keeps only letters and numbers
   - **Removes**: Spaces, punctuation, special characters
   - **Example**: "A man, a plan, a canal: Panama!" → "amanaplanacanalpanama"

3. **Palindrome logic**: `cleaned_text == cleaned_text[::-1]`
   - **`[::-1]`**: Reverses the string
   - **Comparison**: Checks if original equals reversed
   - **Result**: True if palindrome, False otherwise

4. **Friendly responses**: 
   ```python
   return f"The phrase or word '{text}' is a palindrome."
   # vs just returning True/False
   ```
   - **User-friendly**: Complete sentences for better UX
   - **Context**: Includes the original text in response

#### Type 3: AI-Powered Tool (Historical Events)

```python
# Tool that invokes the LLM inside the tool body
@tool
def historical_events(date_input: str) -> str:
    """Provide a list of important historical events for a given date in any format."""
    try:
        # Ask the LLM to interpret the date and enumerate events
        response = llm.invoke(
            f"List important historical events that occurred on {date_input}. "
            f"Answer concisely with bullets and brief dates."
        )
        return response.content
    except Exception as e:
        return f"Error retrieving events: {str(e)}"
```

**What's NEW here?** (AI tool that uses AI internally!)

1. **AI-powered tool**: Uses `llm.invoke()` inside the tool
   - **Meta-AI**: AI tool that uses AI to generate responses
   - **Flexibility**: Handles any date format ("July 20, 1969", "20/7/1969", etc.)
   - **Reasoning**: AI interprets dates and recalls historical knowledge

2. **Flexible input**: `date_input: str`
   - **Any format**: "May 8th 1945", "1989-11-09", "November 9, 1989"
   - **AI parsing**: LLM figures out what date is meant
   - **User-friendly**: No strict format requirements

3. **Structured prompting**: 
   ```python
   f"List important historical events that occurred on {date_input}. "
   f"Answer concisely with bullets and brief dates."
   ```
   - **Clear instructions**: Tells AI exactly what format to use
   - **Concise**: Asks for bullets, not long paragraphs
   - **Consistent**: Always formatted the same way

4. **Error handling**: `try/except` block
   - **Graceful failure**: Doesn't crash if LLM call fails
   - **Informative errors**: Users know what went wrong
   - **Reliable**: Tool always returns something

### Step 4: Tool Binding

```python
# Bind all tools to the LLM (so it can request tool calls)
tools = [wikipedia_tool, palindrome_checker, historical_events]
model_with_tools = llm.bind_tools(tools)
```

**What's happening here?**

1. **Tool collection**: `tools = [wikipedia_tool, palindrome_checker, historical_events]`
   - **All together**: All three tools available to AI
   - **Different types**: External API, custom logic, AI-powered
   - **Extensible**: Easy to add more tools

2. **Tool binding**: `llm.bind_tools(tools)`
   - **Connection**: Links AI model with all available tools
   - **Decision making**: AI can now choose which tool to use
   - **Intelligence**: AI reads tool descriptions to make smart choices

### Step 5: Advanced LangGraph Setup

```python
# ---- LangGraph wiring ----
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# Tool node (executes tool calls requested by the model)
tool_node = ToolNode(tools=tools)
```

**What's NEW here?**

1. **MessagesState**: `from langgraph.graph import MessagesState`
   - **Pre-built state**: LangGraph provides this ready-made
   - **vs TypedDict**: More sophisticated than our custom state
   - **Features**: Built-in message handling and validation

2. **ToolNode creation**: `tool_node = ToolNode(tools=tools)`
   - **All tools together**: One node that can execute any tool
   - **Automatic routing**: Executes whichever tool the AI requests
   - **Efficient**: Single node handles multiple tool types

### Step 6: Custom Routing Logic

```python
# Decide whether to continue (i.e., if last message contains tool calls)
def should_continue(state: MessagesState):
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return END
```

**What's happening here?** (Smart decision making!)

1. **Conditional routing**: AI decides next step dynamically
   - **Check last message**: Looks at AI's most recent response
   - **Tool detection**: Sees if AI requested any tools
   - **Smart routing**: Goes to tools if needed, ends if not

2. **`getattr(last_message, "tool_calls", None)`**:
   - **Safe access**: Doesn't crash if `tool_calls` attribute doesn't exist
   - **Default value**: Returns `None` if no tool calls
   - **Flexibility**: Works with different message types

3. **Return values**:
   - **"tools"**: Go execute the requested tools
   - **END**: Conversation turn is complete

### Step 7: Enhanced Model Calling

```python
# Call the model; if last AI message already contains a tool response, return it
from langchain_core.messages import AIMessage, HumanMessage

def call_model(state: MessagesState):
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
        # Return only the tool's response if present
        return {"messages": [AIMessage(content=last_message.tool_calls[0]["response"])]}
    # Otherwise, call the tool-enabled model with the full history
    return {"messages": [model_with_tools.invoke(state["messages"])]}
```

**What's happening here?** (Optimization and error handling!)

1. **Smart caching**: Checks if response already exists
   - **Efficiency**: Doesn't re-call model unnecessarily
   - **Cost saving**: Avoids duplicate API calls
   - **Performance**: Faster responses

2. **Type checking**: `isinstance(last_message, AIMessage)`
   - **Safety**: Ensures we're looking at AI messages
   - **Robustness**: Handles different message types gracefully

3. **Full context**: `model_with_tools.invoke(state["messages"])`
   - **Complete history**: AI sees entire conversation
   - **Tool capability**: Can request any available tool
   - **Smart decisions**: AI chooses appropriate tool for context

### Step 8: Graph Construction

```python
# Build the workflow graph
workflow = StateGraph(MessagesState)

# Nodes
workflow.add_node("chatbot", call_model)
workflow.add_node("tools", tool_node)

# Edges
workflow.add_edge(START, "chatbot")
workflow.add_conditional_edges("chatbot", should_continue, ["tools", END])
workflow.add_edge("tools", "chatbot")
```

**What's the flow?**

```
START → [Chatbot] → Decision Point
                      ↙         ↘
                 [Tools]      [END]
                      ↓
                 [Chatbot] → [END]
```

1. **Linear start**: Always begins with chatbot
2. **Smart routing**: Chatbot decides whether tools are needed
3. **Tool execution**: If needed, execute tools and return to chatbot
4. **Final response**: Chatbot processes tool results and responds

### Step 9: Memory and Compilation

```python
# Memory + compile
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
```

**What's happening here?**

1. **Persistent memory**: `MemorySaver()`
   - **Cross-session**: Conversations survive restarts
   - **Thread-based**: Multiple conversation threads
   - **Automatic**: Handles all memory management

2. **Compilation**: `workflow.compile(checkpointer=memory)`
   - **Optimization**: Prepares graph for efficient execution
   - **Memory integration**: Enables persistent conversations
   - **Production ready**: Optimized for performance

### Step 10: Visualization

```python
# ---- Save diagram(s) to PNG ----
def save_graph_png(graph_app, out_path: str):
    try:
        png_bytes = graph_app.get_graph().draw_mermaid_png()
        with open(out_path, "wb") as f:
            f.write(png_bytes)
        print(f"Diagram saved to {out_path}")
        # If running in a notebook, also show inline (best-effort)
        try:
            from IPython.display import Image, display
            display(Image(png_bytes))
        except Exception:
            pass
    except Exception:
        print("Diagram generation requires additional dependencies.")

save_graph_png(app, "multi_tool_graph_with_memory.png")
```

**What's improved here?**

1. **Nested error handling**: Two levels of try/except
   - **Outer**: Handles diagram generation failures
   - **Inner**: Handles notebook display failures
   - **Graceful**: Never crashes, always continues

2. **Multiple output options**: File + optional notebook display
   - **File output**: Always saves PNG file
   - **Notebook display**: Shows inline if possible
   - **Flexibility**: Works in different environments

### Step 11: Streaming Interface

```python
# ---- Streaming helpers ----
config = {"configurable": {"thread_id": "1"}}

def multi_tool_output(query: str):
    """Single-turn: stream the agent's answer for a single query."""
    inputs = {"messages": [HumanMessage(content=query)]}
    print(f"User: {query}")
    print("Agent: ", end="", flush=True)
    for msg, metadata in app.stream(inputs, config, stream_mode="messages"):
        if msg.content and not isinstance(msg, HumanMessage):
            print(msg.content, end="", flush=True)
    print("\n")
```

**What's NEW here?** (Real-time streaming!)

1. **Thread configuration**: `{"configurable": {"thread_id": "1"}}`
   - **Memory key**: Links to specific conversation thread
   - **Persistence**: All interactions with thread "1" are remembered
   - **Isolation**: Different thread IDs = different conversations

2. **Streaming interface**: `app.stream(inputs, config, stream_mode="messages")`
   - **Real-time**: Responses appear as they're generated
   - **Better UX**: Users see progress immediately
   - **Interactive**: Feels like chatting with a human

3. **Output formatting**: 
   ```python
   print("Agent: ", end="", flush=True)  # No newline, force output
   print(msg.content, end="", flush=True)  # Add content without newlines
   ```
   - **Continuous output**: Text appears smoothly
   - **No stuttering**: `flush=True` forces immediate display
   - **Clean format**: Proper "User:" and "Agent:" labels

### Step 12: Multi-Turn Conversations

```python
def user_agent_multiturn(queries):
    """Multi-turn: prints user query then streams only agent messages per turn."""
    for query in queries:
        print(f"User: {query}")
        agent_chunks = []
        for msg, metadata in app.stream(
            {"messages": [HumanMessage(content=query)]},
            config,
            stream_mode="messages",
        ):
            if msg.content and not isinstance(msg, HumanMessage):
                agent_chunks.append(msg.content)
        print("Agent: " + "".join(agent_chunks) + "\n")
```

**What's different here?**

1. **Batch processing**: Handles multiple queries in sequence
   - **Loop**: Processes each query one by one
   - **Memory**: Each query builds on previous ones
   - **Context**: AI remembers entire conversation

2. **Chunk collection**: `agent_chunks.append(msg.content)`
   - **Streaming**: Collects all parts of streamed response
   - **Assembly**: Joins chunks into complete response
   - **Clean output**: Single "Agent:" label per response

### Step 13: Comprehensive Demo

```python
# ---- Demo ----
if __name__ == "__main__":
    # Try the different tools
    multi_tool_output("Is `may a moody baby doom a yam` a palindrome?")
    multi_tool_output("What happened on 20th July, 1969?")
    multi_tool_output("Summarize the Eiffel Tower in 2 sentences.")

    # Multi-turn conversation with memory
    queries = [
        "Is `stressed desserts?` a palindrome?",
        "What about the word `kayak`?",
        "What happened on the May 8th, 1945?",
        "What about 9 November 1989?",
    ]
    user_agent_multiturn(queries)
```

**What's being demonstrated?**

1. **Tool variety**: Different tools for different tasks
   - **Palindrome tool**: Text processing logic
   - **Historical events tool**: AI-powered date interpretation
   - **Wikipedia tool**: External knowledge lookup

2. **Memory demonstration**: Follow-up questions
   - **"What about the word `kayak`?"**: References previous palindrome context
   - **"What about 9 November 1989?"**: References previous historical events context
   - **Context awareness**: AI understands what "what about" refers to

## 🧠 What Happens Behind the Scenes?

### Tool Selection Process:

**Query: "Is 'racecar' a palindrome?"**
```
1. AI analyzes query: "This is asking about palindrome detection"
2. AI scans available tools:
   - wikipedia_tool: For general information lookup
   - palindrome_checker: For checking if text is palindrome ✅ MATCH!
   - historical_events: For historical date queries
3. AI generates tool call: palindrome_checker("racecar")
4. Tool executes: Checks if "racecar" == "racecar"[::-1] 
5. Result: "The word 'racecar' is a palindrome."
```

**Query: "What happened on July 20, 1969?"**
```
1. AI analyzes query: "This is asking about historical events on a date"
2. AI scans available tools:
   - wikipedia_tool: For general information lookup
   - palindrome_checker: For checking palindromes
   - historical_events: For historical date queries ✅ MATCH!
3. AI generates tool call: historical_events("July 20, 1969")
4. Tool executes: Calls LLM internally with historical knowledge
5. Result: "Apollo 11 moon landing occurred on July 20, 1969..."
```

**Query: "Tell me about the Eiffel Tower"**
```
1. AI analyzes query: "This is asking for general information about a topic"
2. AI scans available tools:
   - wikipedia_tool: For encyclopedic information ✅ MATCH!
   - palindrome_checker: For checking palindromes
   - historical_events: For historical date queries
3. AI generates tool call: wikipedia_tool.run("Eiffel Tower")
4. Tool executes: Searches Wikipedia API
5. Result: Current, accurate information about Eiffel Tower
```

### Memory Flow:

**Conversation Flow:**
```
User: "Is 'stressed' a palindrome?"
Memory: [HumanMessage("Is 'stressed' a palindrome?")]
AI: Uses palindrome_checker → "No, 'stressed' is not a palindrome"
Memory: [HumanMessage(...), AIMessage(...)]

User: "What about 'desserts'?"
Memory: [Previous messages + HumanMessage("What about 'desserts'?")]
AI: Understands "what about" refers to palindrome checking
AI: Uses palindrome_checker → "No, 'desserts' is not a palindrome"  
Memory: [All previous + new messages]

User: "What if I combine them?"
Memory: [All conversation history]
AI: Understands context - checking if 'stressed desserts' is palindrome
AI: Uses palindrome_checker → "Yes, 'stressed desserts' is a palindrome!"
```

## 🚀 How to Run This Code

### Prerequisites
1. **Dependencies**: 
   ```bash
   pip install langgraph langchain-community wikipedia
   ```
2. **API key**: Set up OpenAI API key in `.env`

### Steps
1. **Run the script**:
   ```bash
   python script_14.py
   ```

2. **What you'll see**:
   ```
   User: Is `may a moody baby doom a yam` a palindrome?
   Agent: The phrase 'may a moody baby doom a yam' is a palindrome.

   User: What happened on 20th July, 1969?
   Agent: • Apollo 11 lunar module landing on the Moon
           • Neil Armstrong and Buzz Aldrin first humans to walk on Moon
           • "That's one small step for man..." famous quote

   User: Summarize the Eiffel Tower in 2 sentences.
   Agent: The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France, built in 1889 as the entrance arch to the 1889 World's Fair. It stands 324 meters tall and has become a global cultural icon of France and one of the most recognizable structures in the world.

   User: Is `stressed desserts?` a palindrome?
   Agent: The phrase 'stressed desserts?' is a palindrome.

   User: What about the word `kayak`?
   Agent: The word 'kayak' is a palindrome.
   ```

3. **Generated files**:
   - **`multi_tool_graph_with_memory.png`**: Visual diagram of the workflow

## 🎓 Key Concepts You've Learned

### Multi-Tool Architecture
- **Tool variety**: External APIs, custom logic, AI-powered tools
- **Intelligent routing**: AI chooses the right tool for each task
- **Extensibility**: Easy to add new tools and capabilities

### Advanced Tool Types
- **External API tools**: Wikipedia integration for real-time data
- **Custom processing tools**: Palindrome checker with text processing
- **AI-powered tools**: Historical events using LLM for reasoning

### Production-Ready Features
- **Error handling**: Graceful failures at multiple levels
- **Streaming interfaces**: Real-time conversation experience
- **Memory persistence**: Conversations survive across sessions
- **Visual documentation**: Automatic workflow diagrams

### LangGraph Advanced Patterns
- **MessagesState**: Sophisticated state management
- **Conditional routing**: Dynamic workflow decisions
- **Tool integration**: Seamless tool execution and result processing

## 🔧 Common Issues and Solutions

**Problem: "Tool not being selected"**
```python
# Wrong - unclear tool description
@tool
def my_tool(input: str) -> str:
    """Does something."""  # Too vague!

# Right - clear, specific description
@tool
def palindrome_checker(text: str) -> str:
    """Check if a word or phrase is a palindrome."""  # Clear purpose!
```

**Problem: "AI-powered tool fails"**
```python
# Wrong - no error handling
@tool
def historical_events(date_input: str) -> str:
    response = llm.invoke(f"Events on {date_input}")  # Might fail!
    return response.content

# Right - robust error handling
@tool
def historical_events(date_input: str) -> str:
    try:
        response = llm.invoke(f"Events on {date_input}")
        return response.content
    except Exception as e:
        return f"Error retrieving events: {str(e)}"
```

**Problem: "Memory not working"**
```python
# Wrong - no thread ID
config = {}  # No persistence!

# Right - consistent thread ID
config = {"configurable": {"thread_id": "1"}}  # Persistent memory!
```

**Problem: "Streaming output messy"**
```python
# Wrong - no output control
for msg, metadata in app.stream(inputs, config):
    print(msg.content)  # Messy output with newlines

# Right - controlled formatting
print("Agent: ", end="", flush=True)
for msg, metadata in app.stream(inputs, config, stream_mode="messages"):
    if msg.content and not isinstance(msg, HumanMessage):
        print(msg.content, end="", flush=True)
print("\n")
```

## 🎯 Try These Experiments

### 1. Add More Tool Types

```python
@tool
def calculator(expression: str) -> str:
    """Evaluate mathematical expressions safely."""
    try:
        # Only allow safe mathematical expressions
        allowed_chars = set('0123456789+-*/().')
        if all(c in allowed_chars or c.isspace() for c in expression):
            result = eval(expression)
            return f"The result of {expression} is {result}"
        else:
            return "Only basic mathematical expressions are allowed."
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"

@tool
def word_counter(text: str) -> str:
    """Count words, characters, and sentences in text."""
    words = len(text.split())
    chars = len(text)
    sentences = len([s for s in text.split('.') if s.strip()])
    return f"Text analysis: {words} words, {chars} characters, {sentences} sentences"

# Add to tools list
tools = [wikipedia_tool, palindrome_checker, historical_events, calculator, word_counter]
```

### 2. Advanced Memory Management

```python
def create_conversation_thread(thread_name: str):
    """Create a new conversation thread"""
    thread_config = {"configurable": {"thread_id": thread_name}}
    
    def chat_in_thread(query: str):
        inputs = {"messages": [HumanMessage(content=query)]}
        response_chunks = []
        for msg, metadata in app.stream(inputs, thread_config, stream_mode="messages"):
            if msg.content and not isinstance(msg, HumanMessage):
                response_chunks.append(msg.content)
        return "".join(response_chunks)
    
    return chat_in_thread

# Create separate conversation threads
work_assistant = create_conversation_thread("work")
personal_assistant = create_conversation_thread("personal")

# Work conversation
work_response = work_assistant("What are some historical events from 1969?")

# Personal conversation (separate memory)
personal_response = personal_assistant("Is 'level' a palindrome?")
```

### 3. Tool Result Processing

```python
@tool
def smart_research(topic: str) -> str:
    """Research a topic using multiple sources and provide comprehensive summary."""
    try:
        # Get Wikipedia info
        wiki_result = wikipedia_tool.run(topic)
        
        # Get historical context if it's a historical topic
        historical_prompt = f"What historical significance does {topic} have?"
        historical_result = llm.invoke(historical_prompt).content
        
        # Combine results
        summary_prompt = f"""
        Based on these sources:
        Wikipedia: {wiki_result}
        Historical context: {historical_result}
        
        Provide a comprehensive but concise summary of {topic}.
        """
        
        final_summary = llm.invoke(summary_prompt).content
        return final_summary
        
    except Exception as e:
        return f"Error researching {topic}: {str(e)}"
```

### 4. Interactive Tool Selection

```python
def interactive_assistant():
    """Interactive assistant that shows available tools"""
    print("🤖 Multi-Tool AI Assistant")
    print("Available tools:")
    for i, tool in enumerate(tools, 1):
        print(f"{i}. {tool.name}: {tool.description}")
    print("\nJust ask me anything! I'll choose the right tool automatically.\n")
    
    config = {"configurable": {"thread_id": "interactive"}}
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Goodbye! 👋")
            break
            
        print("🤖: ", end="", flush=True)
        for msg, metadata in app.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config,
            stream_mode="messages"
        ):
            if msg.content and not isinstance(msg, HumanMessage):
                print(msg.content, end="", flush=True)
        print("\n")

# Run interactive mode
if __name__ == "__main__":
    interactive_assistant()
```

### 5. Tool Performance Analytics

```python
import time
from collections import defaultdict

class ToolAnalytics:
    def __init__(self):
        self.tool_usage = defaultdict(int)
        self.tool_times = defaultdict(list)
    
    def track_tool_call(self, tool_name: str, execution_time: float):
        self.tool_usage[tool_name] += 1
        self.tool_times[tool_name].append(execution_time)
    
    def get_stats(self):
        stats = {}
        for tool_name in self.tool_usage:
            usage_count = self.tool_usage[tool_name]
            times = self.tool_times[tool_name]
            avg_time = sum(times) / len(times) if times else 0
            
            stats[tool_name] = {
                "usage_count": usage_count,
                "average_time": round(avg_time, 3),
                "total_time": round(sum(times), 3)
            }
        return stats

analytics = ToolAnalytics()

# Wrap tools with analytics
def track_tool_performance(original_tool):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = original_tool(*args, **kwargs)
        end_time = time.time()
        
        analytics.track_tool_call(original_tool.name, end_time - start_time)
        return result
    
    return wrapper

# Apply analytics to tools
for tool in tools:
    tool.func = track_tool_performance(tool.func)
```

## 🌟 Advanced Multi-Tool Patterns

### 1. Tool Chaining
```python
@tool
def research_and_verify(topic: str) -> str:
    """Research a topic and verify key facts using multiple tools."""
    # Step 1: Get basic information
    wiki_info = wikipedia_tool.run(topic)
    
    # Step 2: Check for historical dates mentioned
    date_pattern = r'\b(\d{4}|\d{1,2}/\d{1,2}/\d{4})\b'
    dates = re.findall(date_pattern, wiki_info)
    
    verification = []
    for date in dates[:2]:  # Check first 2 dates found
        historical_info = historical_events(date)
        verification.append(f"Date {date}: {historical_info}")
    
    # Step 3: Combine results
    result = f"Research on {topic}:\n\n{wiki_info}\n\nVerification:\n" + "\n".join(verification)
    return result
```

### 2. Conditional Tool Selection
```python
def smart_tool_router(query: str, conversation_history: list):
    """Enhanced routing logic based on query analysis and history"""
    query_lower = query.lower()
    
    # Context analysis
    recent_topics = []
    for msg in conversation_history[-6:]:  # Last 3 exchanges
        if isinstance(msg, HumanMessage):
            recent_topics.extend(msg.content.lower().split())
    
    # Smart routing rules
    if any(word in query_lower for word in ['palindrome', 'reverse', 'spell backwards']):
        return 'palindrome_checker'
    elif any(word in query_lower for word in ['when', 'what happened', 'historical', 'history']):
        return 'historical_events'
    elif any(word in query_lower for word in ['about', 'what is', 'tell me', 'explain']):
        return 'wikipedia_tool'
    elif 'calculate' in recent_topics and any(word in query_lower for word in ['that', 'it', 'same']):
        return 'calculator'  # Context-aware tool selection
    else:
        return 'general_llm'  # Default to general reasoning
```

### 3. Tool Result Fusion
```python
@tool
def comprehensive_analysis(topic: str) -> str:
    """Analyze a topic from multiple perspectives using all available tools."""
    results = {}
    
    # Try each tool that might be relevant
    try:
        results['encyclopedia'] = wikipedia_tool.run(topic)
    except:
        results['encyclopedia'] = "No encyclopedia information available"
    
    # Check if topic might be a historical event
    if any(word in topic.lower() for word in ['war', 'battle', 'revolution', 'independence']):
        try:
            results['historical'] = historical_events(topic)
        except:
            results['historical'] = "No historical information available"
    
    # Check if topic could be analyzed as text
    if len(topic.split()) == 1:  # Single word
        try:
            results['palindrome'] = palindrome_checker(topic)
        except:
            results['palindrome'] = "Not applicable for palindrome check"
    
    # Synthesize results
    synthesis_prompt = f"""
    Analyze this topic comprehensively: {topic}
    
    Available information:
    Encyclopedia: {results.get('encyclopedia', 'N/A')}
    Historical: {results.get('historical', 'N/A')}
    Text Analysis: {results.get('palindrome', 'N/A')}
    
    Provide a well-rounded analysis combining all relevant information.
    """
    
    synthesis = llm.invoke(synthesis_prompt).content
    return synthesis
```

## 📚 Best Practices for Multi-Tool Systems

### 1. Tool Design Principles
```python
# Good - Single responsibility, clear purpose
@tool
def word_count(text: str) -> str:
    """Count the number of words in the given text."""
    return f"Word count: {len(text.split())}"

# Bad - Multiple responsibilities, unclear purpose  
@tool
def text_analyzer(text: str) -> str:
    """Analyze text."""  # Too vague!
    # Does word count, character count, sentiment, etc.
```

### 2. Error Handling Hierarchy
```python
@tool
def robust_tool(input_data: str) -> str:
    """Example of comprehensive error handling."""
    try:
        # Input validation
        if not input_data or not input_data.strip():
            return "Error: Empty input provided"
        
        # Main processing
        result = process_data(input_data)
        
        # Output validation
        if not result:
            return "Warning: No results found for the given input"
        
        return result
        
    except ValueError as e:
        return f"Input error: {str(e)}"
    except ConnectionError as e:
        return f"Connection error: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"
```

### 3. Memory Management
```python
class ConversationManager:
    def __init__(self, max_history=50):
        self.max_history = max_history
        self.conversations = {}
    
    def get_thread_config(self, thread_id: str):
        return {"configurable": {"thread_id": thread_id}}
    
    def cleanup_old_threads(self, keep_recent=10):
        """Keep only recent conversation threads"""
        if len(self.conversations) > keep_recent:
            # Keep only most recently used threads
            sorted_threads = sorted(
                self.conversations.items(),
                key=lambda x: x[1]['last_used'],
                reverse=True
            )
            self.conversations = dict(sorted_threads[:keep_recent])
```

## 🌟 What's Next?

Now that you understand multi-tool AI systems, you can:

- **Build specialized AI assistants** for specific domains (legal, medical, educational)
- **Create workflow automation systems** that chain multiple tools together
- **Develop API integrations** with various services and data sources
- **Build conversational interfaces** for complex business processes
- **Create adaptive AI systems** that learn which tools work best for different scenarios

Congratulations! You've mastered building sophisticated multi-tool AI assistants! 🎉

## 💡 Real-World Applications

### Business Intelligence:
- **Research assistant**: Wikipedia + news APIs + financial data tools
- **Customer service**: FAQ database + ticket system + escalation tools
- **Market analysis**: Web scraping + data analysis + visualization tools

### Education:
- **Homework helper**: Calculator + Wikipedia + citation checker + plagiarism detector
- **Language learning**: Translation + pronunciation + grammar checker + vocabulary tools
- **Research assistant**: Academic databases + citation formatter + fact checker

### Personal Productivity:
- **Travel planner**: Weather API + maps + booking systems + currency converter
- **Health tracker**: Fitness APIs + nutrition database + symptom checker + appointment scheduler
- **Financial advisor**: Bank APIs + investment tools + tax calculator + budget tracker

### Content Creation:
- **Writing assistant**: Grammar checker + fact verifier + citation tool + plagiarism detector
- **Social media manager**: Analytics + scheduling + content generator + hashtag optimizer
- **SEO optimizer**: Keyword research + competitor analysis + content analyzer + link checker

The multi-tool approach transforms AI from a single-purpose tool into a comprehensive digital assistant! 🚀