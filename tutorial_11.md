# Tutorial 11: AI Agents with Conversation Memory

## 🎯 What You'll Learn

In this tutorial, you'll learn how to:

- Add conversation memory to AI agents so they remember previous interactions
- Use chat history to create more natural, contextual conversations
- Handle follow-up questions that reference previous calculations
- Build agents that can maintain context across multiple interactions
- Understand the difference between stateless and stateful AI agents

Think of this as upgrading your AI assistant from having short-term memory loss to having a perfect memory of your entire conversation!

## 🤔 Script 10 vs Script 11: The Memory Upgrade

### Script 10 (No Memory):
```
You: "What's the hypotenuse of a triangle with sides 10 and 12?"
AI: "15.62"

You: "What about one with sides 3 and 4?"
AI: "5.0" (AI has no memory of the previous question)

You: "And double the first triangle's sides?"
AI: "I don't know what the first triangle was..." ❌
```

### Script 11 (With Memory):
```
You: "What's the hypotenuse of a triangle with sides 10 and 12?"
AI: "15.62" (remembers: first triangle was 10x12)

You: "What about one with sides 3 and 4?"
AI: "5.0" (remembers: second triangle was 3x4)

You: "And double the first triangle's sides?"
AI: "You mean 20 and 24? Let me calculate... 31.24" ✅
```

**The Power**: The AI remembers everything you've discussed and can reference previous calculations!

## 🧠 Why Conversation Memory Matters

### Real-World Benefits:
- **Natural conversations**: Ask follow-up questions without repeating context
- **Efficiency**: Don't need to re-explain previous calculations
- **Context awareness**: AI understands "the first triangle" or "that calculation"
- **Better user experience**: Feels like talking to a human assistant

### Use Cases:
- **Tutoring**: "Can you explain that formula again?" 
- **Problem solving**: "Now apply that same method to this new problem"
- **Consultations**: "Based on what we calculated earlier..."
- **Iterative design**: "Modify the previous calculation with these new parameters"

## 🔍 Understanding the Code: Line by Line

Let's examine `script_11.py` and focus on what's **NEW** compared to script_10.py:

### Step 1: New Imports for Memory

```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
```

**What's NEW here?**

1. **`HumanMessage, AIMessage`** - NEW!
   - **HumanMessage**: Represents what the user said
   - **AIMessage**: Represents what the AI responded
   - **Why needed**: To store conversation history in the correct format

2. **`ChatPromptTemplate`** - Different from `PromptTemplate`!
   - **Purpose**: Designed for multi-message conversations
   - **Vs PromptTemplate**: Can handle conversation history, not just single prompts

3. **`MessagesPlaceholder`** - NEW!
   - **Purpose**: Creates a slot where conversation history goes
   - **Think of it as**: A placeholder that says "insert chat history here"

### Step 2: Same Tool Definition

```python
@tool
def hypotenuse_length(input: str) -> float:
    """Calculates the hypotenuse of a right-angled triangle.
    Input format: 'a, b' (two side lengths separated by a comma)."""
    clean_input = input.strip().strip("'\"")
    sides = clean_input.split(',')
    if len(sides) != 2:
        raise ValueError("Please provide exactly two side lengths, e.g. '10, 12'.")
    a = float(sides[0].strip())
    b = float(sides[1].strip())
    return math.sqrt(a**2 + b**2)
```

**What's the same?**
- Tool definition is identical to script_10.py
- The math calculation logic hasn't changed
- Input cleaning and error handling are the same

**Why keep it the same?**
- The tool itself doesn't need to know about conversation history
- Tools are focused on specific tasks (calculating hypotenuse)
- Memory is handled at the agent level, not the tool level

### Step 3: Enhanced Chat Prompt Template

```python
# -----------------------------
# 3) ReAct CHAT prompt with history
# -----------------------------
react_chat_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful construction assistant. Use the available tools when a calculation is needed.

You have access to the following tools:
{tools}

Follow this EXACT format in your replies:
Thought: think about what to do next
Action: the single tool to use, exactly one of [{tool_names}]
Action Input: the input for the action
Observation: the result of the action
...(you can repeat Thought/Action/Observation)...
Thought: I can now answer
Final Answer: the final answer to the user's question

IMPORTANT:
- Never skip 'Final Answer:' when you give the final result.
- Only call tools using the Action/Action Input lines in the format above."""
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "Question: {input}"),
    ("assistant", "{agent_scratchpad}"),
])
```

**What's NEW here?** (This is the heart of memory!)

1. **`ChatPromptTemplate.from_messages([...])`**:
   - **Different from**: `PromptTemplate.from_template()` in script_10
   - **Why**: Designed for conversations with multiple message types
   - **Structure**: List of different message types instead of single template string

2. **Message Types**:
   ```python
   ("system", "...")      # Sets the AI's role and instructions
   MessagesPlaceholder(variable_name="chat_history"),  # Conversation history
   ("human", "Question: {input}"),                     # Current user question
   ("assistant", "{agent_scratchpad}"),                # AI's thinking space
   ```

3. **`MessagesPlaceholder(variable_name="chat_history")`** - THE KEY INNOVATION!
   - **What it does**: Creates a slot for conversation history
   - **Variable name**: Must match what we pass in later (`chat_history`)
   - **Position**: Comes after system message, before current question
   - **Think of it as**: "Insert all previous conversation here"

4. **Message Flow**:
   ```
   System: "You are a construction assistant..."
   [Previous conversation history gets inserted here]
   Human: "Question: What about sides 3 and 4?"
   Assistant: [AI's thinking workspace]
   ```

### Step 4: Enhanced Agent Executor

```python
# -----------------------------
# 4) Create agent + executor (add parsing recovery)
# -----------------------------
agent = create_react_agent(llm=llm, tools=tools, prompt=react_chat_prompt)

# ✅ Key fix: let the agent recover from output parsing issues
app = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,           # <--- add this
    max_iterations=6,                     # optional safety valves
    early_stopping_method="generate",     # optional: generate a final answer when hitting limits
)
```

**What's NEW here?**

1. **`handle_parsing_errors=True`** - NEW!
   - **Problem it solves**: Sometimes AI doesn't follow the exact format
   - **What it does**: Tries to recover and continue instead of crashing
   - **Why important**: Memory makes conversations more complex, more prone to format errors

2. **`max_iterations=6`** - NEW!
   - **Safety feature**: Prevents infinite loops
   - **Why needed**: Complex conversations might cause agents to get stuck
   - **What it means**: Agent will stop after 6 thinking steps

3. **`early_stopping_method="generate"`** - NEW!
   - **What it does**: If max iterations hit, AI generates a final answer anyway
   - **Alternative**: Could just stop and say "I don't know"
   - **Benefit**: More helpful to users even when things go wrong

### Step 5: Conversation Memory Implementation

```python
# -----------------------------
# 5) Run with REAL conversation history
# -----------------------------
chat_history = []
```

**What's NEW here?**

1. **`chat_history = []`** - THE MEMORY STORAGE!
   - **Purpose**: Stores the entire conversation
   - **Format**: List of `HumanMessage` and `AIMessage` objects
   - **Starts empty**: No previous conversation initially
   - **Grows**: Each interaction adds to this list

### Step 6: First Conversation with Memory Setup

```python
# (a) First query
first_query = "What is the hypotenuse length of a triangle with side lengths of 10 and 12?"
first_result = app.invoke({"input": first_query, "chat_history": chat_history})
print("\n--- Answer (first query) ---")
print(first_result["output"])

chat_history.append(HumanMessage(content=first_query))
chat_history.append(AIMessage(content=first_result["output"]))
```

**What's NEW here?** (This is where memory happens!)

1. **`app.invoke({"input": first_query, "chat_history": chat_history})`**:
   - **NEW parameter**: `"chat_history": chat_history`
   - **vs script_10**: Only had `{"input": query}`
   - **What it does**: Sends current question AND conversation history

2. **Memory Storage**:
   ```python
   chat_history.append(HumanMessage(content=first_query))
   chat_history.append(AIMessage(content=first_result["output"]))
   ```
   - **`HumanMessage`**: Stores what the user asked
   - **`AIMessage`**: Stores what the AI responded
   - **Order matters**: Human question, then AI response
   - **Building memory**: Each interaction adds to the history

3. **After first query, chat_history contains**:
   ```python
   [
       HumanMessage(content="What is the hypotenuse length of a triangle with side lengths of 10 and 12?"),
       AIMessage(content="The hypotenuse length is approximately 15.62.")
   ]
   ```

### Step 7: Using Memory for Context

```python
# (b) Comparison print
query = "What is the value of the hypotenuse for a triangle with sides 3 and 5?"
response = app.invoke({"input": query, "chat_history": chat_history})
print("\n--- Comparison print (input vs output) ---")
print({"user_input": query, "agent_output": response["output"]})

chat_history.append(HumanMessage(content=query))
chat_history.append(AIMessage(content=response["output"]))
```

**What's happening here?**

1. **AI sees the conversation context**:
   ```
   Previous conversation:
   Human: "What is the hypotenuse of triangle with sides 10 and 12?"
   AI: "15.62"
   
   Current question:
   Human: "What is the value of the hypotenuse for triangle with sides 3 and 5?"
   ```

2. **Memory keeps growing**:
   - Started with: `[]` (empty)
   - After first query: `[HumanMessage(...), AIMessage(...)]` (2 messages)
   - After second query: `[HumanMessage(...), AIMessage(...), HumanMessage(...), AIMessage(...)]` (4 messages)

### Step 8: Memory-Powered Follow-up Questions

```python
# (c) Follow-up using full history
new_query = "What about one with sides 12 and 14?"
follow_up = app.invoke({"input": new_query, "chat_history": chat_history})

# For teaching: show Human/AI-only log
_display_history = chat_history + [HumanMessage(content=new_query), AIMessage(content=follow_up["output"])]
filtered_messages = [
    f"{msg.__class__.__name__}: {msg.content}"
    for msg in _display_history
    if isinstance(msg, (HumanMessage, AIMessage)) and msg.content.strip()
]
print("\n--- Follow-up using history ---")
print({"user_input": new_query, "agent_output": filtered_messages})
```

**What's NEW here?**

1. **Context-aware question**: `"What about one with sides 12 and 14?"`
   - **"What about one"**: References the pattern of previous questions
   - **AI understands**: This is asking for another hypotenuse calculation
   - **Without memory**: AI would be confused by "what about one"
   - **With memory**: AI knows we're talking about triangles

2. **Complete conversation view**:
   ```python
   _display_history = chat_history + [HumanMessage(...), AIMessage(...)]
   ```
   - **Shows**: The entire conversation including the new exchange
   - **Format**: Easy-to-read format showing who said what
   - **Purpose**: Demonstrates how memory accumulates

## 🧠 What Happens Behind the Scenes?

### Memory Flow During Conversation:

**First Interaction:**
```
Input to AI:
- System: "You are a construction assistant..."
- Chat History: [] (empty)
- Human: "What is hypotenuse of triangle with sides 10 and 12?"
- Agent workspace: (empty)

AI processes → Returns: "15.62"
Memory stores: [HumanMessage("...10 and 12?"), AIMessage("15.62")]
```

**Second Interaction:**
```
Input to AI:
- System: "You are a construction assistant..."
- Chat History: [HumanMessage("...10 and 12?"), AIMessage("15.62")]
- Human: "What about triangle with sides 3 and 5?"
- Agent workspace: (empty)

AI sees previous conversation!
AI processes → Returns: "5.0"
Memory stores: [Previous messages + new messages]
```

**Third Interaction (Context-dependent):**
```
Input to AI:
- System: "You are a construction assistant..."
- Chat History: [All previous messages]
- Human: "What about one with sides 12 and 14?"
- Agent workspace: (empty)

AI understands "what about one" refers to triangles from context!
AI processes → Returns: "18.44"
```

## 🚀 How to Run This Code

### Prerequisites
1. **API key**: Set up your OpenAI API key in `.env`
2. **Dependencies**: Run `pip install -r requirements.txt`

### Steps
1. **Run the script**:
   ```bash
   python script_11.py
   ```

2. **What you'll see**:
   ```
   > Entering new AgentExecutor chain...
   I need to calculate the hypotenuse of a right-angled triangle with side lengths of 10 and 12.
   Action: hypotenuse_length
   Action Input: '10, 12'
   15.620499351813308
   Final Answer: The hypotenuse length is approximately 15.62.
   > Finished chain.
   
   --- Answer (first query) ---
   The hypotenuse length is approximately 15.62.
   
   --- Comparison print (input vs output) ---
   {'user_input': 'What is the value of the hypotenuse for a triangle with sides 3 and 5?', 'agent_output': '5.83'}
   
   --- Follow-up using history ---
   {'user_input': 'What about one with sides 12 and 14?', 
    'agent_output': ['HumanMessage: What is the hypotenuse length...', 'AIMessage: 15.62', ...]}
   ```

## 🎓 Key Concepts You've Learned

### Conversation Memory
- **What**: Storing and using previous interactions in AI conversations
- **Why**: Enables natural, contextual conversations
- **How**: Using `HumanMessage` and `AIMessage` objects in a list

### ChatPromptTemplate vs PromptTemplate
- **ChatPromptTemplate**: For multi-message conversations with history
- **PromptTemplate**: For single-prompt interactions
- **Key difference**: ChatPromptTemplate handles message types and history

### MessagesPlaceholder
- **What**: A slot in chat templates where conversation history goes
- **Why**: Allows dynamic insertion of varying amounts of history
- **How**: `MessagesPlaceholder(variable_name="chat_history")`

### Stateful vs Stateless Agents
- **Stateless (script_10)**: Each interaction is independent
- **Stateful (script_11)**: Each interaction builds on previous ones
- **Trade-offs**: Memory = more natural but more complex

### Error Handling for Complex Agents
- **handle_parsing_errors=True**: Recovers from format mistakes
- **max_iterations**: Prevents infinite loops
- **early_stopping_method**: Provides answers even when limits are hit

## 🔧 Common Issues and Solutions

**Problem: "KeyError: 'chat_history'"**
```python
# Wrong - forgetting to include chat_history
response = app.invoke({"input": query})

# Right - always include chat_history
response = app.invoke({"input": query, "chat_history": chat_history})
```

**Problem: "Memory keeps growing and slowing down"**
```python
# Solution: Limit memory to recent messages
MAX_MESSAGES = 10
if len(chat_history) > MAX_MESSAGES:
    chat_history = chat_history[-MAX_MESSAGES:]  # Keep only last 10 messages
```

**Problem: "AI doesn't remember correctly"**
```python
# Solution: Check message format
print("Current chat history:")
for msg in chat_history:
    print(f"{type(msg).__name__}: {msg.content}")
```

**Problem: "Agent gets confused with long conversations"**
```python
# Solution: Add conversation summary
def summarize_old_conversation(messages):
    # Keep recent messages, summarize older ones
    if len(messages) > 8:
        summary = "Previous conversation summary: User asked about triangles with sides 10,12 and 3,5."
        recent = messages[-6:]  # Keep last 6 messages
        return [AIMessage(content=summary)] + recent
    return messages

chat_history = summarize_old_conversation(chat_history)
```

## 🎯 Try These Experiments

### 1. Memory-Dependent Questions
```python
# Try questions that require memory:
queries = [
    "Calculate hypotenuse for triangle with sides 5 and 12",
    "What about sides 8 and 15?", 
    "Compare the first and second triangles",
    "Which triangle was larger?",
    "Double the sides of the smallest triangle"
]

for query in queries:
    response = app.invoke({"input": query, "chat_history": chat_history})
    print(f"Q: {query}")
    print(f"A: {response['output']}\n")
    
    # Update memory
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=response["output"]))
```

### 2. Multiple Tool Memory
```python
@tool
def rectangle_area(input: str) -> float:
    """Calculate rectangle area. Input: 'length, width'"""
    clean = input.strip().strip("'\"")
    dims = clean.split(',')
    return float(dims[0].strip()) * float(dims[1].strip())

tools = [hypotenuse_length, rectangle_area]

# Now ask mixed questions:
# "Calculate area of rectangle 5x10"
# "What about the hypotenuse of triangle with same dimensions?"
# "Which is bigger, the area or the hypotenuse?"
```

### 3. Conversation Branching
```python
def create_conversation_branch(base_history, new_question):
    """Create a new conversation branch without affecting main history"""
    branch_history = base_history.copy()
    response = app.invoke({"input": new_question, "chat_history": branch_history})
    
    branch_history.append(HumanMessage(content=new_question))
    branch_history.append(AIMessage(content=response["output"]))
    
    return branch_history, response["output"]

# Create different conversation paths
branch1, answer1 = create_conversation_branch(chat_history, "What about sides 1 and 1?")
branch2, answer2 = create_conversation_branch(chat_history, "What about sides 100 and 100?")
```

### 4. Conversation Persistence
```python
import json

def save_conversation(chat_history, filename):
    """Save conversation to file"""
    conversation_data = []
    for msg in chat_history:
        conversation_data.append({
            "type": msg.__class__.__name__,
            "content": msg.content
        })
    
    with open(filename, 'w') as f:
        json.dump(conversation_data, f, indent=2)

def load_conversation(filename):
    """Load conversation from file"""
    with open(filename, 'r') as f:
        conversation_data = json.load(f)
    
    chat_history = []
    for msg_data in conversation_data:
        if msg_data["type"] == "HumanMessage":
            chat_history.append(HumanMessage(content=msg_data["content"]))
        elif msg_data["type"] == "AIMessage":
            chat_history.append(AIMessage(content=msg_data["content"]))
    
    return chat_history

# Save conversation
save_conversation(chat_history, "math_conversation.json")

# Load conversation later
loaded_history = load_conversation("math_conversation.json")
```

## 🌟 Advanced Memory Techniques

### 1. Selective Memory
```python
def filter_important_messages(chat_history):
    """Keep only messages with calculations"""
    important = []
    for i in range(len(chat_history)-1):
        human_msg = chat_history[i]
        ai_msg = chat_history[i+1]
        
        if isinstance(human_msg, HumanMessage) and isinstance(ai_msg, AIMessage):
            # Keep if AI used a tool (calculation)
            if "Action:" in ai_msg.content or any(tool.name in ai_msg.content for tool in tools):
                important.extend([human_msg, ai_msg])
    
    return important
```

### 2. Context-Aware Prompting
```python
def generate_context_summary(chat_history):
    """Generate summary of conversation for context"""
    if not chat_history:
        return "No previous conversation."
    
    calculations = []
    for msg in chat_history:
        if isinstance(msg, AIMessage) and "hypotenuse" in msg.content.lower():
            calculations.append(msg.content)
    
    if calculations:
        return f"Previous calculations: {'; '.join(calculations[:3])}"
    return "Previous conversation about geometry calculations."
```

### 3. Memory Optimization
```python
class ConversationMemory:
    def __init__(self, max_messages=20, max_tokens=4000):
        self.chat_history = []
        self.max_messages = max_messages
        self.max_tokens = max_tokens
    
    def add_exchange(self, human_message, ai_message):
        self.chat_history.extend([
            HumanMessage(content=human_message),
            AIMessage(content=ai_message)
        ])
        self._trim_memory()
    
    def _trim_memory(self):
        # Trim by message count
        if len(self.chat_history) > self.max_messages:
            self.chat_history = self.chat_history[-self.max_messages:]
        
        # Trim by token count (approximate)
        total_tokens = sum(len(msg.content.split()) for msg in self.chat_history)
        while total_tokens > self.max_tokens and len(self.chat_history) > 2:
            self.chat_history = self.chat_history[2:]  # Remove oldest exchange
            total_tokens = sum(len(msg.content.split()) for msg in self.chat_history)
    
    def get_history(self):
        return self.chat_history

# Usage
memory = ConversationMemory()
```

## 📚 Memory Management Best Practices

### 1. Keep It Relevant
```python
# Good - focused conversation memory
relevant_history = [msg for msg in chat_history 
                   if "triangle" in msg.content.lower() or "hypotenuse" in msg.content.lower()]
```

### 2. Manage Memory Size
```python
# Good - prevent memory from growing too large
MAX_HISTORY_MESSAGES = 20
if len(chat_history) > MAX_HISTORY_MESSAGES:
    chat_history = chat_history[-MAX_HISTORY_MESSAGES:]
```

### 3. Clear Memory When Needed
```python
def clear_memory_on_topic_change(chat_history, new_query):
    """Clear memory if topic changes significantly"""
    if not chat_history:
        return chat_history
    
    # Simple topic detection
    geometry_words = ["triangle", "hypotenuse", "sides", "calculate"]
    last_msg = chat_history[-1].content.lower()
    new_query_lower = new_query.lower()
    
    last_is_geometry = any(word in last_msg for word in geometry_words)
    new_is_geometry = any(word in new_query_lower for word in geometry_words)
    
    if last_is_geometry and not new_is_geometry:
        return []  # Clear memory for topic change
    
    return chat_history
```

## 🌟 What's Next?

Now that you understand conversation memory, you can:

- **Build chatbots** that remember user preferences and context
- **Create tutoring systems** that track learning progress
- **Develop consultants** that build on previous discussions
- **Make multi-session applications** that save and restore conversations
- **Combine memory with RAG** for document-aware conversations

Congratulations! You can now create AI agents with perfect memory! 🎉

## 💡 Real-World Applications

### Education & Tutoring:
- **Math tutors**: Remember which concepts student struggles with
- **Language learning**: Track vocabulary and grammar progress
- **Coding mentors**: Reference previous coding problems and solutions

### Business & Consulting:
- **Financial advisors**: Remember client's financial goals and constraints
- **Project managers**: Track decisions and changes over time
- **Technical support**: Remember user's system and previous issues

### Healthcare & Therapy:
- **Medical consultations**: Track symptoms and treatment progress
- **Mental health**: Remember patient's concerns and coping strategies
- **Fitness coaching**: Track goals, progress, and preferences

### Personal Assistants:
- **Travel planning**: Remember preferences and constraints
- **Shopping**: Remember sizes, preferences, and budget
- **Scheduling**: Learn patterns and preferences over time

The ability to remember context transforms AI from a tool into a true assistant! 🚀