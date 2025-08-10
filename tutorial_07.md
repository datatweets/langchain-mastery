# Tutorial 7: Creating AI Agents That Can Use Tools

## 🎯 What You'll Learn

In this tutorial, you'll learn how to:

- Create AI agents that can think and make decisions
- Give AI access to external tools (like Wikipedia)
- Understand the ReAct framework (Reasoning + Acting)
- Build AI that can research and answer questions with real data

Think of this as creating an AI research assistant that can look up information, think about it, and give you well-researched answers!

## 🤔 Regular AI vs. AI Agents

### Regular AI (What we did before):
```
You: "How many people live in New York City?"
AI: "Based on my training data, approximately 8.3 million people..." 
(Limited to training data, might be outdated)
```

### AI Agent (What we're learning now):
```
You: "How many people live in New York City?"

AI Agent thinks: "I need current information about NYC population"
AI Agent: *searches Wikipedia for "New York City"*
AI Agent: *finds current population data*
AI Agent: "According to the latest data from Wikipedia, New York City has approximately 8.3 million residents as of 2023..."
(Uses current, verified information!)
```

**The Power**: Agents can access real-time information and use tools to give better answers!

## 🧠 What is the ReAct Framework?

**ReAct = Reasoning + Acting**

The AI follows this cycle:
1. **Think** about what it needs to do
2. **Act** by using a tool (like searching Wikipedia)
3. **Observe** the results from the tool
4. **Think** about what to do next
5. Repeat until it has enough information

## 🔍 Understanding the Code: Line by Line

Let's examine `script_07.py` step by step:

### Step 1: Importing Agent Tools

```python
from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI
from langchain import hub  # pulls a ready-made ReAct prompt from LangChain Hub
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import create_react_agent, AgentExecutor
```

**What's happening here?**

1. **`from langchain import hub`** - NEW!
   - LangChain Hub is like a library of pre-built prompts
   - We'll download a ready-made ReAct prompt from there

2. **`from langchain_community.agent_toolkits.load_tools import load_tools`** - NEW!
   - This helps us give the AI access to external tools
   - Tools are like apps the AI can use (Wikipedia, calculator, etc.)

3. **`from langchain.agents import create_react_agent, AgentExecutor`** - NEW!
   - `create_react_agent`: Creates an AI that can reason and act
   - `AgentExecutor`: Runs the agent and manages the tool usage

### Step 2: Loading Environment Variables

```python
# 1) Load API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "false"
```

**What's happening here?**

- Same as before - loading our OpenAI API key
- `os.environ["LANGCHAIN_TRACING_V2"] = "false"` disables logging (optional)

### Step 3: Creating the AI Model

```python
# 2) LLM (gpt-4o)
llm = ChatOpenAI(
    model="gpt-4o",
    api_key=api_key,
    temperature=0  # deterministic answers for fact lookups
)
```

**What's happening here?**

1. **`temperature=0`** - Very important for agents!
   - 0 = Consistent, factual responses
   - Agents need to be reliable when using tools
   - We want factual research, not creative writing

### Step 4: Loading Tools for the Agent

```python
# 3) Tools (Wikipedia)
tools = load_tools(["wikipedia"])
```

**What's happening here?**

1. **`tools = load_tools(["wikipedia"])`**
   - Gives our AI agent access to Wikipedia
   - The AI can now search Wikipedia for information
   - Other available tools: calculator, web search, file readers, etc.

2. **What Wikipedia tool provides**:
   - Search Wikipedia articles
   - Get summaries of articles
   - Access current, factual information

### Step 5: Getting the ReAct Prompt

```python
# 4) ReAct prompt (standard one from LangChain Hub)
react_prompt = hub.pull("hwchase17/react")
```

**What's happening here?**

1. **`hub.pull("hwchase17/react")`**
   - Downloads a pre-built ReAct prompt from LangChain Hub
   - "hwchase17" is the creator (Harrison Chase, founder of LangChain)
   - This prompt teaches the AI how to think and use tools

2. **What's in the ReAct prompt?**
   - Instructions on how to reason step-by-step
   - How to use available tools
   - When to give a final answer

### Step 6: Creating the Agent

```python
# 5) Build agent + executor
agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
```

**What's happening here?**

1. **`create_react_agent(...)`**
   - Creates our thinking AI agent
   - Combines: AI model + tools + ReAct prompt
   - The agent knows how to reason and act

2. **`AgentExecutor(...)`**
   - This is like a manager for our agent
   - Runs the agent safely
   - Manages tool usage and prevents infinite loops
   - `verbose=True` shows us the thinking process

### Step 7: Asking a Question

```python
# 6) Ask the question (invoke the agent)
question = "How many people live in New York City?"
result = agent_executor.invoke({"input": question})

# Result keys typically include: 'input', 'output', 'intermediate_steps'
print(result["output"])
```

**What's happening here?**

1. **`question = "How many people live in New York City?"`**
   - Our research question for the agent

2. **`agent_executor.invoke({"input": question})`**
   - Starts the agent thinking and acting process
   - The agent will use Wikipedia to research the answer

3. **`result["output"]`**
   - Gets the final answer after all the research
   - The result also contains intermediate steps (the thinking process)

## 🧠 What Happens Behind the Scenes?

When you run this code, here's the agent's thinking process:

### Step 1: Agent Receives Question
```
Human: How many people live in New York City?
```

### Step 2: Agent Thinks (Reasoning)
```
Thought: I need to find current population data for New York City. 
I should search Wikipedia for this information.
```

### Step 3: Agent Acts (Using Tools)
```
Action: Wikipedia
Action Input: New York City population
```

### Step 4: Agent Observes (Tool Results)
```
Observation: New York City is the most populous city in the United States, 
with an estimated population of 8,336,817 as of 2022...
```

### Step 5: Agent Thinks Again
```
Thought: I found the population information. I can now provide a final answer.
```

### Step 6: Agent Gives Final Answer
```
Final Answer: Based on the latest data from Wikipedia, New York City has 
a population of approximately 8.34 million people as of 2022.
```

## 🚀 How to Run This Code

1. **Make sure your API key is set up**
   ```bash
   # Your .env file should contain:
   OPENAI_API_KEY=sk-your-actual-key-here
   ```

2. **Run the script**
   ```bash
   python script_07.py
   ```

3. **What you'll see**
   ```
   > Entering new AgentExecutor chain...
   I need to find current population data for New York City.
   
   Action: Wikipedia
   Action Input: New York City population
   
   Observation: New York City is the most populous city in the United States...
   
   Thought: I now have the information needed to answer the question.
   
   Final Answer: Based on Wikipedia, New York City has a population of approximately 8.34 million people as of 2022.
   
   > Finished chain.
   
   Based on Wikipedia, New York City has a population of approximately 8.34 million people as of 2022.
   ```

## 🎓 Key Concepts You've Learned

### AI Agents
- **What**: AI that can think, plan, and use tools
- **Why**: Provides current information and can perform research
- **How**: Combines reasoning (thinking) with actions (tool use)

### ReAct Framework
- **What**: A method for AI reasoning: Think → Act → Observe → Repeat
- **Why**: Gives AI a systematic approach to problem-solving
- **How**: Uses structured prompts that guide the thinking process

### Tools for AI
- **What**: External services AI can access (Wikipedia, calculator, web search)
- **Why**: Extends AI capabilities beyond training data
- **How**: Pre-built integrations that AI can invoke

### Agent Executor
- **What**: A manager that safely runs AI agents
- **Why**: Prevents infinite loops and manages tool usage
- **How**: Controls the agent's thinking and acting cycle

### LangChain Hub
- **What**: A repository of pre-built prompts and chains
- **Why**: Saves time and provides tested, optimized prompts
- **How**: Download with `hub.pull("creator/prompt-name")`

## 🔧 Common Issues and Solutions

**Problem: "Wikipedia tool not working"**
```bash
# Solution: Make sure you have the required packages
pip install wikipedia
```

**Problem: "Agent keeps searching forever"**
```python
# Solution: Add max iterations to agent executor
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True,
    max_iterations=5  # Stops after 5 steps
)
```

**Problem: "Can't pull from hub"**
```python
# Solution: Check internet connection and try alternative
# You can create your own prompt if hub is unavailable
from langchain.prompts import PromptTemplate

react_prompt = PromptTemplate.from_template("""
Answer the following questions as best you can. You have access to these tools:
{tools}

Use this format:
Question: {input}
Thought: [think about what to do]
Action: [tool name]  
Action Input: [what to search for]
Observation: [tool result]
... (repeat Thought/Action/Observation as needed)
Final Answer: [your final response]

Question: {input}
{agent_scratchpad}
""")
```

## 🎯 Try These Experiments

### 1. Different Types of Questions
```python
# Current events
question = "What is the latest population of Tokyo?"

# Factual research
question = "When was the Eiffel Tower built and how tall is it?"

# Complex research
question = "Compare the populations of the 3 largest cities in Japan"
```

### 2. Multiple Tools
```python
# Add more tools for the agent
tools = load_tools(["wikipedia", "llm-math"])  # Math calculator too

# Now the agent can do math and research!
question = "What's the population density of Singapore? (population divided by area)"
```

### 3. Custom Questions with Different Topics
```python
questions = [
    "What is the capital of the newest country in the world?",
    "How many Nobel Peace Prize winners are there?",
    "What's the highest mountain in South America and how tall is it?"
]

for question in questions:
    result = agent_executor.invoke({"input": question})
    print(f"Q: {question}")
    print(f"A: {result['output']}\n")
```

### 4. Agent with Memory
```python
# Create an agent that remembers previous conversations
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(return_messages=True)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,  # Now it remembers!
    verbose=True
)
```

## 🌟 Advanced Agent Techniques

### Custom Tools
```python
from langchain.tools import Tool

def calculate_age(birth_year):
    current_year = 2024
    return current_year - int(birth_year)

age_tool = Tool(
    name="Age Calculator",
    description="Calculate someone's age given their birth year",
    func=calculate_age
)

tools = [age_tool] + load_tools(["wikipedia"])
```

### Agent with Different Personalities
```python
# Modify the system message in the prompt
friendly_prompt = hub.pull("hwchase17/react").partial(
    system_message="You are a friendly and enthusiastic research assistant."
)

professional_prompt = hub.pull("hwchase17/react").partial(
    system_message="You are a professional, concise research analyst."
)
```

### Specialized Agents
```python
# Science research agent
science_tools = load_tools(["wikipedia", "arxiv"])  # Academic papers
science_agent = create_react_agent(llm, science_tools, react_prompt)

# Financial research agent  
finance_tools = load_tools(["wikipedia", "alpha_vantage"])  # Stock data
finance_agent = create_react_agent(llm, finance_tools, react_prompt)
```

## 📚 Agent Best Practices

### 1. Choose the Right Tools
```python
# For factual questions: Wikipedia
tools = load_tools(["wikipedia"])

# For math problems: Calculator
tools = load_tools(["llm-math"])

# For current web info: Search
tools = load_tools(["google-search"])
```

### 2. Set Appropriate Limits
```python
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=3,      # Prevent infinite loops
    max_execution_time=60, # Timeout after 1 minute
    verbose=True
)
```

### 3. Use Low Temperature for Research
```python
# Good for agents - factual and consistent
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Bad for agents - too creative
llm = ChatOpenAI(model="gpt-4o", temperature=0.9)
```

## 🚫 What Agents Can't Do

### Limitations to Remember:
- **No real-time web browsing** (unless you add web search tools)
- **Can't perform actions in the real world** (can't send emails, make purchases)
- **Limited by available tools** (can only use tools you provide)
- **Can be slower** (multiple API calls for complex questions)
- **May not always find information** (depends on tool quality)

## 🌟 What's Next?

Now that you understand AI agents, you're ready to learn about:

- **RAG Systems** (Tutorial 8) - AI that can read and analyze your documents
- **Advanced RAG** (Tutorial 9) - Sophisticated document analysis with vector databases

Congratulations! You've created an AI agent that can research and think! 🎉

## 💡 Real-World Applications

- **Research Assistants**: Gather information from multiple sources
- **Customer Support**: Look up product information and policies
- **Educational Tools**: Help students research topics with verified sources
- **Business Intelligence**: Analyze market data and trends
- **Content Creation**: Research facts for articles and reports