# 👔 Activity 09: Supervisor Multi-Agent Systems - Student Practice Guide

## 🎯 Learning Objectives

By the end of this activity, you will:
- Build a supervisor multi-agent system with centralized coordination
- Create specialized worker agents that report to a supervisor
- Use the `create_supervisor()` function for hierarchical agent management
- Understand the differences between supervisor and swarm architectures
- Learn to design effective delegation and communication patterns
- Master the "boss and workers" pattern for AI systems

## 📚 Background Context - Simplified!

**Think of it like a company:**
- **Supervisor** = Boss/Manager who assigns tasks
- **Worker Agents** = Employees with specific skills
- **User** = Customer making requests

**Your Journey So Far:**
- **Activity 01-07**: Single agent doing everything (one-person company)
- **Exercise 2.5**: Swarm agents talking to each other (equal partners)
- **Exercise 2.6**: Supervisor system (boss + workers) ← You are here

## 🏢 Simple Architecture Comparison

### Single Agent (Previous Exercises)
```
User → [One Agent with all tools] → User
```
- Like a freelancer doing everything alone
- ❌ Gets overwhelmed with complex tasks
- ❌ Hard to specialize and optimize

### Swarm (Exercise 2.5)
```
User → [Agent A] ↔ [Agent B] → User
```
- Like equal business partners
- ✅ Agents collaborate directly
- ❌ Can get confusing who does what

### Supervisor (This Exercise)
```
User → [Supervisor] → [Worker A or B] → [Supervisor] → User
      ↓                                      ↑
   Assigns task                        Collects result
```
- Like a company with clear hierarchy
- ✅ **Clear chain of command**
- ✅ **Boss coordinates everything**
- ✅ **Workers focus on their specialty**

## 🧠 Key Concepts - Made Simple

### What is a Supervisor Agent?

**Think of a project manager:**
1. **Receives requests** from users (like project requirements)
2. **Decides which worker** should handle each task
3. **Assigns work** to the right specialist
4. **Collects results** from workers
5. **Gives final answer** to the user

### Worker Agents

**Like specialized employees:**
- **Researcher**: Good at finding information (Wikipedia, stock data)
- **Analyst**: Good at creating charts and running code
- **They only do their job** and report back to supervisor
- **No direct communication** between workers

### Simple Benefits

**Why use supervisor pattern?**
- 📋 **Clear organization** - Everyone knows their role
- 🎯 **Better coordination** - One agent manages workflow
- 📊 **Quality control** - Supervisor reviews all work
- 🔄 **Easy to expand** - Just add new workers to supervisor
- 🐛 **Easier debugging** - Clear path of who did what

## 🔧 Setup Instructions

### Step 1: Install Required Libraries

```bash
pip install --quiet wikipedia==1.4.0 langchain-core==0.3.69 langgraph==0.5.3 langchain-openai==0.3.28 langchain-experimental==0.3.4 langgraph-supervisor==0.0.27
```

### Step 2: Import Dependencies and Define Tools

```python
# Tools are pre-defined (same as previous exercises)
# Focus will be on supervisor architecture
```

## 🏗️ Building Your Supervisor System

### Step 3: Create Specialized Worker Agents

**Your task:** Create worker agents that follow supervisor instructions. You need to write about **60%** of the implementation.

```python
# TODO: Import required modules
from langchain_openai import ________
from langgraph.prebuilt import ________

llm = ChatOpenAI(model="________")

# TODO: Step 3a - Create research worker agent
# This agent ONLY does research tasks and reports to supervisor
research_agent = ________(
    ________,  # LLM
    tools=[________, ________],  # Wikipedia and stock data tools
    prompt=(
        "You are a research agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with research-related tasks, including looking-up factual information and stock data. DO NOT write any code.\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="________"  # Agent name for supervisor to use
)

# TODO: Step 3b - Create analyst worker agent  
# This agent ONLY does Python coding/visualization tasks
analyst_agent = ________(
    ________,  # LLM
    [________],  # Only Python REPL tool
    prompt=(
        "You are an agent that can run arbitrary Python code.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with tasks that require running code to produce an output.\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="________"  # Agent name for supervisor to use
)
```

<details>
<summary>💡 Step 3 Hints</summary>

**Step 3a - Research Agent:**
- LLM: `llm`
- Tools: `[wikipedia_tool, stock_data_tool]`
- Name: `"researcher"`

**Step 3b - Analyst Agent:**
- LLM: `llm`
- Tools: `[python_repl_tool]`
- Name: `"analyst"`

</details>

### Step 4: Create the Supervisor Agent

**Your task:** Build the supervisor that manages both worker agents.

```python
# TODO: Step 4a - Import supervisor components
from langgraph_supervisor import ________
from langgraph.checkpoint.memory import ________

# TODO: Step 4b - Set up memory and configuration  
config = {"configurable": {"thread_id": "1", "user_id": "1"}}
checkpointer = ________()

# TODO: Step 4c - Create the supervisor
# The supervisor decides which worker to use for each task
supervisor = ________(
    model=________,  # LLM for supervisor
    agents=[________, ________],  # List of worker agents
    prompt=(
        "You are a supervisor managing two agents:\n"
        "- a research agent. Assign research and data collection tasks to this agent\n"
        "- an analyst agent. Assign the creation of visualizations via code to this agent\n"
        "Assign work to one agent at a time, do not call agents in parallel.\n"
        "Do not do any work yourself."
    ),
    add_handoff_back_messages=True,  # Show when workers report back
    output_mode="full_history",      # Keep track of all conversations
).compile(checkpointer=________)
```

<details>
<summary>🔍 Step 4 Hints</summary>

**Step 4a:** Import `create_supervisor` and `InMemorySaver`
**Step 4b:** Use `InMemorySaver()`  
**Step 4c:**
- model: `llm`
- agents: `[research_agent, analyst_agent]`
- checkpointer: `checkpointer`

</details>

### Step 5: Test Your Supervisor System

```python
# TODO: Step 5a - Visualize the system
# This shows the supervisor and worker relationship
________

# TODO: Step 5b - Test different types of requests
from course_helper_functions import pretty_print_messages

# Test 1: Research task (should go to researcher)
for chunk in supervisor.stream(
    {"messages": [{"role": "user", "content": "Who is Apple's CEO?"}]}, config
):
    pretty_print_messages(chunk)

# Test 2: Analysis task (should go to analyst)  
for chunk in supervisor.stream(
    {"messages": [{"role": "user", "content": "Create a chart showing numbers 1 to 5"}]}, config
):
    pretty_print_messages(chunk)

# Test 3: Complex task (should use both workers)
for chunk in supervisor.stream(
    {"messages": [{"role": "user", "content": "Research Tesla and create a stock price chart"}]}, config
):
    pretty_print_messages(chunk)
```

## ✅ Expected Behavior - Simplified

### How Your System Should Work

**Simple Request Flow:**
```
1. User asks: "Who is Apple's CEO?"
2. Supervisor thinks: "This needs research"
3. Supervisor assigns: researcher agent
4. Researcher finds: CEO information  
5. Researcher reports back: to supervisor
6. Supervisor responds: to user with answer
```

**Complex Request Flow:**
```
1. User asks: "Research Apple and make a chart"
2. Supervisor thinks: "This needs research AND charts"  
3. Supervisor assigns: researcher first
4. Researcher finds: Apple information
5. Researcher reports: data to supervisor
6. Supervisor assigns: analyst next
7. Analyst creates: chart using data
8. Analyst reports: chart to supervisor
9. Supervisor responds: complete answer to user
```

### Task Assignment Patterns

**Research Tasks → Researcher:**
- ✅ "Tell me about Microsoft"
- ✅ "Get Apple stock data" 
- ✅ "Who founded Tesla?"

**Analysis Tasks → Analyst:**
- ✅ "Create a chart of [1,2,3,4,5]"
- ✅ "Plot some data"
- ✅ "Run this Python code"

**Complex Tasks → Both Workers:**
- ✅ "Research Apple and visualize their stock"
- ✅ "Get Microsoft data and create charts"

## 🎓 Understanding Your Code - Simplified

### Key Components Explained

**1. Worker Agent Design:**
```python
research_agent = create_react_agent(
    llm,
    tools=[wikipedia_tool, stock_data_tool],
    prompt="You are a research agent...",  # Clear role definition
    name="researcher"  # How supervisor identifies this agent
)
```
- **Focused tools**: Each agent only has tools for their specialty
- **Clear instructions**: Prompts tell agents exactly what to do
- **Report back**: Agents return results to supervisor, not user

**2. Supervisor Design:**
```python
supervisor = create_supervisor(
    model=llm,
    agents=[research_agent, analyst_agent],
    prompt="You are a supervisor managing two agents...",
)
```
- **Management role**: Supervisor doesn't do work, only delegates
- **Agent awareness**: Knows what each worker agent can do
- **Coordination**: Manages workflow and collects results

**3. Simple Message Flow:**
```python
# What happens inside the system:
User → Supervisor → Worker → Supervisor → User
      ↓           ↓         ↑           ↑
   "Research"  Does work  Reports    Gives final
    task                 back      answer
```

### Supervisor vs Swarm Comparison

**Supervisor (This Exercise):**
```python
# Hierarchical: Boss → Worker → Boss → User
User → Supervisor → Worker → Supervisor → User
```
- ✅ **Clear control** - One agent manages everything
- ✅ **Organized workflow** - Predictable task assignment
- ✅ **Quality review** - Supervisor checks all work
- ❌ **Single point of failure** - If supervisor fails, system fails

**Swarm (Previous Exercise):**  
```python  
# Peer-to-peer: Worker ↔ Worker
User → Worker A ↔ Worker B → User
```
- ✅ **Direct collaboration** - Workers talk directly
- ✅ **Flexible** - No single control point
- ❌ **Coordination complexity** - Workers must agree on workflow
- ❌ **Potential confusion** - Who's in charge?

## 🔧 Troubleshooting Guide - Simplified

### Common Issues & Easy Fixes

**❌ "Supervisor doesn't assign tasks properly"**
```python
# Problem: Vague supervisor prompt
prompt="Help the user"  # Too general!

# Solution: Clear task assignment instructions
prompt=(
    "You are a supervisor managing two agents:\n"
    "- research agent: for finding information\n" 
    "- analyst agent: for creating charts and code\n"
    "Assign the right agent for each task."
)
```

**❌ "Worker agents don't report back"**
```python
# Problem: Missing report-back instruction
prompt="You do research."  # Doesn't say what to do after!

# Solution: Clear reporting instructions
prompt=(
    "You are a research agent.\n"
    "After you're done, respond to the supervisor directly.\n"
    "Respond ONLY with results, no extra text."
)
```

**❌ "Workers do each other's jobs"**
```python
# Problem: Overlapping tool assignments
research_tools = [wikipedia_tool, python_repl_tool]  # Research agent has coding tool!

# Solution: Clear separation
research_tools = [wikipedia_tool, stock_data_tool]   # Only research tools
analyst_tools = [python_repl_tool]                  # Only coding tools
```

**❌ "create_supervisor not found"**
```python
# Solution: Import from correct module
from langgraph_supervisor import create_supervisor
```

## 🧪 Simple Testing Challenges

### Challenge 1: Task Assignment Validation

Test that your supervisor assigns the right workers:

```python
# Create simple test cases
simple_tests = [
    # (request, expected_worker, task_type)
    ("Tell me about Apple", "researcher", "simple_research"),
    ("Create a chart of [1,2,3]", "analyst", "simple_analysis"), 
    ("Who is Tesla's CEO?", "researcher", "simple_research"),
    ("Plot some data points", "analyst", "simple_analysis"),
]

def test_task_assignment():
    """Test that supervisor assigns tasks to correct workers"""
    for request, expected_worker, task_type in simple_tests:
        print(f"\n--- Testing: {request} ---")
        print(f"Expected worker: {expected_worker}")
        
        # Run the request
        chunks = list(supervisor.stream({
            "messages": [{"role": "user", "content": request}]
        }, config))
        
        # Find which worker was used
        workers_used = []
        for chunk in chunks:
            if "messages" in chunk:
                for message in chunk["messages"]:
                    if message.get("name") in ["researcher", "analyst"]:
                        if message["name"] not in workers_used:
                            workers_used.append(message["name"])
        
        print(f"Workers used: {workers_used}")
        
        # Check if correct worker was assigned
        if expected_worker in workers_used:
            print("✅ Correct assignment!")
        else:
            print("❌ Wrong assignment!")

# Run the test
test_task_assignment()
```

### Challenge 2: Multi-Step Workflow Testing

Test complex tasks that need multiple workers:

```python
def test_complex_workflows():
    """Test tasks that require multiple workers in sequence"""
    
    complex_requests = [
        "Research Apple Inc. and create a simple chart",
        "Get Tesla stock data and visualize it", 
        "Find Microsoft's CEO and make a bar chart of numbers [1,2,3,4,5]"
    ]
    
    for request in complex_requests:
        print(f"\n--- Complex Test: {request} ---")
        
        chunks = list(supervisor.stream({
            "messages": [{"role": "user", "content": request}]
        }, config))
        
        # Track the workflow
        workers_used = []
        supervisor_messages = 0
        
        for chunk in chunks:
            if "messages" in chunk:
                for message in chunk["messages"]:
                    # Count supervisor decisions
                    if message.get("role") == "assistant" and not message.get("name"):
                        if "content" in message:
                            supervisor_messages += 1
                    
                    # Track worker usage
                    if message.get("name") in ["researcher", "analyst"]:
                        workers_used.append(message["name"])
        
        print(f"Workers involved: {set(workers_used)}")
        print(f"Supervisor coordination steps: {supervisor_messages}")
        
        # For complex tasks, we expect multiple workers
        if len(set(workers_used)) > 1:
            print("✅ Multi-agent collaboration!")
        else:
            print("ℹ️ Single agent task")

# Run complex workflow test
test_complex_workflows()
```

### Challenge 3: Simple Response Quality Check

```python
def test_response_quality():
    """Check if supervisor provides good final responses"""
    
    quality_tests = [
        ("Who founded Apple?", "should mention Steve Jobs"),
        ("Create a simple chart", "should show chart creation"),
    ]
    
    for request, expectation in quality_tests:
        print(f"\n--- Quality Test: {request} ---")
        print(f"Should: {expectation}")
        
        chunks = list(supervisor.stream({
            "messages": [{"role": "user", "content": request}]
        }, config))
        
        # Find supervisor's final response to user
        final_response = ""
        for chunk in chunks:
            if "messages" in chunk:
                for message in chunk["messages"]:
                    # Look for supervisor's final response (not from worker)
                    if (message.get("role") == "assistant" and 
                        not message.get("name") and 
                        "content" in message):
                        final_response = message["content"]
        
        print(f"Final response length: {len(final_response)} characters")
        if final_response:
            print(f"Preview: {final_response[:100]}...")
        
        # Simple quality check
        if len(final_response) > 50:  # Has substantial content
            print("✅ Good response length")
        else:
            print("❌ Response too short")

# Run quality test
test_response_quality()
```

## 🚀 Simple Extensions

### Extension 1: Add a Third Worker

```python
# Create a new specialist - Email agent
@tool 
def email_tool(recipient: Annotated[str, "Email recipient"], message: Annotated[str, "Message to send"]):
    """Send an email notification."""
    return f"Email sent to {recipient}: {message}"

# Create email worker
email_agent = create_react_agent(
    llm,
    [email_tool],
    prompt=(
        "You are an email agent.\n"
        "Send emails when requested.\n"
        "Report back to supervisor with confirmation."
    ),
    name="email_specialist"
)

# Create supervisor with three workers
three_worker_supervisor = create_supervisor(
    model=llm,
    agents=[research_agent, analyst_agent, email_agent],
    prompt=(
        "You are a supervisor managing three agents:\n"
        "- research agent: for information gathering\n"
        "- analyst agent: for charts and code\n" 
        "- email agent: for sending notifications\n"
        "Assign the right agent for each task."
    ),
    add_handoff_back_messages=True,
    output_mode="full_history",
).compile(checkpointer=InMemorySaver())
```

### Extension 2: Simple Performance Monitoring

```python
import time

def monitor_supervisor_performance():
    """Simple performance monitoring for supervisor system"""
    
    test_requests = [
        "Tell me about Apple",
        "Create a simple chart", 
        "Research Tesla and make a visualization"
    ]
    
    results = []
    
    for request in test_requests:
        print(f"\n--- Performance Test: {request} ---")
        
        start_time = time.time()
        
        chunks = list(supervisor.stream({
            "messages": [{"role": "user", "content": request}]
        }, config))
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Count workers used
        workers_used = set()
        for chunk in chunks:
            if "messages" in chunk:
                for message in chunk["messages"]:
                    if message.get("name") in ["researcher", "analyst"]:
                        workers_used.add(message["name"])
        
        result = {
            "request": request,
            "time": execution_time,
            "workers": list(workers_used)
        }
        results.append(result)
        
        print(f"Time: {execution_time:.2f}s")
        print(f"Workers: {list(workers_used)}")
    
    # Summary
    avg_time = sum(r["time"] for r in results) / len(results)
    print(f"\n=== Performance Summary ===")
    print(f"Average time: {avg_time:.2f} seconds")
    print(f"Total tests: {len(results)}")

# Run performance monitoring
monitor_supervisor_performance()
```

## 📝 Self-Assessment - Simplified

**Check your understanding:**

□ I can create worker agents with specific roles and tools
□ I understand how a supervisor delegates tasks to workers
□ I can use `create_supervisor()` to build hierarchical systems
□ I know the difference between supervisor and swarm patterns
□ I can test task assignment and workflow coordination
□ I understand when to use supervisor vs other architectures

## 💡 Real-World Applications - Simple Examples

**Customer Service Company:**
- **Supervisor**: Main customer service manager
- **Worker 1**: Technical support specialist
- **Worker 2**: Billing and account specialist  
- **Worker 3**: Product information specialist

**Content Creation Agency:**
- **Supervisor**: Project manager
- **Worker 1**: Researcher (gathers information)
- **Worker 2**: Writer (creates content)
- **Worker 3**: Designer (creates visuals)

**Financial Advisory Firm:**
- **Supervisor**: Senior advisor
- **Worker 1**: Market researcher
- **Worker 2**: Risk analyst
- **Worker 3**: Report generator

## 🎉 Congratulations!

You've built your first supervisor multi-agent system! This system can:

- ✅ **Delegate tasks intelligently** to the right specialists
- ✅ **Coordinate complex workflows** across multiple workers  
- ✅ **Maintain quality control** through centralized supervision
- ✅ **Scale easily** by adding new worker agents
- ✅ **Provide clear organization** with hierarchical structure

**Key Takeaways - Simplified:**
- **Supervisor = Boss** who assigns work and coordinates
- **Workers = Specialists** who do specific tasks and report back
- **Clear hierarchy** makes complex tasks manageable
- **Task delegation** improves efficiency and quality
- **Centralized control** provides better coordination than peer-to-peer

## 🚀 Next Steps

After completing this exercise:

1. **Multi-layer Supervisors**: Supervisor managing other supervisors
2. **Dynamic Worker Creation**: Adding workers based on task needs
3. **Human-in-the-Loop**: Supervisors that can escalate to humans
4. **Production Deployment**: Enterprise supervisor systems with monitoring

## 🔄 Architecture Evolution

**Your Multi-Agent Journey:**
- **Exercise 2.5**: Swarm (peer-to-peer collaboration)
- **Exercise 2.6**: Supervisor (hierarchical management) ← You are here
- **Future**: Multi-layer supervision, hybrid patterns

## 🧠 Simple Architecture Benefits

**Supervisor vs Other Patterns:**

| Pattern | Control | Coordination | Complexity | Best For |
|---------|---------|-------------|------------|----------|
| **Single Agent** | Simple | None needed | Low | Simple tasks |
| **Swarm** | Distributed | Peer-to-peer | Medium | Equal collaboration |
| **Supervisor** | Centralized | Top-down | Medium | Clear hierarchy |

**When to Use Supervisor:**
- ✅ **Clear task boundaries** between different types of work
- ✅ **Quality control** is important
- ✅ **Workflow coordination** is needed
- ✅ **Scalable organization** is desired

You've mastered hierarchical AI systems! Ready for advanced coordination patterns and enterprise applications! 🚀👔✨