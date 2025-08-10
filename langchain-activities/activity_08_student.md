# 🐝 Activity 08: Multi-Agent Swarm Systems - Student Practice Guide

## 🎯 Learning Objectives

By the end of this activity, you will:
- Build your first multi-agent system using the "swarm" architecture
- Create handoff tools for agent-to-agent communication
- Understand agent specialization and collaboration patterns
- Learn to use `create_handoff_tool()` and `create_swarm()` functions
- Explore the benefits and limitations of swarm-based multi-agent systems
- Master memory management in multi-agent environments

## 📚 Background Context

**Evolution of Agent Systems:**
- **Activity 01-03**: Individual tool creation
- **Exercise 1.6-1.7**: Single agent with tool integration  
- **Exercise 2.2**: Conditional routing in single agents
- **Exercise 2.3**: High-level single agent creation
- **Exercise 2.5**: Multi-agent swarm systems ← You are here

**Why Multi-Agent Systems?**

**Single Agent Limitations:**
- ❌ **Jack of all trades**: One agent trying to do everything
- ❌ **Complexity**: Complex system prompts trying to handle all scenarios
- ❌ **Performance**: Slower decision-making with many tools
- ❌ **Maintenance**: Harder to update and optimize single complex agent

**Multi-Agent Benefits:**
- ✅ **Specialization**: Each agent excels at specific tasks
- ✅ **Scalability**: Add new agents for new capabilities
- ✅ **Modularity**: Easy to update individual agents
- ✅ **Collaboration**: Agents can work together on complex tasks
- ✅ **Performance**: Focused agents make faster decisions

## 🧠 Swarm Architecture Concepts

### What is a Swarm?

A **swarm** is a multi-agent architecture where:
1. **Specialized Agents**: Each agent has specific tools and expertise
2. **Handoff Tools**: Agents can transfer control to other agents
3. **Peer Communication**: Agents communicate directly with each other
4. **Shared Memory**: All agents share conversation history and context

### Swarm vs Other Architectures

**Swarm Architecture:**
```
User → [Agent A] ↔ [Agent B] ↔ [Agent C] → User
       ↘         ↙         ↘        ↙
         Handoff Tools & Shared Memory
```

**Supervisor Architecture (Future Exercise):**
```
User → [Supervisor] → [Agent A/B/C] → [Supervisor] → User
            ↓               ↑              ↓
       Route & Collect   Execute     Synthesize
```

### Handoff Tools Explained

**What are Handoff Tools?**
- Special tools that transfer control between agents
- Created using `create_handoff_tool()`
- Include descriptions to help agents decide when to transfer
- Enable seamless collaboration between specialized agents

**Example Handoff Flow:**
```python
User: "Research Apple and create a stock chart"

1. Researcher Agent:
   - Uses wikipedia_tool("Apple Inc.")  
   - Gets company information
   - Calls transfer_to_analyst handoff tool

2. Analyst Agent:
   - Receives context from Researcher
   - Uses stock_data_tool("AAPL", 30)
   - Uses python_repl_tool(chart_code)
   - Returns final response to user
```

## 🔧 Setup Instructions

### Step 1: Install Required Libraries

```bash
pip install --quiet wikipedia==1.4.0 langchain-core==0.3.69 langgraph==0.5.3 langchain-openai==0.3.28 langchain-experimental==0.3.4 langgraph-swarm==0.0.13
```

### Step 2: Import Dependencies and Define Tools

```python
# Tools are pre-defined (same as previous exercises)
# Focus will be on multi-agent architecture and handoff tools
```

## 🏗️ Building Your First Multi-Agent Swarm

### Step 3: Create Handoff Tools

**Your task:** Create handoff tools that enable agents to transfer control to each other. You need to write about **65%** of the implementation.

```python
# TODO: Import required modules
from langchain_openai import ________
from langgraph_swarm import ________

llm = ChatOpenAI(model="________")

# TODO: Step 3a - Create handoff tool for researcher -> analyst
# Hint: This transfers from researcher to analyst for data visualization tasks
transfer_to_analyst = ________(
    agent_name="________",  # Name of target agent
    description="Transfer user to the ________ assistant, who can create visualizations of provided data."
)

# TODO: Step 3b - Create handoff tool for analyst -> researcher  
# Hint: This transfers from analyst to researcher for information gathering
transfer_to_researcher = ________(
    agent_name="________",  # Name of target agent
    description="Transfer user to the ________ assistant, who can retrieve Wikipedia summaries or load stock performance data from CSV files."
)
```

<details>
<summary>💡 Step 3 Hints</summary>

**Step 3a:**
- Use `create_handoff_tool`
- agent_name should be `"analyst"`
- Description should guide when to transfer to analyst

**Step 3b:** 
- Use `create_handoff_tool`
- agent_name should be `"researcher"`
- Description should guide when to transfer to researcher

</details>

### Step 4: Create Specialized Agents

**Your task:** Create two specialized agents with their respective tools and handoff capabilities.

```python
# TODO: Import create_react_agent
from langgraph.prebuilt import ________

# TODO: Step 4a - Create researcher agent
# Hint: Researcher has wikipedia_tool, stock_data_tool, and transfer_to_analyst
research_agent = ________(
    ________,  # LLM
    tools=[________, ________, ________],  # List all three tools
    prompt="You provide summaries from Wikipedia, and can load raw, numerical stock performance data from CSV files.",
    name="________"  # Agent name
)

# TODO: Step 4b - Create analyst agent  
# Hint: Analyst has python_repl_tool and transfer_to_researcher
analyst_agent = ________(
    ________,  # LLM
    [________, ________],  # List both tools
    prompt="You generate plots of stock performance data provided by another assistant.",
    name="________"  # Agent name
)
```

<details>
<summary>🔍 Step 4 Hints</summary>

**Step 4a - Research Agent:**
- Tools: `[wikipedia_tool, stock_data_tool, transfer_to_analyst]`
- Name: `"researcher"`

**Step 4b - Analyst Agent:**
- Tools: `[python_repl_tool, transfer_to_researcher]` 
- Name: `"analyst"`

</details>

### Step 5: Create the Swarm System

**Your task:** Bring the agents together with shared memory and coordination.

```python
# TODO: Step 5a - Import required swarm components
from langgraph_swarm import ________
from langgraph.checkpoint.memory import ________

# TODO: Step 5b - Set up memory and configuration
config = {"configurable": {"thread_id": "1", "user_id": "1"}}
checkpointer = ________()

# TODO: Step 5c - Create the swarm multi-agent system
# Hint: Set researcher as the default agent since users often start with questions
swarm = ________(
    agents=[________, ________],  # List both agents
    default_active_agent="________"  # Which agent receives initial user input
).compile(checkpointer=________)
```

<details>
<summary>💡 Step 5 Hints</summary>

**Step 5a:** Import `create_swarm` and `InMemorySaver`
**Step 5b:** Use `InMemorySaver()`
**Step 5c:** 
- agents: `[research_agent, analyst_agent]`
- default_active_agent: `"researcher"`
- checkpointer: `checkpointer`

</details>

### Step 6: Test Your Swarm System

```python
# TODO: Step 6a - Visualize the swarm
________

# TODO: Step 6b - Test different scenarios
from course_helper_functions import pretty_print_messages

# Test 1: Simple research request (should stay with researcher)
for chunk in swarm.stream(
    {"messages": [{"role": "user", "content": "Tell me about Apple Inc."}]}, config
):
    pretty_print_messages(chunk)

# Test 2: Request requiring handoff (research + visualization)  
for chunk in swarm.stream(
    {"messages": [{"role": "user", "content": "Research Tesla and create a stock price chart"}]}, config
):
    pretty_print_messages(chunk)
```

## ✅ Expected Behavior

Your multi-agent swarm should demonstrate:

### Agent Specialization

**Researcher Agent Capabilities:**
- ✅ Wikipedia research queries
- ✅ Stock data retrieval
- ✅ Handoff to analyst for visualization needs
- ❌ Cannot create charts or visualizations

**Analyst Agent Capabilities:**
- ✅ Data visualization and charting
- ✅ Python code execution
- ✅ Handoff to researcher for data gathering
- ❌ Cannot access Wikipedia or stock data directly

### Intelligent Handoffs

**Single Agent Tasks:**
```
User: "Tell me about Microsoft"
Flow: User → Researcher → Wikipedia Tool → Response
Result: No handoff needed, researcher handles complete request
```

**Multi-Agent Collaboration:**
```
User: "Research Apple and show their stock performance chart"
Flow: User → Researcher → Wikipedia Tool → Handoff to Analyst → Stock + Chart Tools → Response  
Result: Seamless collaboration between specialized agents
```

### Memory and Context Sharing

**Conversation Continuity:**
```
Turn 1: "Tell me about Tesla" (Researcher agent)
Turn 2: "Now create a chart of their stock price" (Should handoff to Analyst with context)
Result: Analyst knows we're discussing Tesla from previous context
```

## 🎓 Understanding Your Code

### Key Concepts Explained

**1. Handoff Tool Mechanism:**
```python
transfer_to_analyst = create_handoff_tool(
    agent_name="analyst",
    description="Transfer user to the analyst assistant..."
)
```
- **agent_name**: Target agent identifier
- **description**: Helps current agent decide when to transfer
- **Automatic**: LLM decides when to use based on user request

**2. Agent Specialization Strategy:**
```python
# Researcher: Information gathering specialist
tools=[wikipedia_tool, stock_data_tool, transfer_to_analyst]

# Analyst: Data visualization specialist  
tools=[python_repl_tool, transfer_to_researcher]
```
- **Focused toolsets**: Each agent has tools for their specialty
- **Cross-agent tools**: Handoff tools enable collaboration
- **Clear responsibilities**: Agents know their role and limitations

**3. Swarm Coordination:**
```python
swarm = create_swarm(
    agents=[research_agent, analyst_agent],
    default_active_agent="researcher"
)
```
- **Agent registry**: All agents available in the system
- **Default entry point**: Where user requests start
- **Dynamic routing**: Agents can transfer control as needed

**4. Shared Memory System:**
```python
checkpointer = InMemorySaver()
config = {"configurable": {"thread_id": "1", "user_id": "1"}}
```
- **Persistent context**: All agents share conversation history
- **Thread management**: Separate conversations with thread IDs
- **User tracking**: Associate conversations with specific users

### Swarm vs Single Agent Comparison

**Single Agent Approach:**
```python
# One agent with all tools
agent = create_react_agent(
    llm,
    [wikipedia_tool, stock_data_tool, python_repl_tool],
    prompt="You are a comprehensive assistant that can..."
)
```
- **Pros**: Simple setup, single point of control
- **Cons**: Complex prompts, slower decisions, harder to optimize

**Swarm Approach:**
```python
# Multiple specialized agents
researcher = create_react_agent(llm, research_tools, research_prompt, name="researcher")
analyst = create_react_agent(llm, analysis_tools, analysis_prompt, name="analyst")
swarm = create_swarm([researcher, analyst], default_active_agent="researcher")
```
- **Pros**: Specialized expertise, faster decisions, modular design
- **Cons**: More complex setup, coordination overhead, potential handoff issues

## 🔧 Troubleshooting Guide

### Common Issues & Solutions

**❌ "create_handoff_tool not found"**
```python
# Solution: Import from correct module
from langgraph_swarm import create_handoff_tool
```

**❌ "Agent doesn't transfer control"**
- **Check:** Handoff tool descriptions are clear and specific
- **Verify:** Target agent names match exactly
- **Test:** Try more explicit requests that clearly need handoff

**❌ "Agents lose context after handoff"**
- **Check:** Checkpointer is properly configured
- **Verify:** Same config is used for all interactions
- **Ensure:** InMemorySaver is shared across agents

**❌ "Infinite handoff loops"**
```python
# Problem: Agents keep transferring back and forth
# Solution: Make tool descriptions more specific about when to transfer
transfer_to_analyst = create_handoff_tool(
    agent_name="analyst",
    description="Transfer ONLY when user needs data visualization, charts, or plots. Do not transfer for simple data requests."
)
```

**❌ "Default agent not working"**
- **Check:** Agent name in `default_active_agent` matches agent's `name` parameter
- **Verify:** Agent names are spelled correctly
- **Test:** Try explicit agent specification

## 🧪 Testing Challenges

### Challenge 1: Agent Specialization Validation

```python
# Test that each agent sticks to their specialization
specialization_tests = [
    # Should stay with researcher
    ("Tell me about Microsoft Corporation", "researcher", False),
    ("Get Apple stock data for 10 days", "researcher", False),
    ("What is Tesla's business model?", "researcher", False),
    
    # Should require handoff to analyst
    ("Create a chart of Apple stock prices", "researcher", True),
    ("Plot Tesla's stock performance", "researcher", True), 
    ("Show me a graph of Microsoft's data", "researcher", True),
    
    # Should require handoff to researcher (if starting with analyst)
    ("Research Google's company information", "analyst", True),
    ("Get Amazon's stock data", "analyst", True),
]

def test_agent_specialization():
    """Test that agents properly specialize and handoff when needed"""
    for request, starting_agent, should_handoff in specialization_tests:
        print(f"\n--- Testing: {request} ---")
        print(f"Starting agent: {starting_agent}")
        print(f"Should handoff: {should_handoff}")
        
        # Test the request
        chunks = list(swarm.stream({
            "messages": [{"role": "user", "content": request}]
        }, config))
        
        # Analyze if handoff occurred
        handoff_detected = False
        agents_used = set()
        
        for chunk in chunks:
            if "messages" in chunk:
                for message in chunk["messages"]:
                    # Check for handoff tool calls
                    if message.get("role") == "assistant" and "tool_calls" in message:
                        for tool_call in message["tool_calls"]:
                            if "transfer_to" in tool_call["name"]:
                                handoff_detected = True
                    
                    # Track which agents were active
                    if "name" in message:
                        agents_used.add(message["name"])
        
        print(f"Handoff detected: {handoff_detected}")
        print(f"Agents used: {list(agents_used)}")
        
        # Validate behavior
        if should_handoff == handoff_detected:
            print("✅ Correct handoff behavior")
        else:
            print("❌ Unexpected handoff behavior")

# Run specialization tests
test_agent_specialization()
```

### Challenge 2: Multi-Turn Conversation Memory

```python
def test_conversation_memory():
    """Test that agents maintain context across turns and handoffs"""
    
    # Multi-turn conversation that requires memory
    conversation_turns = [
        "Tell me about Apple Inc.",  # Researcher handles this
        "What's their stock ticker symbol?",  # Should remember we're talking about Apple
        "Show me their recent stock performance",  # Should get AAPL data
        "Create a chart of that data",  # Should handoff to analyst with context
        "Make the chart title more descriptive"  # Analyst should know what chart we mean
    ]
    
    print("=== Multi-Turn Conversation Memory Test ===")
    
    for i, turn in enumerate(conversation_turns, 1):
        print(f"\n--- Turn {i}: {turn} ---")
        
        chunks = list(swarm.stream({
            "messages": [{"role": "user", "content": turn}]
        }, config))
        
        # Analyze response for context awareness
        final_response = None
        tools_used = []
        
        for chunk in chunks:
            if "messages" in chunk:
                for message in chunk["messages"]:
                    if message.get("role") == "assistant":
                        if "content" in message and not message.get("tool_calls"):
                            final_response = message["content"]
                        elif "tool_calls" in message:
                            for tool_call in message["tool_calls"]:
                                tools_used.append(tool_call["name"])
        
        print(f"Tools used: {tools_used}")
        if final_response:
            print(f"Response preview: {final_response[:100]}...")
        
        # Check for context awareness
        if i > 1:  # After first turn
            context_indicators = ["apple", "aapl", "mentioned", "discussed", "previous"]
            has_context = any(indicator in final_response.lower() for indicator in context_indicators) if final_response else False
            print(f"Context awareness: {'✅' if has_context else '❓'}")

# Run memory test
test_conversation_memory()
```

### Challenge 3: Performance Comparison

```python
import time

def compare_single_vs_swarm_performance():
    """Compare performance between single agent and swarm approaches"""
    
    # Create single agent for comparison (if available)
    try:
        from previous_exercises import create_single_agent  # Hypothetical import
        single_agent = create_single_agent()
        single_agent_available = True
    except:
        single_agent_available = False
        print("Single agent not available for comparison")
    
    test_requests = [
        "Tell me about Apple Inc.",
        "Get Tesla stock data for 7 days",
        "Create a chart of Microsoft stock prices",
        "Research Amazon and visualize their stock performance"
    ]
    
    swarm_times = []
    single_times = []
    
    for request in test_requests:
        print(f"\n--- Performance Test: {request} ---")
        
        # Test swarm performance
        start_time = time.time()
        swarm_chunks = list(swarm.stream({
            "messages": [{"role": "user", "content": request}]
        }, config))
        swarm_time = time.time() - start_time
        swarm_times.append(swarm_time)
        
        print(f"Swarm time: {swarm_time:.2f}s")
        
        # Test single agent if available
        if single_agent_available:
            start_time = time.time()
            single_chunks = list(single_agent.stream({
                "messages": [{"role": "user", "content": request}]
            }))
            single_time = time.time() - start_time
            single_times.append(single_time)
            
            print(f"Single agent time: {single_time:.2f}s")
            print(f"Difference: {swarm_time - single_time:+.2f}s")
    
    # Summary
    print(f"\n=== Performance Summary ===")
    print(f"Swarm average: {sum(swarm_times) / len(swarm_times):.2f}s")
    if single_agent_available:
        print(f"Single agent average: {sum(single_times) / len(single_times):.2f}s")
        print(f"Overhead: {(sum(swarm_times) - sum(single_times)) / len(swarm_times):.2f}s per request")

# Run performance comparison
compare_single_vs_swarm_performance()
```

### Challenge 4: Handoff Decision Analysis

```python
def analyze_handoff_decisions():
    """Analyze how and when agents decide to use handoff tools"""
    
    handoff_scenarios = [
        # Clear handoff triggers
        ("Plot Apple stock data", "Should trigger research→analyst handoff"),
        ("Create a chart", "Should trigger research→analyst handoff"),
        ("Visualize the data", "Should trigger research→analyst handoff"),
        
        # Ambiguous scenarios
        ("Show me Apple information", "Might not need handoff"),
        ("Apple stock prices", "Could be data retrieval or visualization"),
        ("Help with Apple", "Very ambiguous, might stay with researcher"),
        
        # Reverse handoff scenarios  
        ("Tell me about the company", "From analyst, should trigger analyst→research handoff"),
        ("Get more data", "From analyst, might trigger handoff to researcher"),
    ]
    
    for request, expectation in handoff_scenarios:
        print(f"\n--- Handoff Analysis: {request} ---")
        print(f"Expectation: {expectation}")
        
        chunks = list(swarm.stream({
            "messages": [{"role": "user", "content": request}]
        }, config))
        
        # Track handoff decision process
        handoff_reasoning = []
        handoffs_made = []
        
        for chunk in chunks:
            if "messages" in chunk:
                for message in chunk["messages"]:
                    if message.get("role") == "assistant":
                        # Look for handoff tool calls
                        if "tool_calls" in message:
                            for tool_call in message["tool_calls"]:
                                if "transfer_to" in tool_call["name"]:
                                    handoffs_made.append(tool_call["name"])
                        
                        # Capture reasoning in content
                        if "content" in message and "transfer" in message["content"].lower():
                            handoff_reasoning.append(message["content"][:100] + "...")
        
        print(f"Handoffs made: {handoffs_made}")
        if handoff_reasoning:
            print(f"Reasoning: {handoff_reasoning}")
        
        # Assess decision quality
        appropriate_handoff = len(handoffs_made) > 0 and any(trigger in request.lower() 
                            for trigger in ["plot", "chart", "visualize", "graph"])
        print(f"Handoff decision: {'✅ Appropriate' if appropriate_handoff else '❓ Review needed'}")

# Run handoff analysis
analyze_handoff_decisions()
```

## 🚀 Advanced Extensions

### Extension 1: Three-Agent Swarm

```python
# Add a third specialist agent
@tool
def email_notification_tool(
    recipient: Annotated[str, "Email recipient"],
    subject: Annotated[str, "Email subject"],
    content: Annotated[str, "Email content"]
):
    """Send email notifications about analysis results."""
    return f"Email sent to {recipient}: {subject}"

# Create handoff tools for the new agent
transfer_to_notifier = create_handoff_tool(
    agent_name="notifier",
    description="Transfer to send email notifications about completed analysis."
)

# Add handoff tools to existing agents
research_agent_v2 = create_react_agent(
    llm,
    [wikipedia_tool, stock_data_tool, transfer_to_analyst, transfer_to_notifier],
    prompt="You provide research and can transfer to analyst for visualization or notifier for communications.",
    name="researcher"
)

analyst_agent_v2 = create_react_agent(
    llm,
    [python_repl_tool, transfer_to_researcher, transfer_to_notifier],
    prompt="You create visualizations and can transfer to researcher for data or notifier for communications.",
    name="analyst"
)

# Create the new notifier agent
notifier_agent = create_react_agent(
    llm,
    [email_notification_tool, transfer_to_researcher, transfer_to_analyst],
    prompt="You send email notifications and can transfer to other agents for additional work.",
    name="notifier"
)

# Create three-agent swarm
three_agent_swarm = create_swarm(
    agents=[research_agent_v2, analyst_agent_v2, notifier_agent],
    default_active_agent="researcher"
).compile(checkpointer=InMemorySaver())
```

### Extension 2: Dynamic Agent Selection

```python
class SmartSwarmRouter:
    """Enhanced swarm with intelligent default agent selection"""
    
    def __init__(self, swarm):
        self.swarm = swarm
        self.request_patterns = {
            "research": ["tell me", "what is", "who is", "information", "about"],
            "analysis": ["plot", "chart", "visualize", "graph", "show data"],
            "notification": ["send", "email", "notify", "alert"]
        }
    
    def smart_route(self, user_input, config):
        """Route to most appropriate agent based on request content"""
        
        user_input_lower = user_input.lower()
        
        # Score each agent type
        scores = {}
        for agent_type, patterns in self.request_patterns.items():
            scores[agent_type] = sum(1 for pattern in patterns if pattern in user_input_lower)
        
        # Select best agent
        best_agent = max(scores, key=scores.get)
        
        print(f"Smart routing selected: {best_agent} (scores: {scores})")
        
        # Route to specific agent (this would require swarm modification)
        return self.swarm.stream({
            "messages": [{"role": "user", "content": user_input}]
        }, config)

# Use smart routing
smart_router = SmartSwarmRouter(swarm)
```

### Extension 3: Agent Performance Monitoring

```python
class SwarmMonitor:
    """Monitor multi-agent system performance and collaboration"""
    
    def __init__(self):
        self.metrics = {
            "total_requests": 0,
            "handoffs_per_request": [],
            "agent_utilization": {},
            "handoff_patterns": {},
            "response_times": []
        }
    
    def monitor_request(self, swarm, request, config):
        """Monitor a single swarm request"""
        start_time = time.time()
        self.metrics["total_requests"] += 1
        
        chunks = list(swarm.stream({
            "messages": [{"role": "user", "content": request}]
        }, config))
        
        execution_time = time.time() - start_time
        self.metrics["response_times"].append(execution_time)
        
        # Analyze collaboration patterns
        handoff_count = 0
        agents_used = set()
        handoff_sequence = []
        
        for chunk in chunks:
            if "messages" in chunk:
                for message in chunk["messages"]:
                    if message.get("name"):
                        agents_used.add(message["name"])
                    
                    if message.get("role") == "assistant" and "tool_calls" in message:
                        for tool_call in message["tool_calls"]:
                            if "transfer_to" in tool_call["name"]:
                                handoff_count += 1
                                target_agent = tool_call["name"].replace("transfer_to_", "")
                                handoff_sequence.append(target_agent)
        
        self.metrics["handoffs_per_request"].append(handoff_count)
        
        # Update agent utilization
        for agent in agents_used:
            self.metrics["agent_utilization"][agent] = \
                self.metrics["agent_utilization"].get(agent, 0) + 1
        
        # Track handoff patterns
        if handoff_sequence:
            pattern = " → ".join(handoff_sequence)
            self.metrics["handoff_patterns"][pattern] = \
                self.metrics["handoff_patterns"].get(pattern, 0) + 1
        
        return {
            "execution_time": execution_time,
            "handoffs": handoff_count,
            "agents_used": list(agents_used),
            "handoff_sequence": handoff_sequence
        }
    
    def generate_report(self):
        """Generate comprehensive swarm performance report"""
        avg_response_time = sum(self.metrics["response_times"]) / len(self.metrics["response_times"])
        avg_handoffs = sum(self.metrics["handoffs_per_request"]) / len(self.metrics["handoffs_per_request"])
        
        return {
            "summary": {
                "total_requests": self.metrics["total_requests"],
                "avg_response_time": avg_response_time,
                "avg_handoffs_per_request": avg_handoffs
            },
            "agent_utilization": self.metrics["agent_utilization"],
            "common_handoff_patterns": dict(sorted(
                self.metrics["handoff_patterns"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5])
        }

# Use swarm monitoring
monitor = SwarmMonitor()

# Test requests with monitoring
test_requests = [
    "Tell me about Apple Inc.",
    "Create a chart of Tesla stock prices", 
    "Research Microsoft and visualize their performance",
    "Plot Google's stock data"
]

for request in test_requests:
    result = monitor.monitor_request(swarm, request, config)
    print(f"Request: {request}")
    print(f"Result: {result}")

# Generate performance report
report = monitor.generate_report()
print("\n=== Swarm Performance Report ===")
import json
print(json.dumps(report, indent=2))
```

## 📝 Self-Assessment

**Check your understanding:**

□ I can create handoff tools using `create_handoff_tool()`
□ I understand how agents specialize with focused toolsets
□ I can set up shared memory with `InMemorySaver` and config
□ I know how to create multi-agent swarms with `create_swarm()`
□ I understand when agents should handoff vs handle requests locally
□ I can analyze multi-agent collaboration patterns
□ I recognize the benefits and limitations of swarm architecture

## 💡 Real-World Applications

**Where swarm architectures are used:**

**Customer Service:**
- **Router Agent**: Classifies incoming requests
- **Support Agent**: Handles common issues  
- **Specialist Agent**: Escalates complex technical problems

**Financial Analysis:**
- **Research Agent**: Gathers market data and news
- **Analysis Agent**: Performs quantitative analysis
- **Report Agent**: Generates client reports

**Content Creation:**
- **Research Agent**: Gathers information and sources
- **Writer Agent**: Creates content drafts
- **Editor Agent**: Reviews and improves content

**E-commerce:**
- **Product Agent**: Handles product information
- **Order Agent**: Processes transactions
- **Support Agent**: Handles customer issues

## 🎉 Congratulations!

You've successfully built your first multi-agent swarm system! This system can:

- ✅ **Specialize agents** for optimal performance in specific domains
- ✅ **Enable seamless handoffs** between agents based on request needs
- ✅ **Maintain shared context** across agent transfers
- ✅ **Scale horizontally** by adding new specialized agents
- ✅ **Improve modularity** making individual agents easier to optimize

**Key Takeaways:**
- Handoff tools enable agent-to-agent communication
- Specialization improves performance and maintainability  
- Shared memory ensures context continuity across handoffs
- Swarm architecture provides peer-to-peer agent collaboration
- Agent descriptions and handoff tools guide intelligent routing

## 🚀 Next Steps

After completing this exercise:

1. **Exercise 2.6:** Supervisor architecture for centralized coordination
2. **Advanced:** Human-in-the-loop multi-agent workflows
3. **Enterprise:** Production multi-agent systems with monitoring
4. **Research:** Agent negotiation and conflict resolution

## 🔄 Architecture Evolution

**Your Multi-Agent Journey:**
- **Exercise 2.5**: Swarm (peer-to-peer) ← You are here
- **Future**: Supervisor (hierarchical), Pipeline (sequential), Network (complex)

## 🧠 Swarm Architecture Benefits

**Swarm vs Single Agent:**

| Aspect | Single Agent | Swarm Architecture |
|--------|--------------|-------------------|
| **Complexity** | High (one agent does all) | Low (specialized agents) |
| **Performance** | Slower (many tools/decisions) | Faster (focused decisions) |
| **Scalability** | Limited | High (add new agents) |
| **Maintenance** | Difficult | Modular |
| **Debugging** | Complex | Clear agent boundaries |
| **Specialization** | Jack of all trades | Master of specific domains |

You've now mastered multi-agent collaboration! Ready to explore supervisor architectures and advanced coordination patterns! 🚀🐝✨