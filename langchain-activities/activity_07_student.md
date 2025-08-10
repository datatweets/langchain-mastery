# 🚀 Activity 07: One-Line Agents with create_react_agent - Student Practice Guide

## 🎯 Learning Objectives

By the end of this activity, you will:
- Learn to use LangGraph's high-level `create_react_agent()` function
- Understand the ReAct (Reason + Act) agent pattern
- Master rapid agent prototyping with pre-built components
- Compare low-level graph construction vs high-level utilities
- Create production-ready agents with minimal code

## 📚 Background Context

**From Low-Level to High-Level**
You've learned to build agents step-by-step:
1. **Activity 04**: Manual graph construction (nodes, edges, state)
2. **Exercise 1.7**: Linear tool integration (detailed implementation)
3. **Exercise 2.2**: Conditional routing (complex logic)
4. **Exercise 2.3**: One-line agents (abstraction layer) ← You are here

**Why Use High-Level Utilities?**
- ✅ **Rapid Prototyping**: Build agents in minutes, not hours
- ✅ **Proven Patterns**: ReAct pattern is well-tested and optimized
- ✅ **Focus on Business Logic**: Spend time on tools and prompts, not plumbing
- ✅ **Easy Experimentation**: Quick iteration cycles for testing ideas
- ✅ **Production Ready**: Built-in best practices and error handling

**What is ReAct?**
ReAct (Reason + Act) is a popular agent pattern:
1. **Reason**: LLM analyzes the user request and decides what to do
2. **Act**: LLM calls appropriate tools to gather information
3. **Observe**: LLM receives tool results and processes them
4. **Repeat**: Continue until task is complete

## 🧠 ReAct Agent Pattern

### ReAct Workflow Example

**User**: "Tell me about Apple Inc. and show their stock performance"

**Agent Reasoning Process:**
```
1. REASON: "User wants company info and stock data. I need Wikipedia tool for info and stock tool for data."
2. ACT: Calls wikipedia_tool("Apple Inc.")
3. OBSERVE: Receives company information
4. REASON: "Now I have company info. I need stock data."
5. ACT: Calls stock_data_tool("AAPL", 7)  
6. OBSERVE: Receives stock data
7. REASON: "I have all needed information. Time to respond."
8. RESPOND: Provides comprehensive answer with both pieces of information
```

### Benefits of ReAct Pattern

**1. Natural Problem Solving:**
- Mirrors human reasoning process
- Step-by-step decision making
- Transparent reasoning chain

**2. Flexible Tool Usage:**
- Can use multiple tools in sequence
- Adapts tool selection based on context
- Handles complex multi-step tasks

**3. Error Recovery:**
- Can retry with different approaches
- Handles tool failures gracefully
- Learns from previous attempts in conversation

## 🔧 Setup Instructions

### Step 1: Install Required Libraries
```bash
pip install --quiet wikipedia==1.4.0 langchain-core==0.3.59 langgraph==0.5.3 langchain-openai==0.3.16 langchain-experimental==0.3.4
```

### Step 2: Import Dependencies and Verify Tools
```python
# Tools are pre-defined (same as previous exercises)
# Focus will be on using create_react_agent()
```

## 🏗️ Building One-Line Agents

### Step 3: Import Required Components

**Your task:** Import the necessary components for creating a ReAct agent.

```python
# TODO: Import create_react_agent and ChatOpenAI
from langgraph.prebuilt import ________
from langchain_openai import ________
```

<details>
<summary>💡 Step 3 Hint</summary>

You need:
- `create_react_agent` from langgraph.prebuilt for agent creation
- `ChatOpenAI` from langchain_openai for the LLM
</details>

### Step 4: Define Tools and LLM

**Your task:** Set up the components needed for agent creation.

```python
# TODO: Step 4a - Create tools list with all three tools
# Hint: Include wikipedia_tool, stock_data_tool, and python_repl_tool
tools = [________, ________, ________]

# TODO: Step 4b - Create LLM instance
# Hint: Use ChatOpenAI with gpt-4o-mini model
llm = ________(model="________")
```

<details>
<summary>🔍 Step 4 Hints</summary>

**Step 4a:** List all three tools: `[wikipedia_tool, stock_data_tool, python_repl_tool]`
**Step 4b:** Use `ChatOpenAI(model="gpt-4o-mini")`
</details>

### Step 5: Create System Prompt

**Your task:** Write a system prompt that guides the agent's behavior. You need to write about **60%** of the implementation.

```python
# TODO: Step 5 - Define system prompt for the agent
# Hint: Describe the agent's role and available tools
prompt = """
You are an assistant for ________ and ________ of Fortune 500 companies. 

You have access to three tools:
- A ________ tool for retrieving factual summary information about companies
- A ________ tool for retrieving stock price information from local CSV files  
- A ________ tool for executing Python code, which is to be used for creating stock performance visualizations

Use these tools effectively to provide comprehensive analysis and insights.
Always be helpful and provide detailed responses based on the tool results.
"""
```

<details>
<summary>💡 Step 5 Hints</summary>

Fill in the blanks:
- "research and analysis"
- "Wikipedia"
- "stock performance data" 
- "Python"
</details>

### Step 6: Create the ReAct Agent

**Your task:** Use `create_react_agent()` to build the agent in one line!

```python
# TODO: Step 6 - Create agent using create_react_agent
# Hint: Pass llm, tools, and system_prompt parameters
agent = ________(
    ________,  # LLM
    ________,  # Tools list
    system_prompt=________  # System prompt
)
```

<details>
<summary>🔍 Step 6 Hint</summary>

Use `create_react_agent(llm, tools, system_prompt=prompt)`
</details>

### Step 7: Visualize and Test

```python
# TODO: Step 7a - Display agent visualization
# Hint: Just call the agent object
________

# TODO: Step 7b - Test the agent with different requests
from course_helper_functions import pretty_print_messages

# Test simple conversation
for chunk in agent.stream({
    "messages": [{"role": "user", "content": "Hello, what can you help me with?"}]
}):
    pretty_print_messages(chunk)
```

### Step 8: Advanced Testing

**Your task:** Test the agent with complex multi-step requests.

```python
# TODO: Step 8 - Test complex multi-tool workflow
# Hint: Create a request that requires multiple tools
complex_request = "Tell me Tesla's current CEO, their latest stock price, and generate a plot of the closing price with the most up-to-date data you have available."

for chunk in agent.stream({
    "messages": [{"role": "user", "content": complex_request}]
}):
    pretty_print_messages(chunk)
```

## ✅ Expected Behavior

Your one-line agent should demonstrate:

### Intelligent Tool Selection
- **Wikipedia queries**: "Tell me about Apple Inc."
- **Stock data queries**: "Show me AAPL stock prices"
- **Visualization queries**: "Plot Tesla stock performance"
- **Conversational queries**: "Hello, how are you?"

### Multi-Step Workflows
- **Research + Data**: "Research Microsoft and show their stock data"
- **Data + Visualization**: "Get Apple stock data and create a chart"
- **Complete Analysis**: "Full analysis of Tesla including company info, stock data, and chart"

### ReAct Pattern in Action

**Observable Steps:**
1. **User Request** → Agent receives input
2. **Reasoning** → Agent analyzes what's needed
3. **Tool Selection** → Agent chooses appropriate tool
4. **Tool Execution** → Tool runs and returns results
5. **Processing** → Agent interprets results
6. **Response/Continue** → Agent responds or continues with more tools

## 🎓 Understanding Your Code

### Key Concepts Explained:

**1. create_react_agent() Function:**
```python
agent = create_react_agent(llm, tools, system_prompt=prompt)
```
- **Automatic Graph Construction**: Builds nodes, edges, and routing logic
- **ReAct Implementation**: Implements Reason-Act-Observe pattern
- **Tool Integration**: Handles tool binding and execution
- **Error Handling**: Built-in error recovery and handling

**2. System Prompt Importance:**
```python
prompt = """You are an assistant for research and analysis..."""
```
- **Role Definition**: Tells the agent what it is and what it does
- **Tool Descriptions**: Explains available capabilities
- **Behavior Guidance**: Sets expectations for responses
- **Context Setting**: Provides domain-specific knowledge

**3. Automatic State Management:**
```python
# Handled automatically by create_react_agent
# - Message history tracking
# - Tool call management  
# - Response formatting
# - Error recovery
```

**4. Built-in Graph Structure:**
```
START → Agent → Tools (if needed) → Agent → END
             ↓                     ↑
             → Direct Response → END
```

### Comparison: Manual vs One-Line

**Manual Construction (Previous Exercises):**
```python
# 20+ lines of code
class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)
llm_with_tools = llm.bind_tools(tools)

def llm_node(state):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder.add_node("llm", llm_node)
tool_node = ToolNode(tools)
graph_builder.add_node("tools", tool_node)
graph_builder.add_edge(START, "llm")
graph_builder.add_conditional_edges("llm", tools_condition, {...})
graph_builder.add_edge("tools", "llm")
graph = graph_builder.compile()
```

**One-Line Construction:**
```python
# 1 line of code!
agent = create_react_agent(llm, tools, system_prompt=prompt)
```

## 🔧 Troubleshooting Guide

### Common Issues & Solutions:

**❌ "create_react_agent not found"**
```python
# Solution: Import from correct module
from langgraph.prebuilt import create_react_agent
```

**❌ "Agent doesn't use tools appropriately"**
- **Check:** System prompt clearly describes tool capabilities
- **Verify:** Tool descriptions are informative and accurate
- **Test:** With explicit tool-requiring requests

**❌ "Agent gives incorrect responses"**
- **Improve:** System prompt with more specific guidance
- **Add:** Examples of good responses in the prompt
- **Check:** Tool outputs are properly formatted

**❌ "Performance issues"**
- **Note:** One-line agents include built-in optimizations
- **Compare:** Performance should be similar to manual construction
- **Monitor:** Tool execution times, not agent overhead

## 🧪 Testing Challenges

### Challenge 1: Agent Behavior Comparison
```python
# Compare one-line agent with your manual implementation
test_requests = [
    "Tell me about Apple Inc.",
    "Show me AAPL stock data for 5 days", 
    "Create a chart of Tesla stock prices",
    "Hello, how are you?",
    "Research Microsoft and show their recent performance"
]

def compare_agents(manual_graph, onelink_agent, requests):
    """Compare responses from both agent implementations"""
    for request in requests:
        print(f"\n--- Comparing: {request} ---")
        
        # Test manual agent
        print("MANUAL AGENT:")
        manual_chunks = list(manual_graph.stream({
            "messages": [{"role": "user", "content": request}]
        }))
        # Extract final response
        for chunk in manual_chunks[-3:]:  # Show last few chunks
            pretty_print_messages(chunk)
        
        print("\nONE-LINE AGENT:")
        oneline_chunks = list(oneline_agent.stream({
            "messages": [{"role": "user", "content": request}]
        }))
        # Extract final response  
        for chunk in oneline_chunks[-3:]:  # Show last few chunks
            pretty_print_messages(chunk)
        
        print("-" * 50)

# Run comparison if you have both implementations
# compare_agents(your_manual_graph, agent, test_requests)
```

### Challenge 2: Prompt Engineering Optimization
```python
# Test different system prompts to optimize agent behavior
prompts_to_test = [
    # Basic prompt
    """You are a helpful assistant with access to Wikipedia, stock data, and Python tools.""",
    
    # Detailed prompt
    """You are an expert financial analyst assistant. Use Wikipedia for company research, 
    stock data tool for financial information, and Python for creating visualizations. 
    Always provide comprehensive analysis.""",
    
    # Role-specific prompt
    """You are a Fortune 500 company research specialist. Your expertise includes:
    - Company background research using Wikipedia
    - Financial data analysis using stock performance tools  
    - Data visualization using Python programming
    
    Provide detailed, professional analysis for all requests."""
]

def test_prompt_variations(prompts, test_request):
    """Test different system prompts"""
    for i, prompt in enumerate(prompts, 1):
        print(f"\n=== Prompt Variation {i} ===")
        
        # Create agent with this prompt
        test_agent = create_react_agent(llm, tools, system_prompt=prompt)
        
        # Test with same request
        chunks = list(test_agent.stream({
            "messages": [{"role": "user", "content": test_request}]
        }))
        
        # Show response quality
        final_response = None
        tools_used = []
        
        for chunk in chunks:
            if "messages" in chunk:
                for message in chunk["messages"]:
                    if message.get("role") == "assistant":
                        if "tool_calls" in message:
                            for tool_call in message["tool_calls"]:
                                tools_used.append(tool_call["name"])
                        elif "content" in message:
                            final_response = message["content"]
        
        print(f"Tools used: {tools_used}")
        print(f"Response length: {len(final_response) if final_response else 0}")
        if final_response:
            print(f"Response preview: {final_response[:150]}...")

# Test prompt effectiveness
test_prompt_variations(prompts_to_test, "Analyze Apple Inc. comprehensively")
```

### Challenge 3: Multi-Step Workflow Analysis
```python
# Test complex multi-step workflows
def analyze_workflow_steps(agent, request):
    """Analyze the steps taken in a complex workflow"""
    print(f"=== Workflow Analysis: {request} ===")
    
    chunks = list(agent.stream({
        "messages": [{"role": "user", "content": request}]
    }))
    
    step_count = 0
    tool_sequence = []
    reasoning_steps = []
    
    for chunk in chunks:
        if "messages" in chunk:
            for message in chunk["messages"]:
                if message.get("role") == "assistant":
                    if "tool_calls" in message:
                        step_count += 1
                        for tool_call in message["tool_calls"]:
                            tool_sequence.append(tool_call["name"])
                            print(f"Step {step_count}: Using {tool_call['name']}")
                    elif "content" in message and not message.get("tool_calls"):
                        reasoning_steps.append(message["content"][:100] + "...")
    
    print(f"\nWorkflow Summary:")
    print(f"Total steps: {step_count}")
    print(f"Tool sequence: {tool_sequence}")
    print(f"Reasoning steps: {len(reasoning_steps)}")
    
    return {
        "steps": step_count,
        "tools": tool_sequence,
        "reasoning": reasoning_steps
    }

# Test complex workflows
complex_workflows = [
    "Research Tesla, get their stock data, and create a visualization",
    "Compare Apple and Microsoft - give me company info and recent stock performance for both",
    "Find information about Amazon, show their stock trend, and calculate the average price"
]

for workflow in complex_workflows:
    workflow_analysis = analyze_workflow_steps(agent, workflow)
    print("\n" + "="*60)
```

### Challenge 4: Error Handling and Recovery
```python
# Test error handling in one-line agents
def test_error_scenarios(agent):
    """Test how well the agent handles various error scenarios"""
    error_scenarios = [
        ("Invalid stock ticker", "Show me stock data for INVALIDTICKER"),
        ("Impossible request", "Get stock data for 50000 days"),
        ("Broken code", "Execute this Python code: print(undefined_variable)"),
        ("Nonsensical query", "What is the color of Tesla's quarterly earnings?")
    ]
    
    for error_type, request in error_scenarios:
        print(f"\n--- Error Test: {error_type} ---")
        print(f"Request: {request}")
        
        try:
            chunks = list(agent.stream({
                "messages": [{"role": "user", "content": request}]
            }))
            
            error_handled = False
            final_response = None
            
            for chunk in chunks:
                if "messages" in chunk:
                    for message in chunk["messages"]:
                        if message.get("role") == "assistant" and "content" in message:
                            final_response = message["content"]
                        elif (message.get("role") == "tool" and 
                              any(error_word in message.get("content", "").lower() 
                                  for error_word in ["error", "failed", "sorry"])):
                            error_handled = True
            
            print(f"Error handled: {error_handled}")
            if final_response:
                print(f"Agent response: {final_response[:200]}...")
                
        except Exception as e:
            print(f"Unhandled exception: {e}")

# Test error handling
test_error_scenarios(agent)
```

## 🚀 Advanced Extensions

### Extension 1: Custom Agent Configurations
```python
# Experiment with different agent configurations
def create_specialized_agents():
    """Create agents optimized for different tasks"""
    
    # Research-focused agent
    research_agent = create_react_agent(
        llm, 
        [wikipedia_tool],  # Only Wikipedia tool
        system_prompt="""You are a research specialist. Focus on providing comprehensive, 
        factual information about companies using Wikipedia. Always cite sources and provide detailed context."""
    )
    
    # Analysis-focused agent  
    analysis_agent = create_react_agent(
        llm,
        [stock_data_tool, python_repl_tool],  # Only data tools
        system_prompt="""You are a financial data analyst. Specialize in stock data analysis 
        and visualization. Always provide charts and quantitative insights."""
    )
    
    # General-purpose agent (all tools)
    general_agent = create_react_agent(llm, tools, system_prompt=prompt)
    
    return {
        "research": research_agent,
        "analysis": analysis_agent, 
        "general": general_agent
    }

# Test specialized agents
specialized_agents = create_specialized_agents()

# Test each with appropriate requests
test_cases = {
    "research": "Tell me about Microsoft Corporation's history and business model",
    "analysis": "Analyze Apple's stock performance and create a trend chart",
    "general": "Give me a complete analysis of Tesla including company info and stock data"
}

for agent_type, request in test_cases.items():
    print(f"\n=== {agent_type.upper()} AGENT TEST ===")
    agent = specialized_agents[agent_type]
    # Test the specialized agent...
```

### Extension 2: Agent Performance Monitoring
```python
import time

class AgentPerformanceMonitor:
    """Monitor agent performance and behavior"""
    
    def __init__(self):
        self.metrics = {
            "total_requests": 0,
            "avg_response_time": 0,
            "tool_usage": {},
            "error_count": 0,
            "success_count": 0
        }
    
    def monitor_request(self, agent, request):
        """Monitor a single agent request"""
        start_time = time.time()
        
        try:
            # Execute request
            chunks = list(agent.stream({
                "messages": [{"role": "user", "content": request}]
            }))
            
            # Analyze results
            execution_time = time.time() - start_time
            tools_used = self._extract_tools_used(chunks)
            success = self._check_success(chunks)
            
            # Update metrics
            self._update_metrics(execution_time, tools_used, success)
            
            return {
                "success": success,
                "time": execution_time,
                "tools": tools_used,
                "chunks": chunks
            }
            
        except Exception as e:
            self.metrics["error_count"] += 1
            return {"success": False, "error": str(e)}
    
    def _extract_tools_used(self, chunks):
        """Extract tools used from chunks"""
        tools = []
        for chunk in chunks:
            if "messages" in chunk:
                for message in chunk["messages"]:
                    if message.get("role") == "assistant" and "tool_calls" in message:
                        for tool_call in message["tool_calls"]:
                            tools.append(tool_call["name"])
        return tools
    
    def _check_success(self, chunks):
        """Check if request was handled successfully"""
        # Simple heuristic: success if we got a final response
        for chunk in chunks:
            if "messages" in chunk:
                for message in chunk["messages"]:
                    if (message.get("role") == "assistant" and 
                        "content" in message and 
                        not message.get("tool_calls")):
                        return True
        return False
    
    def _update_metrics(self, execution_time, tools_used, success):
        """Update performance metrics"""
        self.metrics["total_requests"] += 1
        
        # Update average response time
        current_avg = self.metrics["avg_response_time"]
        total_requests = self.metrics["total_requests"]
        self.metrics["avg_response_time"] = ((current_avg * (total_requests - 1)) + execution_time) / total_requests
        
        # Update tool usage
        for tool in tools_used:
            self.metrics["tool_usage"][tool] = self.metrics["tool_usage"].get(tool, 0) + 1
        
        # Update success count
        if success:
            self.metrics["success_count"] += 1
    
    def get_report(self):
        """Generate performance report"""
        return {
            "total_requests": self.metrics["total_requests"],
            "success_rate": self.metrics["success_count"] / max(self.metrics["total_requests"], 1),
            "avg_response_time": self.metrics["avg_response_time"],
            "most_used_tool": max(self.metrics["tool_usage"].items(), key=lambda x: x[1], default=("none", 0)),
            "error_rate": self.metrics["error_count"] / max(self.metrics["total_requests"], 1)
        }

# Use the performance monitor
monitor = AgentPerformanceMonitor()

# Test requests
test_requests = [
    "Tell me about Apple Inc.",
    "Show Tesla stock data",
    "Create a chart of Microsoft stock",
    "Hello there!",
    "Analyze Google's performance"
]

for request in test_requests:
    result = monitor.monitor_request(agent, request)
    print(f"Request: {request}")
    print(f"Success: {result['success']}, Time: {result.get('time', 0):.2f}s")

print("\n=== Performance Report ===")
report = monitor.get_report()
for key, value in report.items():
    print(f"{key}: {value}")
```

## 📝 Self-Assessment

**Check your understanding:**

□ I can create agents using `create_react_agent()` in one line  
□ I understand the ReAct (Reason + Act) pattern  
□ I can write effective system prompts for agent behavior  
□ I know when to use high-level vs low-level graph construction  
□ I can compare manual and automatic agent construction  
□ I understand the trade-offs of abstraction layers  
□ I can optimize agent performance through prompt engineering  

## 💡 Real-World Applications

**When to use create_react_agent():**
- **Rapid Prototyping**: Quick proof-of-concepts and demos
- **Standard Workflows**: Common tool-calling patterns
- **Production MVPs**: Fast time-to-market requirements
- **Experimentation**: Testing different tool combinations
- **Educational**: Focus on agent behavior, not implementation

**When to use manual construction:**
- **Custom Routing Logic**: Complex conditional flows
- **Specialized Patterns**: Non-standard agent architectures  
- **Performance Optimization**: Fine-tuned control over execution
- **Advanced Features**: Custom state management or error handling
- **Multi-Agent Systems**: Complex agent coordination (though create_react_agent can be used as components)

## 🎉 Congratulations!

You've mastered rapid agent development! Your one-line agent can:

- ✅ **Handle complex multi-step workflows** with ReAct pattern
- ✅ **Use multiple tools intelligently** based on context
- ✅ **Provide professional responses** with proper formatting
- ✅ **Recover from errors gracefully** with built-in handling
- ✅ **Scale to production** with optimized implementations

**Key Takeaways:**
- `create_react_agent()` provides rapid agent development
- System prompts are crucial for agent behavior
- ReAct pattern enables complex reasoning and tool usage
- High-level utilities abstract away graph construction complexity
- One-line agents are production-ready and fully featured

## 🚀 Next Steps

After completing this exercise:

1. **Multi-Agent Systems**: Use `create_react_agent()` as building blocks
2. **Custom Patterns**: Combine one-line agents with manual construction
3. **Production Deployment**: Scale agents with monitoring and optimization
4. **Advanced Features**: Explore memory, persistence, and coordination

## 🔄 Learning Journey Complete

**Your Progression:**
- **Exercise 1.3-1.5**: Individual tool creation
- **Exercise 1.6**: Graph architecture fundamentals  
- **Exercise 1.7**: Linear tool integration
- **Exercise 2.2**: Conditional routing intelligence
- **Exercise 2.3**: High-level agent abstractions ← You are here

## 🧠 Abstraction Layers

**Understanding the Stack:**

| Layer | Example | Use Case |
|-------|---------|----------|
| **High-Level** | `create_react_agent()` | Rapid prototyping |
| **Mid-Level** | `StateGraph` + utilities | Custom workflows |
| **Low-Level** | Manual nodes/edges | Specialized patterns |

You now have mastery across all abstraction layers - from detailed graph construction to one-line agent creation! 🚀🧠✨

**Choose the Right Tool:**
- **Speed**: Use `create_react_agent()`
- **Control**: Use manual construction
- **Balance**: Combine both approaches

Ready to build multi-agent systems and enterprise applications! 🌟