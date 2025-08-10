# 🐝 Activity 08: Multi-Agent Swarm Systems - Master Solution Guide

## 📋 Activity Overview

**Topic:** Building multi-agent swarm systems with specialized agents and handoff tools  
**Duration:** 60-90 minutes  
**Difficulty:** Advanced  
**Prerequisites:** Single agent creation, tool integration, ReAct patterns, state management

## 🏆 Complete Solution

### Step 1: Environment Setup

```python
# Install required libraries
!pip install --quiet wikipedia==1.4.0 langchain-core==0.3.69 langgraph==0.5.3 langchain-openai==0.3.28 langchain-experimental==0.3.4 langgraph-swarm==0.0.13
```

### Step 2: Complete Tool Definitions (from previous activities)

```python
from typing import Annotated
import wikipedia
from langchain_core.tools import tool
import pandas as pd
import os
from langchain_experimental.utilities import PythonREPL

# Wikipedia Tool
@tool
def wikipedia_tool(
    query: Annotated[str, "The Wikipedia search to execute to find key summary information."],
):
    """Use this to search Wikipedia for factual information."""
    try:
        results = wikipedia.search(query)
        if not results:
            return "No results found on Wikipedia."
        title = results[0]
        summary = wikipedia.summary(title, sentences=8, auto_suggest=False, redirect=True)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    return f"Successfully executed:\nWikipedia summary: {summary}"

# Stock Data Tool
@tool
def stock_data_tool(
    company_ticker: Annotated[str, "The ticker symbol of the company to retrieve their stock performance data."], 
    num_days: Annotated[int, "The number of business days of stock data required to respond to the user query."]
) -> str:
    """Use this to look-up stock performance data for companies to retrieve a table from a CSV. 
    You may need to convert company names into ticker symbols to call this function, 
    e.g, Apple Inc. -> AAPL, and you may need to convert weeks, months, and years, into days."""
    
    file_path = f"data/{company_ticker}.csv"
    if os.path.exists(file_path) is False:
        return f"Sorry, but data for company {company_ticker} is not available. Please try Apple, Amazon, Meta, Microsoft, Netflix, or Tesla."
    
    stock_df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
    stock_df.index = stock_df.index.date
    
    max_num_days = (stock_df.index.max() - stock_df.index.min()).days
    if num_days > max_num_days:
        return "Sorry, but this time period exceeds the data available. Please reduce it to continue."
    
    final_date = stock_df.index.max()
    filtered_df = stock_df[stock_df.index > (final_date - pd.Timedelta(days=num_days))]
    
    return f"Successfully executed the stock performance data retrieval tool to retrieve the last *{num_days} days* of data for company **{company_ticker}**:\n\n{filtered_df.to_markdown()}"

# Python REPL Tool
repl = PythonREPL()

@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user. 
    The chart should be displayed using `plt.show()`."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    return f"Successfully executed the Python REPL tool.\n\nPython code executed:\n```python\n{code}\n```\n\nCode output:\n```\n{result}\n```"
```

### Step 3: Complete Multi-Agent Swarm Implementation

```python
from langchain_openai import ChatOpenAI
from langgraph_swarm import create_handoff_tool

llm = ChatOpenAI(model="gpt-4o-mini")

# Create handoff tools for agent-to-agent communication
transfer_to_analyst = create_handoff_tool(
    agent_name="analyst",
    description="Transfer user to the analyst assistant, who can create visualizations of provided data.",
)

transfer_to_researcher = create_handoff_tool(
    agent_name="researcher", 
    description="Transfer user to the researcher assistant, who can retrieve Wikipedia summaries or load stock performance data from CSV files.",
)

# Create specialized agents
from langgraph.prebuilt import create_react_agent

# Researcher: Specializes in information gathering
research_agent = create_react_agent(
    llm,
    tools=[wikipedia_tool, stock_data_tool, transfer_to_analyst],
    prompt="You provide summaries from Wikipedia, and can load raw, numerical stock performance data from CSV files.",
    name="researcher"
)

# Analyst: Specializes in data visualization and analysis
analyst_agent = create_react_agent(
    llm,
    [python_repl_tool, transfer_to_researcher],
    prompt="You generate plots of stock performance data provided by another assistant.",
    name="analyst"
)

# Create the swarm system
from langgraph_swarm import create_swarm
from langgraph.checkpoint.memory import InMemorySaver

config = {"configurable": {"thread_id": "1", "user_id": "1"}}
checkpointer = InMemorySaver()

swarm = create_swarm(
    agents=[research_agent, analyst_agent],
    default_active_agent="researcher"
).compile(checkpointer=checkpointer)
```

### Step 4: Testing Implementation

```python
# Visualize the swarm
swarm

# Test different scenarios
from course_helper_functions import pretty_print_messages

# Test 1: Simple research (no handoff needed)
for chunk in swarm.stream(
    {"messages": [{"role": "user", "content": "Tell me about Apple Inc."}]}, config
):
    pretty_print_messages(chunk)

# Test 2: Complex workflow requiring handoff
for chunk in swarm.stream(
    {"messages": [{"role": "user", "content": "Research Tesla and create a stock price chart"}]}, config
):
    pretty_print_messages(chunk)
```

## 🧠 Deep Dive: Multi-Agent Swarm Architecture

### 1. Handoff Tool Implementation Analysis

```python
transfer_to_analyst = create_handoff_tool(
    agent_name="analyst",
    description="Transfer user to the analyst assistant, who can create visualizations of provided data."
)
```

**How Handoff Tools Work:**

1. **Tool Registration**: Handoff tools are registered like any other tool in the agent's toolkit
2. **LLM Decision**: The current agent's LLM decides when to use the handoff tool based on the description
3. **Context Transfer**: All conversation context is transferred to the target agent
4. **Control Transfer**: The target agent becomes active and can use its specialized tools

**Handoff Tool Architecture:**
```python
# Internal structure (conceptual)
class HandoffTool:
    def __init__(self, agent_name, description):
        self.agent_name = agent_name
        self.description = description
        
    def invoke(self, input_data):
        # Signal swarm to transfer control
        return {
            "type": "handoff",
            "target_agent": self.agent_name,
            "context": input_data
        }
```

**Handoff Decision Flow:**
```python
# Researcher agent receives: "Create a chart of Apple stock"
# LLM reasoning process:
# 1. "User wants a chart - this is visualization task"
# 2. "I have tools: wikipedia_tool, stock_data_tool, transfer_to_analyst"
# 3. "I don't have visualization capabilities"
# 4. "transfer_to_analyst description says it handles visualizations"
# 5. "I should call transfer_to_analyst tool"

researcher_decision = {
    "tool_calls": [{
        "name": "transfer_to_analyst",
        "args": {"context": "User wants chart of Apple stock data"}
    }]
}
```

### 2. Agent Specialization Strategy

**Researcher Agent Design:**
```python
research_agent = create_react_agent(
    llm,
    tools=[wikipedia_tool, stock_data_tool, transfer_to_analyst],
    prompt="You provide summaries from Wikipedia, and can load raw, numerical stock performance data from CSV files.",
    name="researcher"
)
```

**Design Principles:**
- **Focused Toolset**: Only information gathering tools
- **Clear Role Definition**: Explicit about what the agent does
- **Handoff Capability**: Can transfer when specialized skills needed
- **Domain Expertise**: Optimized prompts for research tasks

**Analyst Agent Design:**
```python
analyst_agent = create_react_agent(
    llm,
    [python_repl_tool, transfer_to_researcher],
    prompt="You generate plots of stock performance data provided by another assistant.",
    name="analyst"
)
```

**Design Principles:**
- **Visualization Focus**: Only has data analysis and visualization tools
- **Dependency Recognition**: Knows it needs data from other agents
- **Specialized Prompting**: Optimized for analytical tasks
- **Complementary Skills**: Fills gaps left by researcher

### 3. Swarm Coordination Mechanism

```python
swarm = create_swarm(
    agents=[research_agent, analyst_agent],
    default_active_agent="researcher"
).compile(checkpointer=checkpointer)
```

**Swarm System Architecture:**

```
┌─────────────────────────────────────────────────────┐
│                  Swarm Controller                   │
├─────────────────────────────────────────────────────┤
│  • Agent Registry: [researcher, analyst]           │
│  • Active Agent Tracking                           │  
│  • Handoff Management                              │
│  • Shared Memory Coordination                     │
└─────────────────┬───────────────────────────────────┘
                  │
    ┌─────────────┼─────────────┐
    │             │             │
┌───▼───┐    ┌───▼───┐    ┌───▼───┐
│Memory │    │Agent A│    │Agent B│
│Store  │    │Tools: │    │Tools: │
│       │    │ • T1  │    │ • T3  │
│Thread │    │ • T2  │    │ • T4  │
│ ID: 1 │    │ • H→B │    │ • H→A │
└───────┘    └───────┘    └───────┘
```

**Key Components:**

1. **Agent Registry**: Central repository of all available agents
2. **Active Agent Management**: Tracks which agent is currently handling requests
3. **Handoff Router**: Processes handoff tool calls and transfers control
4. **Shared Memory**: Ensures all agents have access to conversation context
5. **Default Entry Point**: Determines where user requests initially go

### 4. Memory and Context Management

```python
checkpointer = InMemorySaver()
config = {"configurable": {"thread_id": "1", "user_id": "1"}}
```

**Memory Architecture:**

```python
# Conceptual memory structure
memory_store = {
    "thread_1": {
        "messages": [
            {"role": "user", "content": "Research Apple"},
            {"role": "assistant", "name": "researcher", "content": "I'll research Apple Inc."},
            {"role": "assistant", "name": "researcher", "tool_calls": [...]},
            {"role": "tool", "content": "Apple Inc. is..."},
            {"role": "assistant", "name": "researcher", "tool_calls": [{"name": "transfer_to_analyst"}]},
            {"role": "assistant", "name": "analyst", "content": "I'll create a visualization"},
            # ... continued conversation
        ],
        "active_agent": "analyst",
        "metadata": {
            "user_id": "1",
            "created": "timestamp",
            "last_activity": "timestamp"
        }
    }
}
```

**Context Sharing Benefits:**
- **Seamless Handoffs**: Target agents receive full conversation history
- **Reference Resolution**: Agents can refer to previous mentions ("that data", "the company")
- **Progressive Workflows**: Multi-step tasks across agent boundaries
- **User Experience**: No need to repeat information after handoffs

## 🧪 Comprehensive Testing Suite

### Test Suite 1: Agent Specialization Validation

```python
def test_agent_specialization():
    """Validate that agents properly specialize and only handle appropriate requests"""
    
    specialization_matrix = [
        # (request, expected_agent, should_handoff)
        ("Tell me about Microsoft", "researcher", False),
        ("Get Apple stock data", "researcher", False), 
        ("What is Tesla's business model?", "researcher", False),
        ("Create a chart of Apple stock", "researcher", True),
        ("Plot Tesla performance", "researcher", True),
        ("Show me a graph", "researcher", True),
        ("Visualize the data", "researcher", True),
    ]
    
    results = {}
    
    for request, expected_starter, should_handoff in specialization_matrix:
        print(f"\n--- Specialization Test: {request} ---")
        
        # Reset memory for clean test
        test_config = {"configurable": {"thread_id": f"test_{hash(request)}", "user_id": "test"}}
        
        chunks = list(swarm.stream({
            "messages": [{"role": "user", "content": request}]
        }, test_config))
        
        # Analyze agent behavior
        agents_used = set()
        handoff_occurred = False
        tools_used = []
        
        for chunk in chunks:
            if "messages" in chunk:
                for message in chunk["messages"]:
                    # Track agent usage
                    if message.get("name"):
                        agents_used.add(message["name"])
                    
                    # Track tool usage and handoffs
                    if message.get("role") == "assistant" and "tool_calls" in message:
                        for tool_call in message["tool_calls"]:
                            tools_used.append(tool_call["name"])
                            if "transfer_to" in tool_call["name"]:
                                handoff_occurred = True
        
        # Validate behavior
        started_correctly = expected_starter in agents_used or len(agents_used) == 0
        handoff_appropriate = should_handoff == handoff_occurred
        
        results[request] = {
            "agents_used": list(agents_used),
            "handoff_occurred": handoff_occurred,
            "tools_used": tools_used,
            "started_correctly": started_correctly,
            "handoff_appropriate": handoff_appropriate,
            "overall_correct": started_correctly and handoff_appropriate
        }
        
        print(f"Agents used: {list(agents_used)}")
        print(f"Handoff occurred: {handoff_occurred}")
        print(f"Tools used: {tools_used}")
        print(f"Assessment: {'✅' if results[request]['overall_correct'] else '❌'}")
    
    # Summary
    correct_count = sum(1 for r in results.values() if r["overall_correct"])
    print(f"\n=== Specialization Test Summary ===")
    print(f"Correct behavior: {correct_count}/{len(results)} ({correct_count/len(results)*100:.1f}%)")
    
    return results

# Run specialization tests
specialization_results = test_agent_specialization()
```

### Test Suite 2: Multi-Turn Conversation Flow

```python
def test_conversation_continuity():
    """Test that agents maintain context across multiple turns and handoffs"""
    
    conversation_scenarios = [
        {
            "name": "Research to Visualization Flow",
            "turns": [
                ("Tell me about Apple Inc.", "researcher"),
                ("What's their ticker symbol?", "researcher"), 
                ("Get their recent stock data", "researcher"),
                ("Now create a chart of that data", "analyst"),
                ("Make the chart title more descriptive", "analyst")
            ],
            "expected_handoffs": 1,
            "expected_context_retention": ["apple", "aapl"]
        },
        {
            "name": "Complex Analysis Workflow",
            "turns": [
                ("Research Tesla's business", "researcher"),
                ("Get their stock performance for last month", "researcher"),
                ("Create a visualization showing the trend", "analyst"),
                ("Add a second chart comparing to industry average", "analyst"),
                ("What was Tesla's highest stock price in that data?", "researcher")
            ],
            "expected_handoffs": 2,
            "expected_context_retention": ["tesla", "tsla", "stock"]
        }
    ]
    
    for scenario in conversation_scenarios:
        print(f"\n=== Conversation Test: {scenario['name']} ===")
        
        # Use dedicated thread for this scenario
        scenario_config = {"configurable": {"thread_id": f"scenario_{scenario['name']}", "user_id": "test"}}
        
        handoff_count = 0
        agents_sequence = []
        context_mentions = []
        
        for i, (turn, expected_agent) in enumerate(scenario["turns"]):
            print(f"\nTurn {i+1}: {turn}")
            
            chunks = list(swarm.stream({
                "messages": [{"role": "user", "content": turn}]
            }, scenario_config))
            
            # Analyze turn
            turn_agents = set()
            turn_handoffs = 0
            final_response = ""
            
            for chunk in chunks:
                if "messages" in chunk:
                    for message in chunk["messages"]:
                        if message.get("name"):
                            turn_agents.add(message["name"])
                        
                        if message.get("role") == "assistant":
                            if "tool_calls" in message:
                                for tool_call in message["tool_calls"]:
                                    if "transfer_to" in tool_call["name"]:
                                        turn_handoffs += 1
                                        handoff_count += 1
                            elif "content" in message:
                                final_response = message["content"]
            
            agents_sequence.extend(list(turn_agents))
            
            # Check context retention
            if i > 0:  # After first turn
                for context_item in scenario["expected_context_retention"]:
                    if context_item.lower() in final_response.lower():
                        context_mentions.append((i+1, context_item))
            
            print(f"  Active agents: {list(turn_agents)}")
            print(f"  Handoffs this turn: {turn_handoffs}")
            if final_response:
                print(f"  Response preview: {final_response[:100]}...")
        
        # Scenario assessment
        print(f"\nScenario Assessment:")
        print(f"Total handoffs: {handoff_count} (expected: {scenario['expected_handoffs']})")
        print(f"Agent sequence: {agents_sequence}")
        print(f"Context mentions: {context_mentions}")
        
        handoff_correct = handoff_count >= scenario["expected_handoffs"]
        context_correct = len(context_mentions) >= len(scenario["expected_context_retention"])
        
        print(f"Handoffs: {'✅' if handoff_correct else '❌'}")
        print(f"Context retention: {'✅' if context_correct else '❌'}")

# Run conversation continuity tests
test_conversation_continuity()
```

### Test Suite 3: Performance and Efficiency Analysis

```python
import time

def test_swarm_performance():
    """Analyze performance characteristics of the swarm system"""
    
    performance_tests = [
        {
            "name": "Single Agent Tasks",
            "requests": [
                "Tell me about Apple Inc.",
                "Get Microsoft stock data for 7 days",
                "What is Tesla's business model?"
            ],
            "expected_handoffs": 0,
            "expected_avg_time": 8.0
        },
        {
            "name": "Simple Handoff Tasks", 
            "requests": [
                "Create a chart of Apple stock prices",
                "Plot Tesla's performance",
                "Visualize Microsoft stock data"
            ],
            "expected_handoffs": 1,
            "expected_avg_time": 12.0
        },
        {
            "name": "Complex Multi-Agent Tasks",
            "requests": [
                "Research Amazon and create a comprehensive analysis with charts",
                "Tell me about Google's business and visualize their stock trend"
            ],
            "expected_handoffs": 1,
            "expected_avg_time": 18.0
        }
    ]
    
    results = {}
    
    for test_category in performance_tests:
        print(f"\n=== Performance Test: {test_category['name']} ===")
        
        times = []
        handoff_counts = []
        
        for request in test_category["requests"]:
            print(f"\nTesting: {request}")
            
            # Fresh thread for each test
            test_config = {"configurable": {"thread_id": f"perf_{hash(request)}", "user_id": "test"}}
            
            start_time = time.time()
            
            chunks = list(swarm.stream({
                "messages": [{"role": "user", "content": request}]
            }, test_config))
            
            execution_time = time.time() - start_time
            times.append(execution_time)
            
            # Count handoffs
            handoffs = 0
            agents_used = set()
            
            for chunk in chunks:
                if "messages" in chunk:
                    for message in chunk["messages"]:
                        if message.get("name"):
                            agents_used.add(message["name"])
                        
                        if message.get("role") == "assistant" and "tool_calls" in message:
                            for tool_call in message["tool_calls"]:
                                if "transfer_to" in tool_call["name"]:
                                    handoffs += 1
            
            handoff_counts.append(handoffs)
            
            print(f"  Time: {execution_time:.2f}s")
            print(f"  Handoffs: {handoffs}")
            print(f"  Agents: {list(agents_used)}")
        
        # Calculate category statistics
        avg_time = sum(times) / len(times)
        avg_handoffs = sum(handoff_counts) / len(handoff_counts)
        
        results[test_category["name"]] = {
            "avg_time": avg_time,
            "avg_handoffs": avg_handoffs,
            "expected_time": test_category["expected_avg_time"],
            "expected_handoffs": test_category["expected_handoffs"],
            "time_efficiency": avg_time <= test_category["expected_avg_time"],
            "handoff_accuracy": abs(avg_handoffs - test_category["expected_handoffs"]) <= 0.5
        }
        
        print(f"\nCategory Results:")
        print(f"  Average time: {avg_time:.2f}s (expected: {test_category['expected_avg_time']:.2f}s)")
        print(f"  Average handoffs: {avg_handoffs:.1f} (expected: {test_category['expected_handoffs']})")
        print(f"  Time efficiency: {'✅' if results[test_category['name']]['time_efficiency'] else '❌'}")
        print(f"  Handoff accuracy: {'✅' if results[test_category['name']]['handoff_accuracy'] else '❌'}")
    
    return results

# Run performance analysis
performance_results = test_swarm_performance()
```

### Test Suite 4: Error Handling and Recovery

```python
def test_swarm_error_handling():
    """Test how the swarm handles various error conditions"""
    
    error_scenarios = [
        {
            "name": "Invalid Stock Ticker",
            "request": "Get stock data for INVALIDTICKER and create a chart",
            "expected_behavior": "graceful_error_with_handoff"
        },
        {
            "name": "Broken Python Code Request",
            "request": "Execute this broken code: print(undefined_var)",
            "expected_behavior": "error_handling_by_analyst"
        },
        {
            "name": "Ambiguous Handoff Scenario",
            "request": "Help me with some data stuff",
            "expected_behavior": "clarification_or_stay_with_researcher"
        },
        {
            "name": "Circular Handoff Risk",
            "request": "I need help with Apple information and also want charts but not sure what",
            "expected_behavior": "avoid_infinite_handoffs"
        },
        {
            "name": "Agent Capability Mismatch",
            "request": "Create a chart without any data",
            "expected_behavior": "handoff_to_researcher_for_data"
        }
    ]
    
    for scenario in error_scenarios:
        print(f"\n=== Error Test: {scenario['name']} ===")
        print(f"Request: {scenario['request']}")
        print(f"Expected: {scenario['expected_behavior']}")
        
        error_config = {"configurable": {"thread_id": f"error_{hash(scenario['name'])}", "user_id": "test"}}
        
        try:
            chunks = list(swarm.stream({
                "messages": [{"role": "user", "content": scenario['request']}]
            }, error_config))
            
            # Analyze error handling
            error_messages = []
            handoff_attempts = []
            final_response = None
            agents_involved = set()
            
            for chunk in chunks:
                if "messages" in chunk:
                    for message in chunk["messages"]:
                        if message.get("name"):
                            agents_involved.add(message["name"])
                        
                        if message.get("role") == "tool":
                            content = message.get("content", "")
                            if any(error_word in content.lower() for error_word in ["error", "failed", "invalid", "sorry"]):
                                error_messages.append(content[:100] + "...")
                        
                        elif message.get("role") == "assistant":
                            if "tool_calls" in message:
                                for tool_call in message["tool_calls"]:
                                    if "transfer_to" in tool_call["name"]:
                                        handoff_attempts.append(tool_call["name"])
                            elif "content" in message:
                                final_response = message["content"]
            
            # Assess error handling quality
            print(f"Agents involved: {list(agents_involved)}")
            print(f"Error messages: {len(error_messages)}")
            print(f"Handoff attempts: {len(handoff_attempts)}")
            
            if error_messages:
                print(f"Sample error: {error_messages[0]}")
            
            if final_response:
                print(f"Final response: {final_response[:150]}...")
            
            # Quality assessment
            handled_gracefully = bool(final_response) and len(final_response) > 50
            avoided_loops = len(handoff_attempts) <= 2  # Reasonable handoff limit
            
            print(f"Handled gracefully: {'✅' if handled_gracefully else '❌'}")
            print(f"Avoided handoff loops: {'✅' if avoided_loops else '❌'}")
            
        except Exception as e:
            print(f"❌ Unhandled exception: {e}")

# Run error handling tests
test_swarm_error_handling()
```

### Test Suite 5: Collaborative Workflow Analysis

```python
def analyze_collaborative_workflows():
    """Analyze how well agents collaborate on complex tasks"""
    
    collaboration_scenarios = [
        {
            "name": "Research-Analysis Pipeline",
            "request": "Research Apple Inc., get their recent stock data, and create a comprehensive visualization with trend analysis",
            "expected_flow": ["researcher", "analyst"],
            "expected_tools": ["wikipedia_tool", "stock_data_tool", "python_repl_tool"],
            "collaboration_quality_indicators": [
                "context_transfer",
                "data_continuity", 
                "comprehensive_analysis"
            ]
        },
        {
            "name": "Iterative Refinement",
            "initial_request": "Show me Tesla's stock performance",
            "follow_ups": [
                "Make it more detailed with company background",
                "Add technical indicators to the chart",
                "Compare with industry benchmarks"
            ],
            "expected_collaboration": "multiple_handoffs_with_context"
        },
        {
            "name": "Complex Multi-Step Analysis",
            "request": "I'm considering investing in Microsoft. Give me their company overview, recent stock trends, volatility analysis, and visual charts showing key metrics",
            "expected_components": [
                "company_research",
                "stock_data_retrieval",
                "statistical_analysis",
                "visualization_creation"
            ]
        }
    ]
    
    for scenario in collaboration_scenarios:
        if scenario["name"] == "Iterative Refinement":
            # Handle multi-turn scenario
            print(f"\n=== Collaboration Analysis: {scenario['name']} ===")
            
            collab_config = {"configurable": {"thread_id": f"collab_{scenario['name']}", "user_id": "test"}}
            
            # Initial request
            print(f"Initial: {scenario['initial_request']}")
            chunks = list(swarm.stream({
                "messages": [{"role": "user", "content": scenario['initial_request']}]
            }, collab_config))
            
            # Follow-up requests
            for i, follow_up in enumerate(scenario["follow_ups"]):
                print(f"\nFollow-up {i+1}: {follow_up}")
                chunks = list(swarm.stream({
                    "messages": [{"role": "user", "content": follow_up}]
                }, collab_config))
                
                # Analyze handoffs and context usage
                handoffs_in_turn = 0
                context_references = 0
                
                for chunk in chunks:
                    if "messages" in chunk:
                        for message in chunk["messages"]:
                            if message.get("role") == "assistant":
                                if "tool_calls" in message:
                                    for tool_call in message["tool_calls"]:
                                        if "transfer_to" in tool_call["name"]:
                                            handoffs_in_turn += 1
                                
                                elif "content" in message:
                                    # Check for context references
                                    content = message["content"].lower()
                                    context_words = ["previous", "earlier", "mentioned", "discussed", "that", "those"]
                                    context_references += sum(1 for word in context_words if word in content)
                
                print(f"  Handoffs: {handoffs_in_turn}")
                print(f"  Context references: {context_references}")
        
        else:
            # Handle single-request scenarios
            print(f"\n=== Collaboration Analysis: {scenario['name']} ===")
            
            collab_config = {"configurable": {"thread_id": f"collab_{scenario['name']}", "user_id": "test"}}
            
            chunks = list(swarm.stream({
                "messages": [{"role": "user", "content": scenario['request']}]
            }, collab_config))
            
            # Analyze collaboration
            workflow_sequence = []
            tools_used = []
            agents_used = set()
            handoff_count = 0
            
            for chunk in chunks:
                if "messages" in chunk:
                    for message in chunk["messages"]:
                        if message.get("name"):
                            agents_used.add(message["name"])
                        
                        if message.get("role") == "assistant" and "tool_calls" in message:
                            for tool_call in message["tool_calls"]:
                                if "transfer_to" in tool_call["name"]:
                                    handoff_count += 1
                                    target = tool_call["name"].replace("transfer_to_", "")
                                    workflow_sequence.append(f"handoff_to_{target}")
                                else:
                                    tools_used.append(tool_call["name"])
                                    workflow_sequence.append(f"tool_{tool_call['name']}")
            
            # Assess collaboration quality
            print(f"Workflow sequence: {workflow_sequence}")
            print(f"Agents involved: {list(agents_used)}")
            print(f"Tools used: {tools_used}")
            print(f"Handoff count: {handoff_count}")
            
            # Check against expectations
            if "expected_flow" in scenario:
                flow_correct = all(agent in agents_used for agent in scenario["expected_flow"])
                print(f"Expected agent flow: {'✅' if flow_correct else '❌'}")
            
            if "expected_tools" in scenario:
                tools_correct = all(tool in tools_used for tool in scenario["expected_tools"])
                print(f"Expected tools used: {'✅' if tools_correct else '❌'}")

# Run collaborative workflow analysis
analyze_collaborative_workflows()
```

## 🎓 Educational Insights and Advanced Patterns

### Why Swarm Architecture Matters

1. **Modular Intelligence**: Each agent can be optimized independently
2. **Scalable Complexity**: Add new agents without modifying existing ones
3. **Clear Boundaries**: Well-defined responsibilities prevent confusion
4. **Efficient Resource Usage**: Agents only load tools they need
5. **Fault Isolation**: Problems in one agent don't affect others

### Comparison: Swarm vs Single Agent Performance

```python
def comparative_analysis():
    """Compare swarm vs single agent approaches"""
    
    # Single agent equivalent (for comparison)
    single_agent = create_react_agent(
        llm,
        [wikipedia_tool, stock_data_tool, python_repl_tool],
        prompt="""You are a comprehensive assistant that can research companies using Wikipedia, 
        retrieve stock data from CSV files, and create visualizations using Python. 
        Use appropriate tools based on user requests.""",
        name="comprehensive_agent"
    )
    
    comparison_tests = [
        "Tell me about Apple Inc.",
        "Create a chart of Tesla stock prices",
        "Research Microsoft and show their stock performance",
        "Plot Google's stock data with trend analysis"
    ]
    
    results = {"swarm": [], "single": []}
    
    for request in comparison_tests:
        print(f"\n--- Comparing: {request} ---")
        
        # Test swarm
        swarm_config = {"configurable": {"thread_id": f"swarm_{hash(request)}", "user_id": "test"}}
        start_time = time.time()
        swarm_chunks = list(swarm.stream({"messages": [{"role": "user", "content": request}]}, swarm_config))
        swarm_time = time.time() - start_time
        
        # Test single agent
        start_time = time.time()
        single_chunks = list(single_agent.stream({"messages": [{"role": "user", "content": request}]}))
        single_time = time.time() - start_time
        
        # Analyze tool usage
        swarm_tools = count_tool_usage(swarm_chunks)
        single_tools = count_tool_usage(single_chunks)
        
        results["swarm"].append({
            "request": request,
            "time": swarm_time,
            "tools": swarm_tools
        })
        
        results["single"].append({
            "request": request, 
            "time": single_time,
            "tools": single_tools
        })
        
        print(f"Swarm: {swarm_time:.2f}s, Tools: {swarm_tools}")
        print(f"Single: {single_time:.2f}s, Tools: {single_tools}")
        print(f"Difference: {swarm_time - single_time:+.2f}s")
    
    # Summary analysis
    swarm_avg = sum(r["time"] for r in results["swarm"]) / len(results["swarm"])
    single_avg = sum(r["time"] for r in results["single"]) / len(results["single"])
    
    print(f"\n=== Performance Summary ===")
    print(f"Swarm average: {swarm_avg:.2f}s")
    print(f"Single agent average: {single_avg:.2f}s")
    print(f"Swarm overhead: {swarm_avg - single_avg:+.2f}s ({((swarm_avg - single_avg) / single_avg * 100):+.1f}%)")
    
    return results

def count_tool_usage(chunks):
    """Helper function to count tool usage"""
    tools = []
    for chunk in chunks:
        if "messages" in chunk:
            for message in chunk["messages"]:
                if message.get("role") == "assistant" and "tool_calls" in message:
                    for tool_call in message["tool_calls"]:
                        if not "transfer_to" in tool_call["name"]:  # Exclude handoff tools
                            tools.append(tool_call["name"])
    return tools

# Run comparative analysis
# comparison_results = comparative_analysis()
```

### Advanced Swarm Patterns

#### Pattern 1: Hierarchical Swarm with Coordinator

```python
def create_hierarchical_swarm():
    """Create a swarm with a coordinator agent that manages other agents"""
    
    # Create coordinator handoff tools
    transfer_to_coordinator = create_handoff_tool(
        agent_name="coordinator",
        description="Transfer to coordinator for task planning and result synthesis"
    )
    
    # Enhanced agents with coordinator access
    enhanced_researcher = create_react_agent(
        llm,
        [wikipedia_tool, stock_data_tool, transfer_to_analyst, transfer_to_coordinator],
        prompt="You gather information and can transfer to analyst for visualization or coordinator for task coordination.",
        name="researcher"
    )
    
    enhanced_analyst = create_react_agent(
        llm,
        [python_repl_tool, transfer_to_researcher, transfer_to_coordinator],
        prompt="You create visualizations and can transfer to researcher for data or coordinator for task coordination.",
        name="analyst"
    )
    
    # Coordinator agent
    coordinator_agent = create_react_agent(
        llm,
        [transfer_to_researcher, transfer_to_analyst],
        prompt="""You are a coordinator that plans complex tasks and synthesizes results from other agents.
        You break down complex requests into steps and route to appropriate specialists.
        You also synthesize final responses from multiple agent outputs.""",
        name="coordinator"
    )
    
    # Create hierarchical swarm
    hierarchical_swarm = create_swarm(
        agents=[coordinator_agent, enhanced_researcher, enhanced_analyst],
        default_active_agent="coordinator"
    ).compile(checkpointer=InMemorySaver())
    
    return hierarchical_swarm

# Create and test hierarchical swarm
# hierarchical_swarm = create_hierarchical_swarm()
```

#### Pattern 2: Specialized Swarm with Domain Experts

```python
def create_domain_expert_swarm():
    """Create a swarm with highly specialized domain experts"""
    
    # Create additional specialized tools
    @tool
    def financial_analysis_tool(
        ticker: Annotated[str, "Stock ticker for analysis"],
        metrics: Annotated[str, "Comma-separated list of metrics to calculate"]
    ):
        """Advanced financial analysis with metrics like volatility, RSI, moving averages."""
        # Implementation would include sophisticated financial calculations
        return f"Financial analysis for {ticker}: {metrics}"
    
    @tool
    def market_news_tool(
        company: Annotated[str, "Company name for news search"],
        days: Annotated[int, "Number of days to search back"]
    ):
        """Retrieve recent market news and sentiment for a company."""
        # Implementation would connect to news APIs
        return f"Recent news for {company} over {days} days"
    
    # Create handoff tools for new specialists
    transfer_to_financial_analyst = create_handoff_tool(
        agent_name="financial_analyst",
        description="Transfer for advanced financial analysis, metrics calculation, and market insights"
    )
    
    transfer_to_market_researcher = create_handoff_tool(
        agent_name="market_researcher", 
        description="Transfer for market news, sentiment analysis, and industry insights"
    )
    
    # Create specialized agents
    financial_analyst = create_react_agent(
        llm,
        [financial_analysis_tool, python_repl_tool, stock_data_tool, transfer_to_market_researcher],
        prompt="You are a quantitative financial analyst specializing in technical analysis, risk metrics, and market data interpretation.",
        name="financial_analyst"
    )
    
    market_researcher = create_react_agent(
        llm,
        [market_news_tool, wikipedia_tool, transfer_to_financial_analyst],
        prompt="You specialize in market research, news analysis, and industry trends. You provide context for financial decisions.",
        name="market_researcher"
    )
    
    # Create domain expert swarm
    expert_swarm = create_swarm(
        agents=[market_researcher, financial_analyst],
        default_active_agent="market_researcher"
    ).compile(checkpointer=InMemorySaver())
    
    return expert_swarm

# Create domain expert swarm
# expert_swarm = create_domain_expert_swarm()
```

#### Pattern 3: Dynamic Swarm with Runtime Agent Creation

```python
class DynamicSwarm:
    """Swarm that can create new agents at runtime based on needs"""
    
    def __init__(self, base_agents):
        self.base_agents = {agent.name: agent for agent in base_agents}
        self.dynamic_agents = {}
        self.checkpointer = InMemorySaver()
    
    def create_specialist_agent(self, domain, tools, prompt):
        """Create a new specialist agent for a specific domain"""
        
        # Create handoff tools for the new agent
        handoff_tools = []
        for existing_agent_name in self.base_agents.keys():
            handoff_tool = create_handoff_tool(
                agent_name=existing_agent_name,
                description=f"Transfer to {existing_agent_name} for their specialized tasks"
            )
            handoff_tools.append(handoff_tool)
        
        # Create the new agent
        specialist = create_react_agent(
            llm,
            tools + handoff_tools,
            prompt=prompt,
            name=domain
        )
        
        self.dynamic_agents[domain] = specialist
        
        # Update existing agents to include handoff to new agent
        # This would require rebuilding the swarm with updated handoff tools
        
        return specialist
    
    def get_active_swarm(self):
        """Get current swarm with all agents"""
        all_agents = list(self.base_agents.values()) + list(self.dynamic_agents.values())
        
        return create_swarm(
            agents=all_agents,
            default_active_agent="researcher"  # Or dynamic selection
        ).compile(checkpointer=self.checkpointer)
    
    def handle_request(self, request, config):
        """Handle request with dynamic agent creation if needed"""
        
        # Simple domain detection (could be more sophisticated)
        if "legal" in request.lower() and "legal" not in self.dynamic_agents:
            legal_tools = []  # Would include legal research tools
            self.create_specialist_agent(
                "legal",
                legal_tools,
                "You are a legal research specialist focusing on corporate law and compliance."
            )
        
        # Get current swarm and process request
        current_swarm = self.get_active_swarm()
        return current_swarm.stream({"messages": [{"role": "user", "content": request}]}, config)

# Create dynamic swarm
# dynamic_swarm = DynamicSwarm([research_agent, analyst_agent])
```

## 🔧 Production Deployment Considerations

### Swarm Monitoring and Observability

```python
class SwarmMonitoringWrapper:
    """Production wrapper for swarm systems with comprehensive monitoring"""
    
    def __init__(self, swarm, monitoring_config=None):
        self.swarm = swarm
        self.monitoring_config = monitoring_config or {}
        self.metrics = {
            "total_requests": 0,
            "agent_usage": {},
            "handoff_patterns": {},
            "error_rates": {},
            "performance_metrics": {
                "avg_response_time": 0,
                "handoff_overhead": [],
                "agent_efficiency": {}
            }
        }
    
    def stream(self, input_data, config):
        """Monitored stream with full observability"""
        start_time = time.time()
        request_id = f"req_{int(start_time)}_{hash(str(input_data))}"
        
        # Pre-request logging
        self._log_request_start(request_id, input_data)
        
        try:
            chunks = []
            agent_sequence = []
            handoff_count = 0
            
            for chunk in self.swarm.stream(input_data, config):
                chunks.append(chunk)
                
                # Monitor real-time execution
                if "messages" in chunk:
                    for message in chunk["messages"]:
                        if message.get("name"):
                            agent_sequence.append(message["name"])
                        
                        if message.get("role") == "assistant" and "tool_calls" in message:
                            for tool_call in message["tool_calls"]:
                                if "transfer_to" in tool_call["name"]:
                                    handoff_count += 1
                
                yield chunk
            
            # Post-request analysis
            execution_time = time.time() - start_time
            self._update_metrics(request_id, agent_sequence, handoff_count, execution_time, True)
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._log_error(request_id, e, execution_time)
            self._update_metrics(request_id, [], 0, execution_time, False)
            raise
    
    def _log_request_start(self, request_id, input_data):
        """Log request initiation"""
        if self.monitoring_config.get("detailed_logging", False):
            print(f"[{request_id}] Request started: {str(input_data)[:100]}...")
    
    def _update_metrics(self, request_id, agent_sequence, handoff_count, execution_time, success):
        """Update comprehensive metrics"""
        self.metrics["total_requests"] += 1
        
        # Agent usage tracking
        for agent in set(agent_sequence):
            self.metrics["agent_usage"][agent] = self.metrics["agent_usage"].get(agent, 0) + 1
        
        # Handoff pattern tracking
        if handoff_count > 0:
            pattern = " → ".join(agent_sequence)
            self.metrics["handoff_patterns"][pattern] = self.metrics["handoff_patterns"].get(pattern, 0) + 1
        
        # Performance metrics
        current_avg = self.metrics["performance_metrics"]["avg_response_time"]
        total_requests = self.metrics["total_requests"]
        self.metrics["performance_metrics"]["avg_response_time"] = (
            (current_avg * (total_requests - 1) + execution_time) / total_requests
        )
        
        if handoff_count > 0:
            self.metrics["performance_metrics"]["handoff_overhead"].append(execution_time)
        
        # Agent efficiency (requests per agent)
        for agent in set(agent_sequence):
            if agent not in self.metrics["performance_metrics"]["agent_efficiency"]:
                self.metrics["performance_metrics"]["agent_efficiency"][agent] = {"requests": 0, "total_time": 0}
            
            self.metrics["performance_metrics"]["agent_efficiency"][agent]["requests"] += 1
            self.metrics["performance_metrics"]["agent_efficiency"][agent]["total_time"] += execution_time
        
        # Success/error tracking
        if not success:
            self.metrics["error_rates"]["total"] = self.metrics["error_rates"].get("total", 0) + 1
    
    def _log_error(self, request_id, error, execution_time):
        """Log errors with context"""
        error_info = {
            "request_id": request_id,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "execution_time": execution_time
        }
        
        if self.monitoring_config.get("error_logging", True):
            print(f"[ERROR] {error_info}")
    
    def get_health_metrics(self):
        """Return comprehensive health and performance metrics"""
        total_requests = self.metrics["total_requests"]
        error_count = self.metrics["error_rates"].get("total", 0)
        
        health_metrics = {
            "overall_health": "healthy" if (error_count / max(total_requests, 1)) < 0.05 else "degraded",
            "total_requests": total_requests,
            "error_rate": error_count / max(total_requests, 1),
            "avg_response_time": self.metrics["performance_metrics"]["avg_response_time"],
            "agent_utilization": self.metrics["agent_usage"],
            "most_common_pattern": max(
                self.metrics["handoff_patterns"].items(),
                key=lambda x: x[1],
                default=("direct", 0)
            ),
            "agent_efficiency": {
                agent: data["total_time"] / max(data["requests"], 1)
                for agent, data in self.metrics["performance_metrics"]["agent_efficiency"].items()
            }
        }
        
        return health_metrics
    
    def generate_performance_report(self):
        """Generate detailed performance analysis report"""
        health_metrics = self.get_health_metrics()
        
        # Advanced analysis
        handoff_efficiency = (
            sum(self.metrics["performance_metrics"]["handoff_overhead"]) / 
            max(len(self.metrics["performance_metrics"]["handoff_overhead"]), 1)
        ) if self.metrics["performance_metrics"]["handoff_overhead"] else 0
        
        report = {
            "summary": health_metrics,
            "detailed_analysis": {
                "handoff_efficiency": handoff_efficiency,
                "collaboration_patterns": dict(sorted(
                    self.metrics["handoff_patterns"].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]),
                "bottleneck_analysis": self._identify_bottlenecks(),
                "optimization_recommendations": self._generate_recommendations()
            }
        }
        
        return report
    
    def _identify_bottlenecks(self):
        """Identify performance bottlenecks"""
        bottlenecks = []
        
        # Check agent efficiency
        efficiency_metrics = self.get_health_metrics()["agent_efficiency"]
        if efficiency_metrics:
            slowest_agent = max(efficiency_metrics.items(), key=lambda x: x[1])
            if slowest_agent[1] > self.metrics["performance_metrics"]["avg_response_time"] * 1.5:
                bottlenecks.append(f"Agent {slowest_agent[0]} is significantly slower than average")
        
        # Check handoff patterns
        if self.metrics["handoff_patterns"]:
            complex_patterns = [pattern for pattern, count in self.metrics["handoff_patterns"].items() 
                             if pattern.count("→") > 2]
            if complex_patterns:
                bottlenecks.append("Complex handoff patterns detected - consider workflow optimization")
        
        return bottlenecks
    
    def _generate_recommendations(self):
        """Generate optimization recommendations"""
        recommendations = []
        
        # Agent utilization recommendations
        agent_usage = self.metrics["agent_usage"]
        if agent_usage:
            total_usage = sum(agent_usage.values())
            for agent, usage in agent_usage.items():
                utilization_rate = usage / total_usage
                if utilization_rate < 0.1:
                    recommendations.append(f"Consider removing or consolidating {agent} (low utilization: {utilization_rate:.1%})")
                elif utilization_rate > 0.8:
                    recommendations.append(f"Consider scaling or optimizing {agent} (high utilization: {utilization_rate:.1%})")
        
        # Performance recommendations
        avg_handoff_time = sum(self.metrics["performance_metrics"]["handoff_overhead"]) / max(len(self.metrics["performance_metrics"]["handoff_overhead"]), 1) if self.metrics["performance_metrics"]["handoff_overhead"] else 0
        if avg_handoff_time > self.metrics["performance_metrics"]["avg_response_time"] * 0.3:
            recommendations.append("High handoff overhead - consider agent consolidation or caching")
        
        return recommendations

# Create monitored swarm for production
monitored_swarm = SwarmMonitoringWrapper(swarm, {
    "detailed_logging": True,
    "error_logging": True,
    "performance_tracking": True
})
```

## 📊 Assessment Rubric

### Functionality (35 points)
- **Handoff Tools**: Correct creation and configuration of handoff tools (15 pts)
- **Agent Specialization**: Proper tool assignment and role definition (10 pts)  
- **Swarm Creation**: Correct swarm assembly with memory management (10 pts)

### Understanding (35 points)
- **Multi-Agent Concepts**: Explains benefits of specialization vs single agent (15 pts)
- **Handoff Logic**: Understands when and why agents transfer control (10 pts)
- **Architecture Benefits**: Explains swarm advantages and limitations (10 pts)

### Advanced Implementation (30 points)
- **Testing**: Comprehensive testing of collaboration patterns (15 pts)
- **Error Handling**: Understanding of multi-agent error scenarios (10 pts)
- **Extensions**: Implementation of advanced patterns or monitoring (5 pts)

### Total: 100 points

**Grading Scale:**
- 90-100: Excellent mastery of multi-agent systems and collaboration patterns
- 80-89: Good implementation with minor issues in handoff logic or testing
- 70-79: Basic functionality working, needs deeper understanding of agent coordination
- Below 70: Requires additional practice with multi-agent architecture concepts

## 🚀 Real-World Applications

### Enterprise Multi-Agent Systems

#### Customer Service Swarm
```python
customer_service_swarm = create_swarm([
    technical_support_agent,    # Hardware/software issues
    billing_agent,             # Account and payment issues  
    product_specialist_agent,  # Product information and recommendations
    escalation_agent          # Human handoff for complex issues
], default_active_agent="technical_support")
```

#### Financial Analysis Swarm
```python
financial_swarm = create_swarm([
    market_researcher_agent,   # Market data and trends
    risk_analyst_agent,       # Risk assessment and metrics
    portfolio_manager_agent,  # Investment recommendations
    compliance_agent         # Regulatory and compliance checks
], default_active_agent="market_researcher")
```

#### Content Creation Swarm
```python
content_swarm = create_swarm([
    research_agent,           # Topic research and fact gathering
    writer_agent,            # Content drafting and creation
    editor_agent,            # Review and improvement
    seo_specialist_agent     # SEO optimization and keywords
], default_active_agent="research")
```

## 💡 Pro Tips for Instructors

1. **Start Small**: Begin with 2-agent swarms before adding complexity
2. **Emphasize Handoff Design**: Handoff tool descriptions are critical for success
3. **Show Failure Cases**: Demonstrate what happens with poor handoff logic
4. **Memory Importance**: Stress the role of shared memory in collaboration
5. **Real-World Examples**: Connect to actual multi-agent business applications

## 🏁 Conclusion

This exercise introduces students to the fundamental concepts of multi-agent systems through the swarm architecture. Key learning outcomes include:

- **Agent Specialization**: Understanding how to create focused, expert agents
- **Handoff Mechanisms**: Learning agent-to-agent communication patterns  
- **Collaborative Workflows**: Building systems where agents work together
- **Shared Memory**: Managing context and state across multiple agents
- **Architecture Trade-offs**: Understanding when to use multi-agent vs single agent systems

**Key Architectural Insights:**
- Handoff tools enable peer-to-peer agent collaboration
- Agent specialization improves performance and maintainability
- Shared memory ensures continuity across agent boundaries
- Swarm architecture provides modular, scalable intelligence
- Multi-agent systems trade simplicity for flexibility and performance

Students are now prepared for more advanced multi-agent patterns including supervisor architectures, hierarchical systems, and complex coordination mechanisms! 🚀🐝🧠