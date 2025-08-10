# 👔 Activity 09: Supervisor Multi-Agent Systems - Master Solution Guide

## 📋 Activity Overview

**Topic:** Building hierarchical multi-agent systems with supervisor-worker architecture  
**Duration:** 45-75 minutes  
**Difficulty:** Intermediate to Advanced  
**Prerequisites:** Single agent creation, multi-agent concepts, ReAct patterns, basic coordination

## 🏆 Complete Solution

### Step 1: Environment Setup

```python
# Install required libraries
!pip install --quiet wikipedia==1.4.0 langchain-core==0.3.69 langgraph==0.5.3 langchain-openai==0.3.28 langchain-experimental==0.3.4 langgraph-supervisor==0.0.27
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

### Step 3: Complete Supervisor Multi-Agent Implementation

```python
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

llm = ChatOpenAI(model="gpt-4o-mini")

# Create specialized worker agents
research_agent = create_react_agent(
    llm,
    tools=[wikipedia_tool, stock_data_tool],
    prompt=(
        "You are a research agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with research-related tasks, including looking-up factual information and stock data. DO NOT write any code.\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="researcher"
)

analyst_agent = create_react_agent(
    llm,
    [python_repl_tool],
    prompt=(
        "You are an agent that can run arbitrary Python code.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with tasks that require running code to produce an output.\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="analyst"
)

# Create supervisor system
from langgraph_supervisor import create_supervisor
from langgraph.checkpoint.memory import InMemorySaver

config = {"configurable": {"thread_id": "1", "user_id": "1"}}
checkpointer = InMemorySaver()

supervisor = create_supervisor(
    model=llm,
    agents=[research_agent, analyst_agent],
    prompt=(
        "You are a supervisor managing two agents:\n"
        "- a research agent. Assign research and data collection tasks to this agent\n"
        "- an analyst agent. Assign the creation of visualizations via code to this agent\n"
        "Assign work to one agent at a time, do not call agents in parallel.\n"
        "Do not do any work yourself."
    ),
    add_handoff_back_messages=True,
    output_mode="full_history",
).compile(checkpointer=checkpointer)
```

### Step 4: Testing Implementation

```python
# Visualize supervisor system
supervisor

# Test different scenarios
from course_helper_functions import pretty_print_messages

# Test 1: Simple research task
for chunk in supervisor.stream(
    {"messages": [{"role": "user", "content": "Who is Apple's CEO?"}]}, config
):
    pretty_print_messages(chunk)

# Test 2: Complex multi-worker task
for chunk in supervisor.stream(
    {"messages": [{"role": "user", "content": "Research Tesla and create a stock visualization"}]}, config
):
    pretty_print_messages(chunk)
```

## 🧠 Deep Dive: Supervisor Architecture Analysis

### 1. Supervisor vs Worker Agent Design Philosophy

**Worker Agent Design Principles:**
```python
research_agent = create_react_agent(
    llm,
    tools=[wikipedia_tool, stock_data_tool],
    prompt=(
        "You are a research agent.\n"
        "- Assist ONLY with research-related tasks\n"
        "- After you're done, respond to the supervisor directly\n" 
        "- Respond ONLY with results, do NOT include ANY other text."
    ),
    name="researcher"
)
```

**Key Design Elements:**
1. **Specialized Tools**: Each worker has only tools relevant to their domain
2. **Clear Role Definition**: Explicit boundaries on what they should/shouldn't do
3. **Supervisor Communication**: Instructions to report back to supervisor
4. **Minimal Output**: Results only, no conversational fluff
5. **Named Identity**: Clear identification for supervisor routing

**Supervisor Design Principles:**
```python
supervisor = create_supervisor(
    model=llm,
    agents=[research_agent, analyst_agent],
    prompt=(
        "You are a supervisor managing two agents:\n"
        "- a research agent. Assign research and data collection tasks\n"
        "- an analyst agent. Assign visualization and code tasks\n"
        "Assign work to one agent at a time, do not call agents in parallel.\n"
        "Do not do any work yourself."
    )
)
```

**Key Design Elements:**
1. **Management Role**: Supervisor delegates, doesn't execute
2. **Agent Awareness**: Knows capabilities of each worker
3. **Sequential Processing**: One agent at a time for clear workflow
4. **Quality Control**: Reviews and synthesizes worker outputs
5. **User Interface**: Provides coherent responses to end users

### 2. Hierarchical Communication Architecture

**Message Flow Analysis:**
```python
# Conceptual message flow in supervisor system
class SupervisorMessageFlow:
    def handle_user_request(self, user_input):
        """
        1. User Request Reception
        """
        user_message = {"role": "user", "content": user_input}
        
        """
        2. Supervisor Analysis & Delegation
        """
        supervisor_decision = self.supervisor.analyze_request(user_message)
        # Supervisor thinks: "This needs research" or "This needs analysis"
        
        """
        3. Worker Assignment & Execution
        """
        if supervisor_decision.requires_research:
            worker_result = self.research_agent.execute(supervisor_decision.task)
        elif supervisor_decision.requires_analysis:
            worker_result = self.analyst_agent.execute(supervisor_decision.task)
        
        """
        4. Result Collection & Processing
        """
        supervisor_synthesis = self.supervisor.process_worker_result(worker_result)
        
        """
        5. User Response Generation
        """
        final_response = self.supervisor.generate_user_response(supervisor_synthesis)
        return final_response
```

**Communication Patterns:**

1. **Vertical Communication** (Supervisor ↔ Workers):
   ```python
   # Supervisor to Worker
   supervisor_assignment = {
       "role": "assistant", 
       "content": "Please research Apple Inc. company information",
       "agent_assignment": "researcher"
   }
   
   # Worker to Supervisor
   worker_report = {
       "role": "assistant",
       "name": "researcher", 
       "content": "Apple Inc. is a multinational corporation..."
   }
   ```

2. **No Horizontal Communication** (Worker ≠ Worker):
   - Workers never communicate directly with each other
   - All coordination goes through supervisor
   - Ensures clear chain of command

3. **External Communication** (Supervisor ↔ User):
   ```python
   # User to System
   user_request = {"role": "user", "content": "Research Apple and create chart"}
   
   # System to User (via Supervisor)
   final_response = {
       "role": "assistant", 
       "content": "Based on research and analysis: [comprehensive answer]"
   }
   ```

### 3. create_supervisor() Function Deep Analysis

```python
supervisor = create_supervisor(
    model=llm,                           # LLM for supervisor decisions
    agents=[research_agent, analyst_agent], # Worker agent registry
    prompt="...",                        # Supervisor behavior instructions
    add_handoff_back_messages=True,      # Show worker handoff confirmations
    output_mode="full_history",          # Conversation memory management
)
```

**Parameter Analysis:**

1. **model**: The LLM used for supervisor reasoning
   - Makes delegation decisions
   - Synthesizes worker results
   - Generates user responses

2. **agents**: Registry of available worker agents
   - Each agent has name, tools, and capabilities
   - Supervisor chooses from this pool
   - Can be extended dynamically

3. **prompt**: Supervisor behavior definition
   - Defines management style
   - Sets delegation criteria
   - Establishes quality standards

4. **add_handoff_back_messages**: Transparency control
   - Shows when workers complete tasks
   - Helps debug workflow issues
   - Provides visibility into coordination

5. **output_mode**: Memory and history management
   - "full_history": Keep all messages
   - "last_message": Only recent interaction
   - "filtered": Custom message filtering

**Internal Architecture (Conceptual):**
```python
class SupervisorInternals:
    def __init__(self, model, agents, prompt, **kwargs):
        self.model = model
        self.agents = {agent.name: agent for agent in agents}
        self.delegation_prompt = prompt
        self.workflow_state = "awaiting_user_input"
    
    def process_message(self, message):
        if self.workflow_state == "awaiting_user_input":
            return self.delegate_task(message)
        elif self.workflow_state == "awaiting_worker_result":
            return self.process_worker_result(message)
    
    def delegate_task(self, user_message):
        # Supervisor analyzes request and chooses worker
        delegation_decision = self.model.invoke([
            {"role": "system", "content": self.delegation_prompt},
            user_message
        ])
        
        # Route to appropriate worker
        selected_worker = self.choose_worker(delegation_decision)
        return selected_worker.invoke(user_message)
    
    def choose_worker(self, delegation_decision):
        # Logic to select appropriate worker based on task type
        if "research" in delegation_decision.lower():
            return self.agents["researcher"]
        elif "code" in delegation_decision.lower():
            return self.agents["analyst"]
        else:
            return self.agents[list(self.agents.keys())[0]]  # Default
```

### 4. Coordination and State Management

**State Transitions:**
```python
# Supervisor system state machine
class SupervisorStateMachine:
    states = [
        "idle",                    # Waiting for user input
        "analyzing_request",       # Supervisor deciding on delegation
        "delegating_task",         # Assigning work to worker
        "worker_executing",        # Worker performing task
        "collecting_results",      # Supervisor receiving worker output
        "synthesizing_response",   # Supervisor preparing final answer
        "responding_to_user"       # Delivering final response
    ]
    
    def transition_flow(self, user_request):
        """Complete workflow from user request to final response"""
        
        # State 1: idle → analyzing_request
        self.state = "analyzing_request"
        analysis = self.supervisor_analyzes_request(user_request)
        
        # State 2: analyzing_request → delegating_task  
        self.state = "delegating_task"
        worker_assignment = self.supervisor_delegates_to_worker(analysis)
        
        # State 3: delegating_task → worker_executing
        self.state = "worker_executing" 
        worker_result = self.worker_executes_task(worker_assignment)
        
        # State 4: worker_executing → collecting_results
        self.state = "collecting_results"
        collected_data = self.supervisor_collects_worker_output(worker_result)
        
        # State 5: collecting_results → synthesizing_response
        self.state = "synthesizing_response"
        synthesized_answer = self.supervisor_synthesizes_final_response(collected_data)
        
        # State 6: synthesizing_response → responding_to_user
        self.state = "responding_to_user"
        final_response = self.supervisor_responds_to_user(synthesized_answer)
        
        # State 7: responding_to_user → idle
        self.state = "idle"
        return final_response
```

**Memory and Context Management:**
```python
# Memory structure in supervisor systems
memory_structure = {
    "conversation_id": "thread_1",
    "messages": [
        # User input
        {"role": "user", "content": "Research Apple and create chart", "timestamp": "T1"},
        
        # Supervisor delegation
        {"role": "assistant", "content": "I'll assign the research task to the researcher", "timestamp": "T2"},
        
        # Worker execution  
        {"role": "assistant", "name": "researcher", "content": "Apple Inc. is...", "timestamp": "T3"},
        
        # Supervisor synthesis
        {"role": "assistant", "content": "Now I'll assign chart creation to analyst", "timestamp": "T4"},
        
        # Worker execution
        {"role": "assistant", "name": "analyst", "content": "Chart created...", "timestamp": "T5"},
        
        # Final response
        {"role": "assistant", "content": "Here's your complete analysis...", "timestamp": "T6"}
    ],
    "workflow_state": "completed",
    "active_worker": None,
    "pending_tasks": []
}
```

## 🧪 Comprehensive Testing Suite

### Test Suite 1: Task Delegation Accuracy

```python
def test_task_delegation_accuracy():
    """Test that supervisor correctly delegates tasks to appropriate workers"""
    
    delegation_test_matrix = [
        # (user_request, expected_primary_worker, task_complexity)
        ("Who is the CEO of Apple?", "researcher", "simple"),
        ("Tell me about Tesla's business model", "researcher", "simple"),
        ("Get Microsoft stock data for 10 days", "researcher", "simple"),
        ("Create a bar chart of [1,2,3,4,5]", "analyst", "simple"),
        ("Plot a simple line graph", "analyst", "simple"),
        ("Execute this code: print('hello')", "analyst", "simple"),
        ("Research Apple and create a stock chart", "both", "complex"),
        ("Get Tesla data and visualize the trends", "both", "complex"),
        ("Find Google's revenue and make a pie chart", "both", "complex"),
    ]
    
    results = {}
    
    for request, expected_worker, complexity in delegation_test_matrix:
        print(f"\n--- Delegation Test: {request} ---")
        print(f"Expected worker(s): {expected_worker}")
        print(f"Complexity: {complexity}")
        
        # Create unique thread for each test
        test_config = {"configurable": {"thread_id": f"delegation_{hash(request)}", "user_id": "test"}}
        
        chunks = list(supervisor.stream({
            "messages": [{"role": "user", "content": request}]
        }, test_config))
        
        # Analyze delegation behavior
        workers_used = set()
        supervisor_decisions = []
        delegation_sequence = []
        
        for chunk in chunks:
            if "messages" in chunk:
                for message in chunk["messages"]:
                    # Track worker usage
                    if message.get("name") in ["researcher", "analyst"]:
                        workers_used.add(message["name"])
                        delegation_sequence.append(message["name"])
                    
                    # Track supervisor decision-making
                    if (message.get("role") == "assistant" and 
                        not message.get("name") and
                        "content" in message):
                        supervisor_decisions.append(message["content"][:100])
        
        # Evaluate delegation accuracy
        delegation_correct = False
        if expected_worker == "both":
            delegation_correct = len(workers_used) >= 2
        elif expected_worker in workers_used:
            delegation_correct = True
        
        results[request] = {
            "workers_used": list(workers_used),
            "expected": expected_worker,
            "correct": delegation_correct,
            "sequence": delegation_sequence,
            "supervisor_decisions": len(supervisor_decisions)
        }
        
        print(f"Workers used: {list(workers_used)}")
        print(f"Delegation sequence: {delegation_sequence}")
        print(f"Assessment: {'✅' if delegation_correct else '❌'}")
    
    # Summary statistics
    correct_delegations = sum(1 for r in results.values() if r["correct"])
    total_tests = len(results)
    accuracy = correct_delegations / total_tests * 100
    
    print(f"\n=== Delegation Accuracy Summary ===")
    print(f"Correct delegations: {correct_delegations}/{total_tests}")
    print(f"Accuracy rate: {accuracy:.1f}%")
    
    return results

# Run delegation accuracy tests
delegation_results = test_task_delegation_accuracy()
```

### Test Suite 2: Multi-Step Workflow Coordination

```python
def test_multi_step_coordination():
    """Test supervisor's ability to coordinate complex multi-step workflows"""
    
    multi_step_scenarios = [
        {
            "name": "Research-to-Analysis Pipeline",
            "request": "Research Apple Inc. company information and create a visualization of their stock performance",
            "expected_sequence": ["researcher", "analyst"],
            "expected_handoffs": 1,
            "complexity": "high"
        },
        {
            "name": "Data-to-Visualization Pipeline", 
            "request": "Get Tesla stock data for 30 days and create a trend chart with moving averages",
            "expected_sequence": ["researcher", "analyst"],
            "expected_handoffs": 1,
            "complexity": "high"
        },
        {
            "name": "Multi-Company Analysis",
            "request": "Compare Apple and Microsoft - research both companies and create comparative charts",
            "expected_sequence": ["researcher", "analyst"],
            "expected_handoffs": 1,
            "complexity": "very_high"
        }
    ]
    
    for scenario in multi_step_scenarios:
        print(f"\n=== Multi-Step Test: {scenario['name']} ===")
        print(f"Request: {scenario['request']}")
        print(f"Expected sequence: {scenario['expected_sequence']}")
        
        # Use scenario-specific thread
        scenario_config = {"configurable": {"thread_id": f"multistep_{scenario['name']}", "user_id": "test"}}
        
        chunks = list(supervisor.stream({
            "messages": [{"role": "user", "content": scenario['request']}]
        }, scenario_config))
        
        # Analyze workflow coordination
        worker_sequence = []
        handoff_count = 0
        supervisor_coordination_steps = 0
        task_transitions = []
        
        previous_worker = None
        for chunk in chunks:
            if "messages" in chunk:
                for message in chunk["messages"]:
                    # Track worker sequence
                    if message.get("name") in ["researcher", "analyst"]:
                        current_worker = message["name"]
                        worker_sequence.append(current_worker)
                        
                        # Detect handoffs
                        if previous_worker and previous_worker != current_worker:
                            handoff_count += 1
                            task_transitions.append(f"{previous_worker} → {current_worker}")
                        
                        previous_worker = current_worker
                    
                    # Track supervisor coordination
                    elif (message.get("role") == "assistant" and 
                          not message.get("name") and
                          len(message.get("content", "")) > 20):
                        supervisor_coordination_steps += 1
        
        # Remove duplicates while preserving order
        unique_sequence = []
        seen = set()
        for worker in worker_sequence:
            if worker not in seen:
                unique_sequence.append(worker)
                seen.add(worker)
        
        # Evaluate coordination quality
        sequence_correct = unique_sequence == scenario["expected_sequence"]
        handoff_appropriate = handoff_count >= scenario["expected_handoffs"]
        
        print(f"Actual sequence: {unique_sequence}")
        print(f"Task transitions: {task_transitions}")
        print(f"Handoff count: {handoff_count}")
        print(f"Supervisor coordination steps: {supervisor_coordination_steps}")
        
        print(f"Sequence correct: {'✅' if sequence_correct else '❌'}")
        print(f"Handoff appropriate: {'✅' if handoff_appropriate else '❌'}")
        
        overall_success = sequence_correct and handoff_appropriate
        print(f"Overall coordination: {'✅' if overall_success else '❌'}")

# Run multi-step coordination tests
test_multi_step_coordination()
```

### Test Suite 3: Supervisor Decision Quality Analysis

```python
def test_supervisor_decision_quality():
    """Analyze quality of supervisor's task assignment decisions"""
    
    decision_quality_tests = [
        {
            "name": "Clear Research Task",
            "request": "Tell me about Apple Inc.'s business model and history", 
            "optimal_assignment": "researcher",
            "should_avoid": "analyst",
            "reasoning": "Pure information retrieval task"
        },
        {
            "name": "Clear Analysis Task",
            "request": "Create a bar chart showing values [10, 20, 30, 40, 50]",
            "optimal_assignment": "analyst", 
            "should_avoid": "researcher",
            "reasoning": "Pure visualization task"
        },
        {
            "name": "Ambiguous Task",
            "request": "Help me understand Tesla's performance",
            "optimal_assignment": "researcher",
            "acceptable_alternative": "both",
            "reasoning": "Could interpret as research or research+analysis"
        },
        {
            "name": "Complex Integrated Task",
            "request": "Analyze Microsoft's competitive position with data visualizations",
            "optimal_assignment": "both",
            "should_avoid": "single_agent_only",
            "reasoning": "Requires both research and visualization"
        }
    ]
    
    for test in decision_quality_tests:
        print(f"\n=== Decision Quality Test: {test['name']} ===")
        print(f"Request: {test['request']}")
        print(f"Optimal assignment: {test['optimal_assignment']}")
        print(f"Reasoning: {test['reasoning']}")
        
        # Test decision making
        test_config = {"configurable": {"thread_id": f"decision_{test['name']}", "user_id": "test"}}
        
        chunks = list(supervisor.stream({
            "messages": [{"role": "user", "content": test['request']}]
        }, test_config))
        
        # Analyze supervisor's decision process
        workers_assigned = set()
        supervisor_reasoning = []
        assignment_order = []
        
        for chunk in chunks:
            if "messages" in chunk:
                for message in chunk["messages"]:
                    # Track actual assignments
                    if message.get("name") in ["researcher", "analyst"]:
                        workers_assigned.add(message["name"])
                        assignment_order.append(message["name"])
                    
                    # Capture supervisor's reasoning (if visible)
                    elif (message.get("role") == "assistant" and 
                          not message.get("name") and
                          "content" in message):
                        reasoning_snippet = message["content"][:150]
                        supervisor_reasoning.append(reasoning_snippet)
        
        # Evaluate decision quality
        actual_assignment = "both" if len(workers_assigned) > 1 else list(workers_assigned)[0] if workers_assigned else "none"
        
        decision_quality = "excellent"
        if test["optimal_assignment"] == "both":
            if len(workers_assigned) < 2:
                decision_quality = "suboptimal" 
        elif test["optimal_assignment"] not in workers_assigned:
            if "acceptable_alternative" in test and actual_assignment == test["acceptable_alternative"]:
                decision_quality = "acceptable"
            else:
                decision_quality = "poor"
        
        print(f"Actual assignment: {actual_assignment}")
        print(f"Assignment order: {assignment_order}")
        print(f"Decision quality: {decision_quality}")
        
        if supervisor_reasoning:
            print(f"Supervisor reasoning sample: {supervisor_reasoning[0]}...")
        
        print(f"Assessment: {'✅' if decision_quality in ['excellent', 'acceptable'] else '❌'}")

# Run decision quality analysis
test_supervisor_decision_quality()
```

### Test Suite 4: Error Handling and Recovery

```python
def test_supervisor_error_handling():
    """Test how supervisor handles various error conditions and worker failures"""
    
    error_scenarios = [
        {
            "name": "Invalid Research Request",
            "request": "Research the company INVALIDCOMPANY123 that doesn't exist",
            "expected_behavior": "graceful_handling_with_explanation",
            "primary_worker": "researcher"
        },
        {
            "name": "Broken Code Request", 
            "request": "Execute this broken Python code: print(undefined_variable_xyz)",
            "expected_behavior": "error_reporting_and_alternative_suggestion",
            "primary_worker": "analyst"
        },
        {
            "name": "Impossible Data Request",
            "request": "Get Apple stock data for 50000 days",
            "expected_behavior": "constraint_explanation_and_alternative",
            "primary_worker": "researcher"
        },
        {
            "name": "Vague Request",
            "request": "Do something with data",
            "expected_behavior": "clarification_request_or_reasonable_assumption",
            "primary_worker": "unclear"
        }
    ]
    
    for scenario in error_scenarios:
        print(f"\n=== Error Handling Test: {scenario['name']} ===")
        print(f"Request: {scenario['request']}")
        print(f"Expected behavior: {scenario['expected_behavior']}")
        
        error_config = {"configurable": {"thread_id": f"error_{scenario['name']}", "user_id": "test"}}
        
        try:
            chunks = list(supervisor.stream({
                "messages": [{"role": "user", "content": scenario['request']}]
            }, error_config))
            
            # Analyze error handling
            error_detected = False
            recovery_attempted = False
            final_supervisor_response = None
            worker_errors = []
            
            for chunk in chunks:
                if "messages" in chunk:
                    for message in chunk["messages"]:
                        # Check for worker-level errors
                        if message.get("name") in ["researcher", "analyst"]:
                            content = message.get("content", "")
                            if any(error_indicator in content.lower() 
                                   for error_indicator in ["error", "failed", "sorry", "invalid"]):
                                error_detected = True
                                worker_errors.append(content[:100] + "...")
                        
                        # Check for supervisor's final response
                        elif (message.get("role") == "assistant" and 
                              not message.get("name") and
                              "content" in message and
                              len(message.get("content", "")) > 30):
                            final_supervisor_response = message["content"]
                            
                            # Check if supervisor provided helpful recovery
                            if error_detected:
                                recovery_keywords = ["however", "alternatively", "instead", "try", "recommend"]
                                if any(keyword in final_supervisor_response.lower() for keyword in recovery_keywords):
                                    recovery_attempted = True
            
            # Assess error handling quality
            print(f"Error detected: {'✅' if error_detected else '❌'}")
            print(f"Recovery attempted: {'✅' if recovery_attempted else '❌'}")
            
            if worker_errors:
                print(f"Worker errors: {len(worker_errors)} detected")
                print(f"Sample error: {worker_errors[0]}")
            
            if final_supervisor_response:
                print(f"Final response length: {len(final_supervisor_response)} chars")
                print(f"Response preview: {final_supervisor_response[:200]}...")
            
            # Overall error handling assessment
            if error_detected and recovery_attempted:
                print("✅ Excellent error handling - detected and recovered")
            elif error_detected and final_supervisor_response:
                print("⚠️ Basic error handling - detected but limited recovery")
            else:
                print("ℹ️ No clear error pattern detected")
                
        except Exception as e:
            print(f"❌ System exception: {e}")

# Run error handling tests
test_supervisor_error_handling()
```

### Test Suite 5: Performance and Efficiency Analysis

```python
import time

def test_supervisor_performance():
    """Analyze performance characteristics of supervisor vs direct worker usage"""
    
    performance_comparisons = [
        {
            "name": "Simple Research Task",
            "request": "Tell me about Apple Inc.",
            "expected_workers": 1,
            "complexity": "low"
        },
        {
            "name": "Simple Analysis Task", 
            "request": "Create a bar chart of [1,2,3,4,5]",
            "expected_workers": 1,
            "complexity": "low"
        },
        {
            "name": "Complex Integrated Task",
            "request": "Research Microsoft and create stock visualizations",
            "expected_workers": 2,
            "complexity": "high"
        }
    ]
    
    # Performance metrics
    supervisor_metrics = []
    direct_worker_metrics = []
    
    for test_case in performance_comparisons:
        print(f"\n=== Performance Test: {test_case['name']} ===")
        print(f"Request: {test_case['request']}")
        print(f"Expected complexity: {test_case['complexity']}")
        
        # Test 1: Supervisor system performance
        supervisor_config = {"configurable": {"thread_id": f"perf_sup_{test_case['name']}", "user_id": "test"}}
        
        start_time = time.time()
        supervisor_chunks = list(supervisor.stream({
            "messages": [{"role": "user", "content": test_case['request']}]
        }, supervisor_config))
        supervisor_time = time.time() - start_time
        
        # Analyze supervisor execution
        supervisor_workers_used = set()
        supervisor_coordination_steps = 0
        
        for chunk in supervisor_chunks:
            if "messages" in chunk:
                for message in chunk["messages"]:
                    if message.get("name") in ["researcher", "analyst"]:
                        supervisor_workers_used.add(message["name"])
                    elif (message.get("role") == "assistant" and not message.get("name")):
                        supervisor_coordination_steps += 1
        
        # Test 2: Direct worker performance (for simple tasks)
        direct_worker_time = None
        if test_case["expected_workers"] == 1:
            # Determine which worker would be used directly
            if "research" in test_case['request'].lower() or "apple" in test_case['request'].lower():
                direct_worker = research_agent
            else:
                direct_worker = analyst_agent
            
            start_time = time.time()
            direct_chunks = list(direct_worker.stream({
                "messages": [{"role": "user", "content": test_case['request']}]
            }))
            direct_worker_time = time.time() - start_time
        
        # Record metrics
        supervisor_metrics.append({
            "test": test_case['name'],
            "time": supervisor_time,
            "workers_used": len(supervisor_workers_used),
            "coordination_steps": supervisor_coordination_steps
        })
        
        if direct_worker_time:
            direct_worker_metrics.append({
                "test": test_case['name'],
                "time": direct_worker_time
            })
        
        # Display results
        print(f"Supervisor time: {supervisor_time:.2f}s")
        print(f"Workers used: {len(supervisor_workers_used)}")
        print(f"Coordination steps: {supervisor_coordination_steps}")
        
        if direct_worker_time:
            print(f"Direct worker time: {direct_worker_time:.2f}s")
            overhead = supervisor_time - direct_worker_time
            print(f"Supervisor overhead: {overhead:.2f}s ({overhead/direct_worker_time*100:.1f}%)")
        
        # Efficiency assessment
        efficiency = "high" if supervisor_time < 10 else "medium" if supervisor_time < 20 else "low"
        print(f"Overall efficiency: {efficiency}")
    
    # Performance summary
    print(f"\n=== Performance Summary ===")
    avg_supervisor_time = sum(m["time"] for m in supervisor_metrics) / len(supervisor_metrics)
    print(f"Average supervisor time: {avg_supervisor_time:.2f}s")
    
    if direct_worker_metrics:
        avg_direct_time = sum(m["time"] for m in direct_worker_metrics) / len(direct_worker_metrics)
        avg_overhead = avg_supervisor_time - avg_direct_time
        print(f"Average direct worker time: {avg_direct_time:.2f}s")
        print(f"Average supervisor overhead: {avg_overhead:.2f}s ({avg_overhead/avg_direct_time*100:.1f}%)")
    
    avg_coordination = sum(m["coordination_steps"] for m in supervisor_metrics) / len(supervisor_metrics)
    print(f"Average coordination steps: {avg_coordination:.1f}")
    
    return supervisor_metrics, direct_worker_metrics

# Run performance analysis
supervisor_perf, direct_perf = test_supervisor_performance()
```

## 🎓 Educational Insights and Advanced Patterns

### Why Supervisor Architecture is Revolutionary

1. **Organizational Clarity**: Clear hierarchy eliminates coordination confusion
2. **Quality Control**: Centralized oversight ensures consistent outputs
3. **Scalable Management**: Easy to add new workers without complexity explosion
4. **Fault Isolation**: Worker failures don't cascade to other workers
5. **User Experience**: Single point of contact for complex workflows

### Supervisor vs Swarm Architecture Detailed Comparison

```python
def architectural_comparison():
    """Detailed comparison of supervisor vs swarm approaches"""
    
    comparison_metrics = {
        "coordination_complexity": {
            "supervisor": "O(n)",      # Supervisor manages n workers
            "swarm": "O(n²)",         # n workers coordinate with each other
            "winner": "supervisor"
        },
        "single_point_of_failure": {
            "supervisor": "yes",       # Supervisor failure breaks system
            "swarm": "no",            # Distributed failure resistance
            "winner": "swarm"
        },
        "debugging_complexity": {
            "supervisor": "low",       # Clear workflow traces
            "swarm": "high",          # Complex interaction patterns
            "winner": "supervisor"
        },
        "workflow_predictability": {
            "supervisor": "high",      # Deterministic delegation
            "swarm": "medium",        # Peer negotiation variability
            "winner": "supervisor"
        },
        "implementation_complexity": {
            "supervisor": "medium",    # Need delegation logic
            "swarm": "high",          # Need coordination protocols
            "winner": "supervisor"
        }
    }
    
    print("=== Architecture Comparison ===")
    for metric, data in comparison_metrics.items():
        print(f"{metric.replace('_', ' ').title()}:")
        print(f"  Supervisor: {data['supervisor']}")
        print(f"  Swarm: {data['swarm']}")
        print(f"  Winner: {data['winner']}")
        print()
    
    return comparison_metrics

# Run architectural comparison
arch_comparison = architectural_comparison()
```

### Advanced Supervisor Patterns

#### Pattern 1: Multi-Layer Supervisor Hierarchy

```python
def create_hierarchical_supervisor_system():
    """Create a multi-layer supervisor system with department managers"""
    
    # Department-level supervisors
    research_department_supervisor = create_supervisor(
        model=llm,
        agents=[research_agent, 
                # Could add: news_researcher, academic_researcher, market_researcher
               ],
        prompt="You are a research department supervisor. Coordinate all information gathering tasks.",
        name="research_supervisor"
    )
    
    analysis_department_supervisor = create_supervisor(
        model=llm,
        agents=[analyst_agent,
                # Could add: statistical_analyst, visualization_specialist
               ],
        prompt="You are an analysis department supervisor. Coordinate all data analysis and visualization tasks.",
        name="analysis_supervisor"
    )
    
    # Top-level executive supervisor
    executive_supervisor = create_supervisor(
        model=llm,
        agents=[research_department_supervisor, analysis_department_supervisor],
        prompt="""You are the executive supervisor managing department supervisors.
        Delegate complex projects to appropriate departments:
        - Research department: for all information gathering needs
        - Analysis department: for all data processing and visualization needs
        Coordinate cross-departmental projects when necessary.""",
        name="executive"
    )
    
    return executive_supervisor

# Create hierarchical system
# hierarchical_supervisor = create_hierarchical_supervisor_system()
```

#### Pattern 2: Dynamic Supervisor with Adaptive Worker Pool

```python
class AdaptiveSupervisor:
    """Supervisor that can dynamically add/remove workers based on workload"""
    
    def __init__(self, base_supervisor, worker_pool):
        self.base_supervisor = base_supervisor
        self.available_workers = worker_pool
        self.active_workers = []
        self.workload_history = []
    
    def adapt_worker_pool(self, current_request_complexity):
        """Adapt active worker pool based on request complexity"""
        
        if current_request_complexity == "high":
            # Add specialized workers for complex tasks
            self.activate_specialist_workers()
        elif current_request_complexity == "low":
            # Use minimal worker set for efficiency
            self.use_core_workers_only()
    
    def activate_specialist_workers(self):
        """Add specialist workers for complex tasks"""
        # Add domain experts to active pool
        specialist_workers = [
            worker for worker in self.available_workers 
            if "specialist" in worker.name
        ]
        self.active_workers.extend(specialist_workers)
    
    def use_core_workers_only(self):
        """Keep only essential workers for simple tasks"""
        core_workers = ["researcher", "analyst"]
        self.active_workers = [
            worker for worker in self.active_workers 
            if worker.name in core_workers
        ]
    
    def process_request(self, request, config):
        """Process request with adaptive worker management"""
        
        # Analyze request complexity
        complexity = self.analyze_request_complexity(request)
        
        # Adapt worker pool
        self.adapt_worker_pool(complexity)
        
        # Rebuild supervisor with current worker pool
        adaptive_supervisor = create_supervisor(
            model=llm,
            agents=self.active_workers,
            prompt=self.generate_adaptive_prompt()
        ).compile(checkpointer=InMemorySaver())
        
        # Process request
        return adaptive_supervisor.stream({"messages": [{"role": "user", "content": request}]}, config)
    
    def analyze_request_complexity(self, request):
        """Simple complexity analysis based on request content"""
        complexity_indicators = {
            "high": ["comprehensive", "detailed", "analysis", "compare", "multiple"],
            "medium": ["create", "generate", "show", "plot"],
            "low": ["tell", "what", "who", "when", "simple"]
        }
        
        request_lower = request.lower()
        
        for complexity, indicators in complexity_indicators.items():
            if any(indicator in request_lower for indicator in indicators):
                return complexity
        
        return "medium"  # Default
    
    def generate_adaptive_prompt(self):
        """Generate supervisor prompt based on current active workers"""
        worker_descriptions = []
        for worker in self.active_workers:
            worker_descriptions.append(f"- {worker.name}: {self.get_worker_description(worker)}")
        
        return f"""You are an adaptive supervisor managing the following workers:
        {chr(10).join(worker_descriptions)}
        
        Delegate tasks efficiently based on worker capabilities.
        Coordinate multi-worker tasks when necessary."""
    
    def get_worker_description(self, worker):
        """Get description of worker capabilities"""
        descriptions = {
            "researcher": "information gathering and data collection",
            "analyst": "data analysis and visualization", 
            "specialist": "domain-specific expertise"
        }
        return descriptions.get(worker.name, "general assistance")

# Create adaptive supervisor
# adaptive_supervisor = AdaptiveSupervisor(supervisor, [research_agent, analyst_agent])
```

#### Pattern 3: Supervisor with Quality Assurance and Review

```python
class QualityAssuredSupervisor:
    """Supervisor with built-in quality control and review processes"""
    
    def __init__(self, base_supervisor, quality_standards):
        self.base_supervisor = base_supervisor
        self.quality_standards = quality_standards
        self.quality_history = []
    
    def process_with_quality_control(self, request, config):
        """Process request with quality assurance at each step"""
        
        # Phase 1: Initial processing
        initial_response = list(self.base_supervisor.stream({
            "messages": [{"role": "user", "content": request}]
        }, config))
        
        # Phase 2: Quality assessment
        quality_score = self.assess_response_quality(initial_response, request)
        
        # Phase 3: Quality-based action
        if quality_score >= self.quality_standards["minimum_acceptable"]:
            return initial_response
        else:
            return self.improve_response_quality(request, initial_response, config)
    
    def assess_response_quality(self, response_chunks, original_request):
        """Assess quality of supervisor's response"""
        
        # Extract final response
        final_response = self.extract_final_response(response_chunks)
        
        quality_metrics = {
            "completeness": self.check_completeness(final_response, original_request),
            "accuracy": self.check_accuracy(final_response),
            "clarity": self.check_clarity(final_response),
            "usefulness": self.check_usefulness(final_response, original_request)
        }
        
        # Calculate weighted quality score
        weights = {"completeness": 0.3, "accuracy": 0.3, "clarity": 0.2, "usefulness": 0.2}
        quality_score = sum(quality_metrics[metric] * weights[metric] 
                          for metric in quality_metrics)
        
        self.quality_history.append({
            "request": original_request,
            "score": quality_score,
            "metrics": quality_metrics
        })
        
        return quality_score
    
    def check_completeness(self, response, request):
        """Check if response addresses all parts of the request"""
        # Simple heuristic: longer responses for complex requests
        request_complexity = len(request.split())
        response_length = len(response) if response else 0
        
        expected_length = request_complexity * 10  # Rough heuristic
        completeness_ratio = min(response_length / expected_length, 1.0)
        
        return completeness_ratio
    
    def check_accuracy(self, response):
        """Check response accuracy (simplified)"""
        if not response:
            return 0.0
        
        # Heuristics for accuracy assessment
        accuracy_indicators = ["successfully", "data", "analysis", "chart", "information"]
        error_indicators = ["error", "failed", "sorry", "cannot"]
        
        positive_count = sum(1 for indicator in accuracy_indicators if indicator in response.lower())
        negative_count = sum(1 for indicator in error_indicators if indicator in response.lower())
        
        accuracy_score = max(0, (positive_count - negative_count) / max(positive_count + negative_count, 1))
        return min(accuracy_score, 1.0)
    
    def check_clarity(self, response):
        """Check response clarity and readability"""
        if not response:
            return 0.0
        
        # Simple readability heuristics
        sentences = response.split('.')
        avg_sentence_length = sum(len(sentence.split()) for sentence in sentences) / max(len(sentences), 1)
        
        # Prefer moderate sentence length (10-20 words)
        clarity_score = 1.0 - abs(avg_sentence_length - 15) / 15
        return max(0, min(clarity_score, 1.0))
    
    def check_usefulness(self, response, request):
        """Check if response is useful for the original request"""
        if not response:
            return 0.0
        
        # Check for relevant keywords from request
        request_keywords = set(request.lower().split())
        response_keywords = set(response.lower().split())
        
        keyword_overlap = len(request_keywords & response_keywords) / max(len(request_keywords), 1)
        return min(keyword_overlap * 2, 1.0)  # Scale up overlap score
    
    def improve_response_quality(self, request, initial_response, config):
        """Attempt to improve response quality"""
        
        # Create improvement request
        improvement_request = f"""
        The previous response to "{request}" needs improvement.
        Please provide a more comprehensive and accurate response.
        Focus on completeness, accuracy, and clarity.
        """
        
        # Process improvement request
        improved_response = list(self.base_supervisor.stream({
            "messages": [{"role": "user", "content": improvement_request}]
        }, config))
        
        return improved_response
    
    def extract_final_response(self, chunks):
        """Extract final response from supervisor chunks"""
        final_response = ""
        for chunk in chunks:
            if "messages" in chunk:
                for message in chunk["messages"]:
                    if (message.get("role") == "assistant" and 
                        not message.get("name") and
                        "content" in message):
                        final_response = message["content"]
        return final_response
    
    def get_quality_report(self):
        """Generate quality assurance report"""
        if not self.quality_history:
            return {"message": "No quality data available"}
        
        avg_quality = sum(entry["score"] for entry in self.quality_history) / len(self.quality_history)
        
        return {
            "average_quality_score": avg_quality,
            "total_assessments": len(self.quality_history),
            "quality_trend": self.calculate_quality_trend(),
            "improvement_recommendations": self.generate_improvement_recommendations()
        }
    
    def calculate_quality_trend(self):
        """Calculate quality trend over time"""
        if len(self.quality_history) < 2:
            return "insufficient_data"
        
        recent_avg = sum(entry["score"] for entry in self.quality_history[-3:]) / min(3, len(self.quality_history))
        overall_avg = sum(entry["score"] for entry in self.quality_history) / len(self.quality_history)
        
        if recent_avg > overall_avg + 0.1:
            return "improving"
        elif recent_avg < overall_avg - 0.1:
            return "declining"
        else:
            return "stable"
    
    def generate_improvement_recommendations(self):
        """Generate recommendations based on quality history"""
        recommendations = []
        
        if not self.quality_history:
            return ["Collect more quality data"]
        
        # Analyze common quality issues
        avg_metrics = {
            "completeness": sum(entry["metrics"]["completeness"] for entry in self.quality_history) / len(self.quality_history),
            "accuracy": sum(entry["metrics"]["accuracy"] for entry in self.quality_history) / len(self.quality_history),
            "clarity": sum(entry["metrics"]["clarity"] for entry in self.quality_history) / len(self.quality_history),
            "usefulness": sum(entry["metrics"]["usefulness"] for entry in self.quality_history) / len(self.quality_history)
        }
        
        for metric, score in avg_metrics.items():
            if score < 0.7:
                recommendations.append(f"Improve {metric} - current average: {score:.2f}")
        
        return recommendations if recommendations else ["Quality metrics are satisfactory"]

# Create quality-assured supervisor
# qa_supervisor = QualityAssuredSupervisor(supervisor, {"minimum_acceptable": 0.7})
```

## 🔧 Production Deployment Considerations

### Enterprise Supervisor Systems

```python
class EnterpriseSupervisorSystem:
    """Enterprise-grade supervisor system with full observability and management"""
    
    def __init__(self, config):
        self.config = config
        self.supervisor = None
        self.metrics = {
            "total_requests": 0,
            "delegation_patterns": {},
            "worker_utilization": {},
            "error_rates": {},
            "performance_metrics": {}
        }
        self.initialize_system()
    
    def initialize_system(self):
        """Initialize enterprise supervisor system"""
        
        # Create enhanced worker agents with monitoring
        workers = self.create_monitored_workers()
        
        # Create supervisor with enterprise settings
        self.supervisor = create_supervisor(
            model=self.config["llm"],
            agents=workers,
            prompt=self.config["supervisor_prompt"],
            add_handoff_back_messages=True,
            output_mode="full_history"
        ).compile(checkpointer=self.config["checkpointer"])
        
        print("✅ Enterprise supervisor system initialized")
    
    def create_monitored_workers(self):
        """Create worker agents with monitoring capabilities"""
        
        # Enhanced research agent
        research_agent = create_react_agent(
            self.config["llm"],
            [wikipedia_tool, stock_data_tool],
            prompt=self.config["research_agent_prompt"],
            name="researcher"
        )
        
        # Enhanced analyst agent
        analyst_agent = create_react_agent(
            self.config["llm"],
            [python_repl_tool],
            prompt=self.config["analyst_agent_prompt"], 
            name="analyst"
        )
        
        return [research_agent, analyst_agent]
    
    def process_request(self, request, config, user_id=None):
        """Process request with full enterprise monitoring"""
        
        start_time = time.time()
        request_id = f"req_{int(start_time)}_{hash(request)}"
        
        # Pre-processing
        self.log_request(request_id, request, user_id)
        
        try:
            # Execute request
            chunks = list(self.supervisor.stream({
                "messages": [{"role": "user", "content": request}]
            }, config))
            
            # Post-processing
            execution_time = time.time() - start_time
            self.analyze_and_log_results(request_id, chunks, execution_time)
            
            return chunks
            
        except Exception as e:
            self.log_error(request_id, e)
            raise
    
    def log_request(self, request_id, request, user_id):
        """Log incoming request with metadata"""
        self.metrics["total_requests"] += 1
        
        log_entry = {
            "request_id": request_id,
            "timestamp": time.time(),
            "user_id": user_id,
            "request": request[:200],  # Truncate for privacy
            "status": "processing"
        }
        
        # In production: send to logging system
        if self.config.get("detailed_logging"):
            print(f"[REQUEST] {log_entry}")
    
    def analyze_and_log_results(self, request_id, chunks, execution_time):
        """Analyze results and update metrics"""
        
        # Extract workflow information
        workers_used = []
        delegation_count = 0
        
        for chunk in chunks:
            if "messages" in chunk:
                for message in chunk["messages"]:
                    if message.get("name") in ["researcher", "analyst"]:
                        workers_used.append(message["name"])
                    elif (message.get("role") == "assistant" and not message.get("name")):
                        delegation_count += 1
        
        # Update metrics
        self.update_delegation_metrics(workers_used)
        self.update_performance_metrics(execution_time, delegation_count)
        self.update_worker_utilization(workers_used)
        
        # Log completion
        completion_log = {
            "request_id": request_id,
            "execution_time": execution_time,
            "workers_used": list(set(workers_used)),
            "delegations": delegation_count,
            "status": "completed"
        }
        
        if self.config.get("detailed_logging"):
            print(f"[COMPLETED] {completion_log}")
    
    def update_delegation_metrics(self, workers_used):
        """Update delegation pattern metrics"""
        pattern = " → ".join(workers_used) if workers_used else "direct"
        self.metrics["delegation_patterns"][pattern] = \
            self.metrics["delegation_patterns"].get(pattern, 0) + 1
    
    def update_performance_metrics(self, execution_time, delegation_count):
        """Update performance metrics"""
        if "response_times" not in self.metrics["performance_metrics"]:
            self.metrics["performance_metrics"]["response_times"] = []
        
        self.metrics["performance_metrics"]["response_times"].append(execution_time)
        
        # Keep only recent metrics
        if len(self.metrics["performance_metrics"]["response_times"]) > 100:
            self.metrics["performance_metrics"]["response_times"] = \
                self.metrics["performance_metrics"]["response_times"][-50:]
    
    def update_worker_utilization(self, workers_used):
        """Update worker utilization statistics"""
        for worker in set(workers_used):
            self.metrics["worker_utilization"][worker] = \
                self.metrics["worker_utilization"].get(worker, 0) + 1
    
    def log_error(self, request_id, error):
        """Log system errors"""
        error_type = type(error).__name__
        self.metrics["error_rates"][error_type] = \
            self.metrics["error_rates"].get(error_type, 0) + 1
        
        error_log = {
            "request_id": request_id,
            "error_type": error_type,
            "error_message": str(error)[:500],
            "timestamp": time.time()
        }
        
        print(f"[ERROR] {error_log}")
    
    def get_health_status(self):
        """Get comprehensive system health status"""
        
        # Calculate key metrics
        total_requests = self.metrics["total_requests"]
        total_errors = sum(self.metrics["error_rates"].values())
        error_rate = total_errors / max(total_requests, 1)
        
        response_times = self.metrics["performance_metrics"].get("response_times", [])
        avg_response_time = sum(response_times) / max(len(response_times), 1) if response_times else 0
        
        # Determine health status
        if error_rate > 0.1:
            health_status = "unhealthy"
        elif error_rate > 0.05:
            health_status = "degraded" 
        else:
            health_status = "healthy"
        
        return {
            "status": health_status,
            "total_requests": total_requests,
            "error_rate": error_rate,
            "avg_response_time": avg_response_time,
            "worker_utilization": self.metrics["worker_utilization"],
            "top_delegation_patterns": dict(sorted(
                self.metrics["delegation_patterns"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5])
        }
    
    def generate_analytics_report(self):
        """Generate comprehensive analytics report"""
        
        health_status = self.get_health_status()
        
        # Advanced analytics
        response_times = self.metrics["performance_metrics"].get("response_times", [])
        
        report = {
            "system_health": health_status,
            "performance_analysis": {
                "avg_response_time": health_status["avg_response_time"],
                "p95_response_time": sorted(response_times)[int(len(response_times) * 0.95)] if response_times else 0,
                "fastest_response": min(response_times) if response_times else 0,
                "slowest_response": max(response_times) if response_times else 0
            },
            "delegation_insights": self.analyze_delegation_patterns(),
            "optimization_recommendations": self.generate_optimization_recommendations()
        }
        
        return report
    
    def analyze_delegation_patterns(self):
        """Analyze delegation patterns for insights"""
        patterns = self.metrics["delegation_patterns"]
        
        if not patterns:
            return {"message": "No delegation data available"}
        
        total_delegations = sum(patterns.values())
        
        insights = {
            "most_common_pattern": max(patterns.items(), key=lambda x: x[1]),
            "pattern_diversity": len(patterns),
            "single_agent_ratio": patterns.get("researcher", 0) + patterns.get("analyst", 0) / total_delegations,
            "multi_agent_ratio": (total_delegations - patterns.get("researcher", 0) - patterns.get("analyst", 0)) / total_delegations
        }
        
        return insights
    
    def generate_optimization_recommendations(self):
        """Generate system optimization recommendations"""
        recommendations = []
        
        # Performance recommendations
        response_times = self.metrics["performance_metrics"].get("response_times", [])
        if response_times:
            avg_time = sum(response_times) / len(response_times)
            if avg_time > 15:
                recommendations.append("Consider optimizing slow worker agents or caching frequent requests")
        
        # Utilization recommendations
        utilization = self.metrics["worker_utilization"]
        if utilization:
            total_usage = sum(utilization.values())
            for worker, usage in utilization.items():
                ratio = usage / total_usage
                if ratio > 0.8:
                    recommendations.append(f"Consider scaling {worker} agent due to high utilization ({ratio:.1%})")
                elif ratio < 0.1:
                    recommendations.append(f"Consider consolidating {worker} agent due to low utilization ({ratio:.1%})")
        
        # Error rate recommendations
        error_rate = sum(self.metrics["error_rates"].values()) / max(self.metrics["total_requests"], 1)
        if error_rate > 0.05:
            recommendations.append(f"High error rate ({error_rate:.1%}) - review error handling and worker reliability")
        
        return recommendations if recommendations else ["System performance is optimal"]

# Example enterprise configuration
enterprise_config = {
    "llm": llm,
    "checkpointer": InMemorySaver(),
    "supervisor_prompt": """You are an enterprise supervisor managing research and analysis teams.
    Ensure high-quality, professional responses to all requests.
    Coordinate teams efficiently and escalate issues when necessary.""",
    "research_agent_prompt": """You are a professional research analyst.
    Provide accurate, comprehensive information from reliable sources.
    Report findings clearly and concisely to the supervisor.""",
    "analyst_agent_prompt": """You are a professional data analyst. 
    Create high-quality visualizations and analysis.
    Report results professionally to the supervisor.""",
    "detailed_logging": True
}

# Create enterprise supervisor system
# enterprise_supervisor = EnterpriseSupervisorSystem(enterprise_config)
```

## 📊 Assessment Rubric

### Functionality (40 points)
- **Worker Agent Creation**: Correct specialization and tool assignment (15 pts)
- **Supervisor Creation**: Proper use of `create_supervisor()` with appropriate configuration (15 pts)
- **Task Delegation**: Supervisor correctly assigns tasks to appropriate workers (10 pts)

### Understanding (35 points)
- **Hierarchical Concepts**: Understands supervisor-worker relationship and benefits (15 pts)
- **Coordination Patterns**: Explains how supervisor coordinates multi-step workflows (10 pts)
- **Architecture Comparison**: Can compare supervisor vs swarm vs single agent approaches (10 pts)

### Advanced Implementation (25 points)
- **Testing**: Comprehensive testing of delegation patterns and coordination (15 pts)
- **Error Handling**: Understanding of error scenarios in multi-agent hierarchies (5 pts)
- **Extensions**: Implementation of advanced patterns or monitoring (5 pts)

### Total: 100 points

**Grading Scale:**
- 90-100: Excellent mastery of hierarchical multi-agent systems and coordination patterns
- 80-89: Good implementation with minor issues in delegation logic or workflow understanding
- 70-79: Basic functionality working, needs deeper understanding of supervisor patterns
- Below 70: Requires additional practice with hierarchical agent architecture concepts

## 🚀 Real-World Applications

### Enterprise Hierarchical AI Systems

#### Corporate Research Division
```python
corporate_research = create_supervisor(
    model=llm,
    agents=[
        market_research_agent,      # Market analysis and trends
        competitive_intelligence_agent, # Competitor analysis
        financial_research_agent    # Financial data and modeling
    ],
    prompt="You are a research division supervisor coordinating comprehensive business intelligence."
)
```

#### Customer Service Organization
```python
customer_service_org = create_supervisor(
    model=llm,
    agents=[
        technical_support_agent,    # Technical issue resolution
        billing_support_agent,      # Account and payment issues
        product_specialist_agent    # Product information and guidance
    ],
    prompt="You are a customer service supervisor ensuring excellent customer experience through specialized support teams."
)
```

#### Content Production Studio
```python
content_studio = create_supervisor(
    model=llm,
    agents=[
        content_researcher_agent,   # Topic research and fact-checking
        content_writer_agent,      # Content creation and editing
        content_optimizer_agent    # SEO and performance optimization
    ],
    prompt="You are a content studio supervisor orchestrating high-quality content production workflows."
)
```

## 💡 Pro Tips for Instructors

1. **Start with Clear Hierarchy**: Emphasize the boss-worker relationship analogy
2. **Demonstrate Delegation**: Show how supervisor decisions affect workflow
3. **Compare Architectures**: Contrast with swarm to highlight benefits
4. **Real Examples**: Use business organization analogies students understand
5. **Error Scenarios**: Show what happens when delegation goes wrong

## 🏁 Conclusion

This exercise introduces students to hierarchical multi-agent systems through the supervisor architecture. Key learning outcomes include:

- **Hierarchical Coordination**: Understanding centralized management of distributed AI workers
- **Task Delegation**: Learning how supervisors analyze requests and assign appropriate workers
- **Quality Control**: Implementing oversight and result synthesis through supervisor agents
- **Scalable Organization**: Building systems that can grow by adding new worker agents
- **Enterprise Patterns**: Creating production-ready hierarchical AI systems

**Key Architectural Insights:**
- Supervisor agents provide centralized coordination and quality control
- Worker agents specialize in specific domains for optimal performance
- Hierarchical communication eliminates peer-to-peer coordination complexity
- Clear role boundaries improve system predictability and debugging
- Supervisor architecture scales better than peer-to-peer for large systems

Students are now equipped to design and implement sophisticated hierarchical AI systems suitable for enterprise deployment! 🚀👔🧠