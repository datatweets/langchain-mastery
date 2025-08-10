# Tutorial 10: Creating Custom Tools for AI Agents

## 🎯 What You'll Learn

In this tutorial, you'll learn how to:

- Create your own custom tools that AI agents can use
- Build specialized functions for specific tasks (like math calculations)
- Understand how agents decide when and how to use tools
- Create a math assistant that can solve geometry problems
- Handle tool inputs and outputs properly

Think of this as teaching your AI assistant a new skill - like giving it a calculator that it knows exactly when and how to use!

## 🤔 Built-in Tools vs. Custom Tools

### Using Built-in Tools (Previous tutorials):
```
You: "What is the population of Tokyo?"
AI Agent: *uses Wikipedia tool* → "According to Wikipedia, Tokyo has 14 million people..."
```

### Creating Custom Tools (This tutorial):
```
You: "What's the hypotenuse of a triangle with sides 10 and 12?"
AI Agent: *uses your custom math tool* → "Using the hypotenuse calculator, the answer is 15.62"
```

**The Power**: You can teach AI agents exactly the skills you need for your specific use case!

## 🧠 Why Create Custom Tools?

### Real-World Applications:
- **Business**: Calculate pricing, inventory, or financial metrics
- **Engineering**: Solve specific equations or measurements
- **Education**: Create tutoring tools for different subjects
- **Healthcare**: Calculate dosages or medical metrics
- **Construction**: Measure distances, areas, or materials needed

### Advantages of Custom Tools:
- **Accuracy**: No guessing - precise calculations every time
- **Specialization**: Designed for your exact needs
- **Control**: You decide exactly how the tool works
- **Integration**: Fits perfectly into your workflow

## 🔍 Understanding the Code: Line by Line

Let's examine `script_10.py` step by step:

### Step 1: Imports and Setup

```python
# tool_math_agent.py
import os
import math
from dotenv import load_dotenv

# Disable LangSmith tracing (optional) BEFORE importing langchain bits
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "false"
for var in ["LANGSMITH_API_KEY", "LANGCHAIN_API_KEY", "LANGCHAIN_ENDPOINT", "LANGSMITH_ENDPOINT", "LANGCHAIN_PROJECT"]:
    os.environ.pop(var, None)
```

**What's happening here?**

1. **`import math`** - NEW!
   - Python's built-in math library
   - Provides functions like `sqrt()` for square root calculations
   - Essential for our geometry calculations

2. **Environment cleanup** - Same as previous tutorials
   - Disables logging to keep output clean
   - Focuses on the important information

### Step 2: LangChain Tool Imports

```python
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
```

**What's happening here?**

1. **`from langchain.tools import tool`** - NEW!
   - This is the magic decorator that turns regular Python functions into AI tools
   - `@tool` tells LangChain "this function can be used by AI agents"

2. **Agent imports** - Same as Tutorial 7
   - `create_react_agent`: Creates thinking AI agents
   - `AgentExecutor`: Manages the agent safely

### Step 3: Creating Our Custom Tool

```python
# -----------------------------
# 1) Define the math tool
# -----------------------------
@tool
def hypotenuse_length(input: str) -> float:
    """Calculates the hypotenuse of a right-angled triangle.
    Input format: 'a, b' (two side lengths separated by a comma)."""
    # Clean the input - remove quotes and split
    clean_input = input.strip().strip("'\"")
    sides = clean_input.split(',')
    if len(sides) != 2:
        raise ValueError("Please provide exactly two side lengths, e.g. '10, 12'.")

    # Convert to floats after stripping whitespace
    a = float(sides[0].strip())
    b = float(sides[1].strip())

    # a^2 + b^2, then square root
    return math.sqrt(a**2 + b**2)
```

**What's happening here?** (This is the heart of custom tools!)

1. **`@tool` Decorator**:
   - This special symbol tells LangChain: "This function is now an AI tool"
   - The AI can see this function and decide when to use it
   - It's like giving the AI a new app on its phone

2. **Function Definition**:
   ```python
   def hypotenuse_length(input: str) -> float:
   ```
   - **Function name**: `hypotenuse_length` (the AI will see this name)
   - **Input type**: `str` (AI agents always send text to tools)
   - **Output type**: `float` (we're returning a decimal number)

3. **Docstring (VERY IMPORTANT)**:
   ```python
   """Calculates the hypotenuse of a right-angled triangle.
   Input format: 'a, b' (two side lengths separated by a comma)."""
   ```
   - **The AI reads this!** This is how the AI knows what the tool does
   - **Must be clear**: The AI uses this to decide when to use the tool
   - **Include format**: Tell the AI exactly how to format the input

4. **Input Cleaning**:
   ```python
   clean_input = input.strip().strip("'\"")
   ```
   - **Why needed**: AI sometimes adds extra quotes around input
   - **`strip()`**: Removes spaces from beginning and end
   - **`strip("'\"")`**: Removes quote marks that AI might add

5. **Input Parsing**:
   ```python
   sides = clean_input.split(',')
   if len(sides) != 2:
       raise ValueError("Please provide exactly two side lengths, e.g. '10, 12'.")
   ```
   - **`split(',')`**: Breaks "10, 12" into ["10", "12"]
   - **Error checking**: Makes sure we got exactly 2 numbers
   - **Clear error message**: Helps both AI and humans understand what went wrong

6. **Mathematical Calculation**:
   ```python
   a = float(sides[0].strip())
   b = float(sides[1].strip())
   return math.sqrt(a**2 + b**2)
   ```
   - **Convert to numbers**: `float()` turns text "10" into number 10.0
   - **Pythagorean theorem**: c² = a² + b², so c = √(a² + b²)
   - **`math.sqrt()`**: Calculates square root
   - **`**2`**: Python's way to say "to the power of 2"

### Step 4: Setting Up the AI Model

```python
# -----------------------------
# 2) Model and tools
# -----------------------------
llm = ChatOpenAI(
    model="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0  # deterministic for calculations
)

tools = [hypotenuse_length]
```

**What's happening here?**

1. **`temperature=0`** - VERY IMPORTANT for tools!
   - **Why**: We want consistent, accurate tool usage
   - **0 = deterministic**: Same input → same output
   - **Higher values**: More creative but less reliable for calculations

2. **`tools = [hypotenuse_length]`**:
   - Creates a list containing our custom tool
   - The AI agent will have access to this tool
   - You can add multiple tools to this list

### Step 5: Creating the ReAct Prompt

```python
# -----------------------------
# 3) ReAct prompt with required variables
# -----------------------------
template = """
You are a helpful construction assistant. If a calculation is needed, use the available tools.

You have access to the following tools:
{tools}

Use this exact format:
Thought: think about what to do next
Action: the single tool to use, exactly one of [{tool_names}]
Action Input: the input for the action
Observation: the result of the action
...(you can repeat Thought/Action/Observation)...
Thought: I can now answer
Final Answer: the final answer to the user's question

Question: {input}
Thought: {agent_scratchpad}
""".strip()

react_prompt = PromptTemplate.from_template(template)
```

**What's happening here?** (This teaches the AI how to think!)

1. **Role Definition**:
   ```python
   "You are a helpful construction assistant..."
   ```
   - Sets the AI's personality and purpose
   - "Construction assistant" suggests it should help with measurements
   - Encourages tool usage: "if a calculation is needed, use the available tools"

2. **Required Variables** (CRITICAL for agents):
   - **`{tools}`**: LangChain fills this with tool descriptions
   - **`{tool_names}`**: LangChain fills this with tool names
   - **`{input}`**: The user's question
   - **`{agent_scratchpad}`**: The AI's thinking workspace

3. **ReAct Format Instructions**:
   ```
   Thought: think about what to do next
   Action: the single tool to use
   Action Input: the input for the action
   Observation: the result of the action
   ```
   - **Thought**: AI decides what to do
   - **Action**: AI chooses which tool to use
   - **Action Input**: AI provides input to the tool
   - **Observation**: AI sees the tool's result
   - **Repeat**: Until enough information is gathered
   - **Final Answer**: AI gives the final response

### Step 6: Creating the Agent

```python
# -----------------------------
# 4) Create agent + executor
# -----------------------------
agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)
app = AgentExecutor(agent=agent, tools=tools, verbose=True)
```

**What's happening here?**

1. **`create_react_agent(...)`**:
   - Combines the AI model, tools, and prompt
   - Creates an intelligent agent that can think and act
   - The agent knows how to use your custom tool

2. **`AgentExecutor(...)`**:
   - **Safety wrapper**: Prevents infinite loops and errors
   - **`verbose=True`**: Shows the AI's thinking process
   - **Tool management**: Safely executes tools and handles results

### Step 7: Using the Agent

```python
# -----------------------------
# 5) Ask a natural-language query
# -----------------------------
query = "What is the hypotenuse length of a triangle with side lengths of 10 and 12?"

# NOTE: AgentExecutor expects {"input": "..."} by default.
# This will trigger tool use with our hypotenuse_length tool.
result = app.invoke({"input": query})

# Print the final answer
print("\n--- Answer ---")
print(result["output"])
```

**What's happening here?**

1. **Natural Language Query**:
   - **Human-friendly**: You ask in plain English
   - **No technical formats**: Just describe what you want
   - **AI interprets**: The agent figures out it needs to calculate a hypotenuse

2. **Agent Processing**:
   - AI reads your question
   - AI thinks: "This is about triangle hypotenuse calculation"
   - AI decides: "I should use the hypotenuse_length tool"
   - AI formats input: "10, 12"
   - Tool calculates: 15.62...
   - AI responds: "The answer is 15.62"

## 🧠 What Happens Behind the Scenes?

### The Agent's Complete Thought Process:

1. **Receives Question**:
   ```
   Human: "What is the hypotenuse length of a triangle with side lengths of 10 and 12?"
   ```

2. **AI Thinks**:
   ```
   Thought: I need to calculate the hypotenuse of a right-angled triangle with side lengths of 10 and 12.
   ```

3. **AI Acts**:
   ```
   Action: hypotenuse_length
   Action Input: '10, 12'
   ```

4. **Tool Executes**:
   ```python
   # Your custom function runs:
   a = 10.0
   b = 12.0
   result = math.sqrt(10**2 + 12**2)  # = sqrt(244) = 15.62...
   ```

5. **AI Observes**:
   ```
   Observation: 15.620499351813308
   ```

6. **AI Concludes**:
   ```
   Thought: I can now answer
   Final Answer: The hypotenuse length of a triangle with side lengths of 10 and 12 is approximately 15.62.
   ```

## 🚀 How to Run This Code

### Prerequisites
1. **API key**: Set up your OpenAI API key in `.env`
2. **Dependencies**: Run `pip install -r requirements.txt`

### Steps
1. **Run the script**:
   ```bash
   python script_10.py
   ```

2. **What you'll see**:
   ```
   > Entering new AgentExecutor chain...
   I need to calculate the hypotenuse of a right-angled triangle with side lengths of 10 and 12.
   Action: hypotenuse_length
   Action Input: '10, 12'
   15.620499351813308
   I can now answer
   Final Answer: The hypotenuse length of a triangle with side lengths of 10 and 12 is approximately 15.62.
   > Finished chain.
   
   --- Answer ---
   The hypotenuse length of a triangle with side lengths of 10 and 12 is approximately 15.62.
   ```

## 🎓 Key Concepts You've Learned

### Custom Tools
- **What**: Python functions that AI agents can use
- **Why**: Extend AI capabilities with specific skills
- **How**: Use the `@tool` decorator on regular functions

### Tool Design Principles
- **Clear docstrings**: AI reads these to understand the tool
- **Input handling**: Always expect string input from AI
- **Error handling**: Provide clear error messages
- **Output format**: Return the appropriate data type

### Agent-Tool Integration
- **What**: AI agents that can decide when to use tools
- **Why**: Combines AI reasoning with precise calculations
- **How**: ReAct framework guides the decision-making process

### Pythagorean Theorem Implementation
- **What**: Mathematical formula for right triangle hypotenuse
- **Why**: Common calculation needed in construction and engineering
- **How**: c = √(a² + b²) using Python's math.sqrt()

## 🔧 Common Issues and Solutions

**Problem: "Tool not found" or "Invalid tool name"**
```python
# Solution: Make sure tool is in the tools list
tools = [hypotenuse_length]  # Must include your custom tool

# And check the function name matches
@tool
def hypotenuse_length(input: str) -> float:  # This name is what AI sees
```

**Problem: "ValueError: could not convert string to float"**
```python
# Solution: Improve input cleaning
def hypotenuse_length(input: str) -> float:
    # Clean more thoroughly
    clean_input = input.strip().strip("'\"").strip()
    # Add more error checking
    if not clean_input:
        raise ValueError("Input cannot be empty")
```

**Problem: "Agent keeps trying wrong format"**
```python
# Solution: Make docstring more specific
@tool
def hypotenuse_length(input: str) -> float:
    """Calculates the hypotenuse of a right-angled triangle.
    
    IMPORTANT: Input must be exactly 'a, b' where a and b are numbers.
    Example: '3, 4' returns 5.0
    Example: '10, 12' returns 15.62
    """
```

**Problem: "Missing required variables in prompt"**
```python
# Solution: Include all required variables
template = """
{tools}        # Required: Tool descriptions
{tool_names}   # Required: Tool names
{input}        # Required: User input
{agent_scratchpad}  # Required: Agent's workspace
"""
```

## 🎯 Try These Experiments

### 1. Create Additional Math Tools
```python
@tool
def circle_area(radius: str) -> float:
    """Calculates the area of a circle.
    Input: radius as a string number, e.g., '5'"""
    r = float(radius.strip().strip("'\""))
    return math.pi * r**2

@tool
def rectangle_area(input: str) -> float:
    """Calculates the area of a rectangle.
    Input format: 'length, width' separated by comma"""
    clean_input = input.strip().strip("'\"")
    dims = clean_input.split(',')
    length = float(dims[0].strip())
    width = float(dims[1].strip())
    return length * width

# Add to tools list
tools = [hypotenuse_length, circle_area, rectangle_area]
```

### 2. Create Business Tools
```python
@tool
def calculate_tax(input: str) -> float:
    """Calculates tax amount.
    Input format: 'amount, tax_rate' (tax_rate as percentage, e.g., '100, 8.5')"""
    clean_input = input.strip().strip("'\"")
    parts = clean_input.split(',')
    amount = float(parts[0].strip())
    tax_rate = float(parts[1].strip())
    return amount * (tax_rate / 100)

@tool
def compound_interest(input: str) -> float:
    """Calculates compound interest.
    Input format: 'principal, rate, time' (rate as percentage, time in years)"""
    clean_input = input.strip().strip("'\"")
    parts = clean_input.split(',')
    principal = float(parts[0].strip())
    rate = float(parts[1].strip()) / 100
    time = float(parts[2].strip())
    return principal * ((1 + rate) ** time)
```

### 3. Create Text Processing Tools
```python
@tool
def word_count(text: str) -> int:
    """Counts words in text.
    Input: any text string"""
    clean_text = text.strip().strip("'\"")
    return len(clean_text.split())

@tool
def reverse_text(text: str) -> str:
    """Reverses text.
    Input: any text string"""
    clean_text = text.strip().strip("'\"")
    return clean_text[::-1]
```

### 4. Create Data Tools
```python
import statistics

@tool
def calculate_average(numbers: str) -> float:
    """Calculates average of numbers.
    Input format: 'num1, num2, num3, ...' separated by commas"""
    clean_input = numbers.strip().strip("'\"")
    num_list = [float(x.strip()) for x in clean_input.split(',')]
    return statistics.mean(num_list)
```

## 🌟 Advanced Custom Tool Techniques

### 1. Tools with Multiple Parameters
```python
from langchain.tools import tool
from typing import Dict, Any

@tool
def advanced_calculator(operation: str) -> float:
    """Performs advanced calculations.
    
    Supported operations:
    - 'sqrt:25' for square root
    - 'power:2,3' for 2 to the power of 3
    - 'log:100' for natural logarithm
    """
    clean_op = operation.strip().strip("'\"")
    
    if clean_op.startswith('sqrt:'):
        number = float(clean_op.split(':')[1])
        return math.sqrt(number)
    elif clean_op.startswith('power:'):
        parts = clean_op.split(':')[1].split(',')
        base = float(parts[0])
        exponent = float(parts[1])
        return base ** exponent
    elif clean_op.startswith('log:'):
        number = float(clean_op.split(':')[1])
        return math.log(number)
    else:
        raise ValueError("Unsupported operation")
```

### 2. Tools with External Data
```python
import requests

@tool
def weather_info(city: str) -> str:
    """Gets weather information for a city.
    Input: city name as string"""
    # Note: This is a simplified example
    # In real applications, you'd need an API key
    clean_city = city.strip().strip("'\"")
    # This is just a mock response
    return f"Weather in {clean_city}: 72°F, Sunny"
```

### 3. File Processing Tools
```python
@tool
def count_lines_in_file(filename: str) -> int:
    """Counts lines in a text file.
    Input: filename as string"""
    clean_filename = filename.strip().strip("'\"")
    try:
        with open(clean_filename, 'r') as file:
            return sum(1 for line in file)
    except FileNotFoundError:
        return -1  # Error indicator
```

## 📚 Tool Design Best Practices

### 1. Clear Documentation
```python
@tool
def my_tool(input: str) -> float:
    """One-line summary of what the tool does.
    
    Detailed explanation of the tool's purpose.
    
    Input format: Exact format expected (very important!)
    Example: 'value1, value2' for two numbers
    Example: 'single_value' for one parameter
    
    Returns: What type of data is returned
    """
```

### 2. Robust Input Handling
```python
@tool
def robust_tool(input: str) -> str:
    """Tool with robust input handling."""
    # Clean input
    clean = input.strip().strip("'\"")
    
    # Validate input
    if not clean:
        raise ValueError("Input cannot be empty")
    
    # Handle different formats
    if ',' in clean:
        parts = clean.split(',')
    else:
        parts = [clean]
    
    # Process and return
    return "processed_result"
```

### 3. Error Handling
```python
@tool
def safe_tool(input: str) -> str:
    """Tool with comprehensive error handling."""
    try:
        clean_input = input.strip().strip("'\"")
        
        if not clean_input:
            return "Error: Empty input provided"
        
        # Your processing logic here
        result = process_input(clean_input)
        return str(result)
        
    except ValueError as e:
        return f"Error: Invalid input format - {str(e)}"
    except Exception as e:
        return f"Error: Unexpected error - {str(e)}"
```

## 🌟 What's Next?

Now that you understand custom tools, you can:

- **Create specialized tools** for your industry or use case
- **Combine multiple tools** in one agent for complex workflows
- **Build tool libraries** that can be reused across different agents
- **Integrate external APIs** and services as tools
- **Create conversational interfaces** for complex calculations

Congratulations! You can now extend AI agents with any functionality you need! 🎉

## 💡 Real-World Applications

### Construction & Engineering:
- Material calculation tools
- Load bearing calculators
- Cost estimation tools
- Measurement converters

### Finance & Business:
- Loan calculators
- ROI calculators
- Tax computation tools
- Budgeting assistants

### Education:
- Math problem solvers
- Physics calculators
- Chemistry tools
- Statistics helpers

### Healthcare:
- Dosage calculators
- BMI tools
- Medical converters
- Health trackers

### Data Analysis:
- Statistics calculators
- Data processors
- Chart generators
- Report builders

The possibilities are endless - any calculation or process you can code, you can turn into an AI tool! 🚀