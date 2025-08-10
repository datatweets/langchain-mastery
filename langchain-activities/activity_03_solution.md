# 🐍 Activity 03: Python Code Execution Tool - Master Solution Guide

## 📋 Activity Overview

**Topic:** Building a Python code execution tool with LangChain's PythonREPL  
**Duration:** 30-45 minutes  
**Difficulty:** Intermediate  
**Prerequisites:** Basic Python, understanding of code execution environments

## 🏆 Complete Solution

### Step 1: Environment Setup

```python
# Install required libraries
!pip install --quiet langchain-core==0.3.59 langchain-experimental==0.3.4
```

### Step 2: Import Dependencies

```python
from langchain_core.tools import tool
from typing import Annotated
from langchain_experimental.utilities import PythonREPL
```

**Explanation:**
- `tool`: Decorator to convert Python functions into LangChain tools
- `Annotated`: Provides metadata for type hints, essential for AI agent parameter understanding
- `PythonREPL`: Execution environment for running Python code dynamically

### Step 3: Complete Python REPL Tool Implementation

```python
# Initialize the Python REPL environment
repl = PythonREPL()

@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user. The chart should be displayed using `plt.show()`."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    
    return f"""Successfully executed the Python REPL tool.

Python code executed:
```python
{code}
```

Code output:
```
{result}
```"""
```

### Step 4: Testing Implementation

```python
# Test with NumPy array operations
code = f"""
import numpy as np

arr = np.arange(0, 9)
print(arr)
print(2 * arr)
"""

result = python_repl_tool.invoke({"code": code})
print(result)
```

## 🧠 Code Breakdown & Best Practices

### 1. PythonREPL Environment Management

```python
repl = PythonREPL()
```

**Key Features:**
- **Persistent State**: Variables and imports persist across executions within the same session
- **Isolation**: Creates a separate execution context from the main application
- **Output Capture**: Captures both stdout and stderr from executed code
- **Memory Management**: Maintains execution history and variable state

### 2. Tool Decorator and Parameters

```python
@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
```

**Design Decisions:**
- **Single Parameter**: Takes only code string for maximum flexibility
- **Clear Description**: Helps AI agents understand the tool's purpose
- **Visualization Focus**: Specifically mentions chart generation capability
- **User Visibility**: Explains how to make output visible with print statements

### 3. Execution and Error Handling

```python
try:
    result = repl.run(code)
except BaseException as e:
    return f"Failed to execute. Error: {repr(e)}"
```

**Robust Error Management:**
- **Broad Exception Catching**: `BaseException` catches all possible execution errors
- **Error Representation**: `repr(e)` provides detailed error information
- **Graceful Degradation**: Returns error information instead of crashing
- **Debugging Support**: Preserves full error context for troubleshooting

### 4. Professional Output Formatting

```python
return f"""Successfully executed the Python REPL tool.

Python code executed:
```python
{code}
```

Code output:
```
{result}
```"""
```

**Formatting Features:**
- **Clear Status**: Confirms successful execution
- **Code Echo**: Shows the executed code for transparency
- **Syntax Highlighting**: Uses markdown code blocks for readability
- **Structured Output**: Separates code from results clearly

## 🧪 Comprehensive Testing Suite

### Test Suite 1: Basic Functionality

```python
def test_basic_operations():
    """Test fundamental Python operations"""
    basic_tests = [
        # Arithmetic operations
        """
print("Basic arithmetic:")
print(f"2 + 3 = {2 + 3}")
print(f"10 * 4 = {10 * 4}")
print(f"15 / 3 = {15 / 3}")
        """,
        
        # String operations
        """
text = "LangChain Tools"
print(f"Original: {text}")
print(f"Uppercase: {text.upper()}")
print(f"Length: {len(text)}")
        """,
        
        # List operations
        """
numbers = [1, 2, 3, 4, 5]
print(f"Numbers: {numbers}")
print(f"Sum: {sum(numbers)}")
print(f"Max: {max(numbers)}")
        """
    ]
    
    for i, test_code in enumerate(basic_tests, 1):
        print(f"\n--- Basic Test {i} ---")
        result = python_repl_tool.invoke({"code": test_code})
        assert "Successfully executed" in result
        print("✅ Test passed")
        print(result)
```

### Test Suite 2: Data Science Libraries

```python
def test_data_science_libraries():
    """Test integration with popular data science libraries"""
    
    # NumPy operations
    numpy_test = """
import numpy as np

# Create arrays
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([2, 3, 4, 5, 6])

print("NumPy Operations:")
print(f"Array 1: {arr1}")
print(f"Array 2: {arr2}")
print(f"Sum: {arr1 + arr2}")
print(f"Mean of arr1: {arr1.mean()}")
print(f"Standard deviation: {arr1.std():.2f}")

# Matrix operations
matrix = np.array([[1, 2], [3, 4]])
print(f"\\nMatrix:\\n{matrix}")
print(f"Determinant: {np.linalg.det(matrix)}")
    """
    
    # Pandas operations
    pandas_test = """
import pandas as pd

# Create DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Age': [25, 30, 35, 28],
    'Salary': [50000, 60000, 70000, 55000],
    'Department': ['IT', 'HR', 'Finance', 'IT']
}

df = pd.DataFrame(data)
print("DataFrame:")
print(df)
print(f"\\nAverage Age: {df['Age'].mean()}")
print(f"Average Salary: ${df['Salary'].mean():,.2f}")
print(f"\\nIT Department:")
print(df[df['Department'] == 'IT'])
    """
    
    test_cases = [
        ("NumPy", numpy_test),
        ("Pandas", pandas_test)
    ]
    
    for library, test_code in test_cases:
        print(f"\n--- {library} Test ---")
        result = python_repl_tool.invoke({"code": test_code})
        assert "Successfully executed" in result
        print(f"✅ {library} test passed")
        # Print first 300 chars to avoid too much output
        print(result[:300] + "..." if len(result) > 300 else result)
```

### Test Suite 3: Visualization Testing

```python
def test_visualization_capabilities():
    """Test matplotlib chart generation"""
    
    # Line plot
    line_plot_test = """
import matplotlib.pyplot as plt
import numpy as np

# Create sample data
x = np.linspace(0, 10, 50)
y1 = np.sin(x)
y2 = np.cos(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y1, 'b-', label='sin(x)', linewidth=2)
plt.plot(x, y2, 'r--', label='cos(x)', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Trigonometric Functions')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("Generated line plot with sin and cos functions")
    """
    
    # Stock price simulation
    stock_chart_test = """
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Simulate stock price data
np.random.seed(42)
days = pd.date_range('2024-01-01', periods=30)
base_price = 100
daily_returns = np.random.normal(0.001, 0.02, 30)
prices = [base_price]

for return_rate in daily_returns[1:]:
    new_price = prices[-1] * (1 + return_rate)
    prices.append(new_price)

# Create comprehensive stock chart
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Price chart
ax1.plot(days, prices, 'b-', linewidth=2)
ax1.fill_between(days, prices, alpha=0.3)
ax1.set_title('Stock Price Movement', fontsize=14)
ax1.set_ylabel('Price ($)')
ax1.grid(True, alpha=0.3)

# Daily returns
daily_returns_pct = np.array(daily_returns) * 100
colors = ['green' if x > 0 else 'red' for x in daily_returns_pct]
ax2.bar(days, daily_returns_pct, color=colors, alpha=0.7)
ax2.set_title('Daily Returns', fontsize=14)
ax2.set_ylabel('Return (%)')
ax2.set_xlabel('Date')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Stock analysis complete:")
print(f"Starting price: ${prices[0]:.2f}")
print(f"Ending price: ${prices[-1]:.2f}")
print(f"Total return: {((prices[-1] / prices[0]) - 1) * 100:.1f}%")
    """
    
    # Multiple chart types
    multi_chart_test = """
import matplotlib.pyplot as plt
import numpy as np

# Create sample financial data
companies = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
market_caps = [2800, 2300, 1700, 1500, 800]  # billions
pe_ratios = [28, 25, 22, 45, 55]
revenues = [365, 198, 257, 469, 96]  # billions

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Market cap bar chart
bars1 = ax1.bar(companies, market_caps, color='skyblue', alpha=0.8)
ax1.set_title('Market Capitalization (Billions)', fontsize=12)
ax1.set_ylabel('Market Cap ($B)')
for bar, value in zip(bars1, market_caps):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
             f'${value}B', ha='center', fontweight='bold')

# P/E ratio scatter plot
ax2.scatter(companies, pe_ratios, s=200, alpha=0.7, c='coral')
ax2.set_title('Price-to-Earnings Ratios', fontsize=12)
ax2.set_ylabel('P/E Ratio')
ax2.tick_params(axis='x', rotation=45)

# Revenue comparison
ax3.barh(companies, revenues, color='lightgreen', alpha=0.8)
ax3.set_title('Annual Revenue (Billions)', fontsize=12)
ax3.set_xlabel('Revenue ($B)')

# Market cap vs Revenue scatter
ax4.scatter(revenues, market_caps, s=300, alpha=0.7, c='purple')
ax4.set_xlabel('Revenue ($B)')
ax4.set_ylabel('Market Cap ($B)')
ax4.set_title('Market Cap vs Revenue', fontsize=12)
for i, company in enumerate(companies):
    ax4.annotate(company, (revenues[i], market_caps[i]), 
                xytext=(5, 5), textcoords='offset points', fontweight='bold')

plt.tight_layout()
plt.show()

print("Generated comprehensive financial dashboard with 4 chart types")
    """
    
    visualization_tests = [
        ("Line Plot", line_plot_test),
        ("Stock Chart", stock_chart_test),
        ("Multi Chart", multi_chart_test)
    ]
    
    for chart_type, test_code in visualization_tests:
        print(f"\n--- {chart_type} Test ---")
        result = python_repl_tool.invoke({"code": test_code})
        assert "Successfully executed" in result
        print(f"✅ {chart_type} test passed")
```

### Test Suite 4: Error Handling Validation

```python
def test_error_handling():
    """Test various error scenarios"""
    
    error_scenarios = [
        # Syntax error
        ("Syntax Error", """
print("Hello World"
# Missing closing parenthesis
        """),
        
        # Runtime error
        ("Division by Zero", """
result = 10 / 0
print(result)
        """),
        
        # Import error
        ("Import Error", """
import nonexistent_module
print("This won't work")
        """),
        
        # Variable error
        ("Name Error", """
print(undefined_variable)
        """),
        
        # Type error
        ("Type Error", """
result = "string" + 5
print(result)
        """),
    ]
    
    for error_type, error_code in error_scenarios:
        print(f"\n--- Testing: {error_type} ---")
        result = python_repl_tool.invoke({"code": error_code})
        
        # Should return error message, not crash
        assert "Failed to execute" in result
        assert "Error:" in result
        print(f"✅ {error_type} handled correctly")
        print(f"Error message: {result[:100]}...")
```

### Test Suite 5: State Persistence Testing

```python
def test_state_persistence():
    """Test that variables persist across multiple executions"""
    
    # First execution: Define variables
    setup_code = """
# Define some variables and functions
global_var = "I persist across executions"
calculation_result = 42

def my_function(x):
    return x * 2 + 10

print("Variables defined successfully")
print(f"Global variable: {global_var}")
print(f"Calculation result: {calculation_result}")
    """
    
    # Second execution: Use previously defined variables
    use_variables_code = """
# Use variables from previous execution
print(f"Using global_var: {global_var}")
print(f"Using calculation_result: {calculation_result}")
print(f"Function result: {my_function(5)}")

# Define new variable
new_var = "This is new"
print(f"New variable: {new_var}")
    """
    
    # Third execution: Use all variables
    final_test_code = """
# All variables should be available
print("=== Final State Test ===")
print(f"Global var: {global_var}")
print(f"Calculation: {calculation_result}")
print(f"New var: {new_var}")
print(f"Function call: {my_function(10)}")
    """
    
    test_sequence = [
        ("Setup", setup_code),
        ("Use Variables", use_variables_code),
        ("Final Test", final_test_code)
    ]
    
    for step_name, test_code in test_sequence:
        print(f"\n--- State Test: {step_name} ---")
        result = python_repl_tool.invoke({"code": test_code})
        assert "Successfully executed" in result
        print(f"✅ {step_name} completed")
        print(result[:200] + "..." if len(result) > 200 else result)
```

## 🎓 Educational Insights

### Why This Implementation is Production-Ready

1. **Robust Error Handling**
   - Catches all possible Python execution errors
   - Returns informative error messages
   - Prevents tool crashes from malformed code
   - Maintains system stability

2. **Clear Output Formatting**
   - Shows both executed code and results
   - Uses markdown formatting for readability
   - Provides clear success/failure indicators
   - Professional presentation suitable for reports

3. **Flexible Code Execution**
   - Accepts any valid Python code
   - Supports all standard libraries
   - Maintains execution state between calls
   - Handles complex multi-line scripts

4. **Agent-Friendly Interface**
   - Single parameter design for simplicity
   - Clear parameter description for AI agents
   - Consistent return format
   - Easy integration with agent workflows

### Common Student Mistakes & Solutions

#### ❌ Mistake 1: Missing PythonREPL Import

```python
# Wrong - missing import
from langchain_core.tools import tool
# Missing: from langchain_experimental.utilities import PythonREPL
```

**✅ Solution:** Always import PythonREPL from langchain_experimental

#### ❌ Mistake 2: Incorrect REPL Initialization

```python
# Wrong - not instantiating the class
repl = PythonREPL  # Missing parentheses

# Wrong - recreating REPL each time
@tool
def python_repl_tool(code):
    repl = PythonREPL()  # Creates new instance, loses state
    return repl.run(code)
```

**✅ Solution:** Initialize once and reuse the same instance

#### ❌ Mistake 3: Poor Error Handling

```python
# Wrong - no error handling
def python_repl_tool(code):
    return repl.run(code)  # Will crash on any error

# Wrong - too specific error catching
try:
    result = repl.run(code)
except SyntaxError as e:  # Misses other error types
    return f"Error: {e}"
```

**✅ Solution:** Use BaseException to catch all errors

#### ❌ Mistake 4: Inconsistent Output Formatting

```python
# Wrong - minimal output
return result

# Wrong - inconsistent formatting
return f"Code: {code}, Result: {result}"
```

**✅ Solution:** Use consistent, professional formatting with markdown

### Advanced Implementation Variations

#### Variation 1: Enhanced Security Tool

```python
import re
from typing import List

# Dangerous operations to restrict
RESTRICTED_PATTERNS = [
    r'import\s+os',
    r'import\s+subprocess',
    r'import\s+sys',
    r'__import__',
    r'exec\s*\(',
    r'eval\s*\(',
    r'open\s*\(',
    r'file\s*\(',
]

@tool
def secure_python_repl_tool(
    code: Annotated[str, "Safe Python code to execute"],
):
    """Secure Python execution with restricted operations."""
    
    # Check for dangerous patterns
    for pattern in RESTRICTED_PATTERNS:
        if re.search(pattern, code, re.IGNORECASE):
            return f"Security Error: Code contains restricted operation matching '{pattern}'"
    
    # Limit code length
    if len(code) > 2000:
        return "Error: Code too long. Maximum 2000 characters allowed."
    
    try:
        result = repl.run(code)
        # Limit output length
        if len(result) > 5000:
            result = result[:5000] + "\n... [Output truncated]"
            
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    
    return f"Successfully executed (secure mode):\n```python\n{code}\n```\n\nOutput:\n```\n{result}\n```"
```

#### Variation 2: Statistical Analysis Tool

```python
@tool
def stats_python_repl_tool(
    code: Annotated[str, "Python code for statistical analysis"],
    auto_imports: Annotated[bool, "Automatically import common stats libraries"] = True,
):
    """Python tool with automatic statistical library imports."""
    
    if auto_imports:
        imports = """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("Statistical libraries loaded: numpy, pandas, matplotlib, seaborn, scipy")
"""
        full_code = imports + "\n" + code
    else:
        full_code = code
    
    try:
        result = repl.run(full_code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    
    return f"Statistical analysis completed:\n```python\n{code}\n```\n\nResults:\n```\n{result}\n```"
```

#### Variation 3: Timed Execution Tool

```python
import time
from contextlib import contextmanager

@contextmanager
def timeout_context(seconds):
    """Context manager for execution timeout"""
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Code execution exceeded {seconds} seconds")
    
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

@tool
def timed_python_repl_tool(
    code: Annotated[str, "Python code to execute with timing"],
    timeout: Annotated[int, "Maximum execution time in seconds"] = 30,
):
    """Python tool with execution timing and timeout."""
    
    start_time = time.time()
    
    try:
        with timeout_context(timeout):
            result = repl.run(code)
        execution_time = time.time() - start_time
        
    except TimeoutError as e:
        return f"Execution timeout: {str(e)}"
    except BaseException as e:
        execution_time = time.time() - start_time
        return f"Failed after {execution_time:.2f}s. Error: {repr(e)}"
    
    return f"""Execution completed in {execution_time:.2f} seconds:

```python
{code}
```

Output:
```
{result}
```"""
```

## 📊 Performance Optimization Tips

### 1. Memory Management

```python
# Monitor memory usage
memory_test = """
import psutil
import os

process = psutil.Process(os.getpid())
memory_mb = process.memory_info().rss / 1024 / 1024

print(f"Current memory usage: {memory_mb:.2f} MB")

# Create some data to test memory
import numpy as np
large_array = np.random.random((1000, 1000))
new_memory_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

print(f"Memory after creating large array: {new_memory_mb:.2f} MB")
print(f"Memory increase: {new_memory_mb - memory_mb:.2f} MB")
"""
```

### 2. Execution Time Monitoring

```python
# Benchmark different operations
benchmark_test = """
import time
import numpy as np

def benchmark_operation(func, name):
    start = time.time()
    result = func()
    end = time.time()
    print(f"{name}: {end - start:.4f} seconds")
    return result

# Test different operations
def numpy_ops():
    return np.random.random((1000, 1000)).mean()

def list_ops():
    return sum([i**2 for i in range(10000)]) / 10000

def pandas_ops():
    import pandas as pd
    df = pd.DataFrame(np.random.random((1000, 10)))
    return df.mean().sum()

print("Performance benchmarks:")
benchmark_operation(numpy_ops, "NumPy operations")
benchmark_operation(list_ops, "Pure Python operations")
benchmark_operation(pandas_ops, "Pandas operations")
"""
```

## 🔗 Integration with Financial Analysis Tools

### Complete Workflow Example

```python
workflow_integration_test = """
# Demonstrate integration with previous tools
print("=== Complete Financial Analysis Workflow ===")
print()

# Step 1: Simulate company research (Wikipedia tool result)
company_info = {
    'name': 'Apple Inc.',
    'industry': 'Technology',
    'founded': '1976',
    'headquarters': 'Cupertino, California'
}

print("Step 1: Company Research Complete")
for key, value in company_info.items():
    print(f"  {key.title()}: {value}")
print()

# Step 2: Simulate stock data retrieval (Stock data tool result)
import pandas as pd
import numpy as np

# Generate sample stock data
dates = pd.date_range('2024-01-01', periods=30)
np.random.seed(42)
base_price = 180
returns = np.random.normal(0.001, 0.015, 30)
prices = [base_price]

for return_rate in returns[1:]:
    new_price = prices[-1] * (1 + return_rate)
    prices.append(new_price)

stock_data = pd.DataFrame({
    'Date': dates,
    'Close': prices,
    'Volume': np.random.randint(50000000, 150000000, 30)
})

print("Step 2: Stock Data Retrieved")
print(f"  Date Range: {stock_data['Date'].min().date()} to {stock_data['Date'].max().date()}")
print(f"  Price Range: ${stock_data['Close'].min():.2f} - ${stock_data['Close'].max():.2f}")
print(f"  Average Volume: {stock_data['Volume'].mean():,.0f}")
print()

# Step 3: Analysis and Visualization (Python tool)
import matplotlib.pyplot as plt

print("Step 3: Analysis and Visualization")

# Calculate technical indicators
stock_data['SMA_5'] = stock_data['Close'].rolling(window=5).mean()
stock_data['SMA_10'] = stock_data['Close'].rolling(window=10).mean()
stock_data['Daily_Return'] = stock_data['Close'].pct_change()

# Create comprehensive analysis chart
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Price and moving averages
ax1.plot(stock_data['Date'], stock_data['Close'], 'b-', label='Close Price', linewidth=2)
ax1.plot(stock_data['Date'], stock_data['SMA_5'], 'r--', label='5-day SMA', alpha=0.8)
ax1.plot(stock_data['Date'], stock_data['SMA_10'], 'g--', label='10-day SMA', alpha=0.8)
ax1.set_title(f'{company_info["name"]} - Stock Price with Moving Averages', fontsize=14)
ax1.set_ylabel('Price ($)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Volume
ax2.bar(stock_data['Date'], stock_data['Volume'], alpha=0.7, color='orange')
ax2.set_title('Trading Volume', fontsize=14)
ax2.set_ylabel('Volume')
ax2.tick_params(axis='x', rotation=45)
ax2.grid(True, alpha=0.3)

# Daily returns
returns_clean = stock_data['Daily_Return'].dropna()
colors = ['green' if x > 0 else 'red' for x in returns_clean]
ax3.bar(stock_data['Date'][1:], returns_clean * 100, color=colors, alpha=0.7)
ax3.set_title('Daily Returns (%)', fontsize=14)
ax3.set_ylabel('Return (%)')
ax3.tick_params(axis='x', rotation=45)
ax3.grid(True, alpha=0.3)

# Price distribution
ax4.hist(stock_data['Close'], bins=15, alpha=0.7, color='purple', edgecolor='black')
ax4.set_title('Price Distribution', fontsize=14)
ax4.set_xlabel('Price ($)')
ax4.set_ylabel('Frequency')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Summary statistics
total_return = ((stock_data['Close'].iloc[-1] / stock_data['Close'].iloc[0]) - 1) * 100
volatility = stock_data['Daily_Return'].std() * np.sqrt(252) * 100  # Annualized
avg_volume = stock_data['Volume'].mean()

print()
print("=== Analysis Summary ===")
print(f"Company: {company_info['name']} ({company_info['industry']})")
print(f"Analysis Period: {stock_data['Date'].iloc[0].date()} to {stock_data['Date'].iloc[-1].date()}")
print(f"Starting Price: ${stock_data['Close'].iloc[0]:.2f}")
print(f"Ending Price: ${stock_data['Close'].iloc[-1]:.2f}")
print(f"Total Return: {total_return:.2f}%")
print(f"Annualized Volatility: {volatility:.2f}%")
print(f"Average Daily Volume: {avg_volume:,.0f}")
print()
print("✅ Complete financial analysis workflow demonstrated!")
"""

result = python_repl_tool.invoke({"code": workflow_integration_test})
print(result)
```

## 📝 Assessment Rubric

### Functionality (50 points)
- **Tool creation:** Proper `@tool` decorator and function structure (15 pts)
- **REPL integration:** Correct PythonREPL initialization and usage (15 pts)
- **Code execution:** Successfully runs Python code and captures output (10 pts)
- **Error handling:** Robust exception catching and error reporting (10 pts)

### Code Quality (30 points)
- **Type annotations:** Proper use of `Annotated` parameters (10 pts)
- **Output formatting:** Professional result presentation (10 pts)
- **Code structure:** Clean, readable implementation (10 pts)

### Understanding (20 points)
- **Execution environment:** Understands PythonREPL state management (10 pts)
- **Integration concepts:** Explains how tool fits in agent workflows (10 pts)

### Total: 100 points

**Grading Scale:**
- 90-100: Excellent implementation with advanced understanding
- 80-89: Good functionality with minor optimization opportunities
- 70-79: Basic requirements met, needs improvement in robustness
- Below 70: Requires additional practice with execution environments

## 🚀 Advanced Applications

### Real-Time Data Analysis

```python
# Example: Real-time financial calculations
realtime_analysis = """
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Note: This would work with real yfinance data
# For demo, we'll simulate the API response
print("Real-time Financial Analysis Demo")
print("=" * 40)

# Simulate fetching current market data
current_time = datetime.now()
print(f"Analysis Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")

# Market summary simulation
market_data = {
    'AAPL': {'price': 185.50, 'change': 2.30, 'volume': 45000000},
    'MSFT': {'price': 280.75, 'change': -1.25, 'volume': 35000000},
    'GOOGL': {'price': 2650.25, 'change': 15.50, 'volume': 28000000}
}

print("\\nCurrent Market Snapshot:")
print("-" * 30)
for symbol, data in market_data.items():
    change_pct = (data['change'] / data['price']) * 100
    direction = "↗" if data['change'] > 0 else "↘"
    print(f"{symbol}: ${data['price']:.2f} ({direction} {data['change']:+.2f}, {change_pct:+.2f}%)")

# Calculate portfolio metrics
portfolio = {'AAPL': 100, 'MSFT': 50, 'GOOGL': 25}  # shares
total_value = sum(portfolio[symbol] * market_data[symbol]['price'] for symbol in portfolio)
print(f"\\nPortfolio Value: ${total_value:,.2f}")
"""
```

### Machine Learning Integration

```python
# Example: Simple predictive modeling
ml_integration = """
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

print("Stock Price Prediction Model Demo")
print("=" * 35)

# Generate sample historical data
np.random.seed(42)
days = np.arange(1, 101)  # 100 days of data
trend = 0.5 * days  # Upward trend
noise = np.random.normal(0, 5, 100)  # Market noise
prices = 100 + trend + noise

# Prepare features (using moving averages as predictors)
def calculate_features(prices, window=5):
    features = []
    targets = []
    for i in range(window, len(prices)):
        # Features: moving averages of different windows
        ma_short = np.mean(prices[i-window:i])
        ma_long = np.mean(prices[i-min(window*2, i):i])
        volume_proxy = np.std(prices[i-window:i])  # Using volatility as proxy
        
        features.append([ma_short, ma_long, volume_proxy])
        targets.append(prices[i])
    
    return np.array(features), np.array(targets)

X, y = calculate_features(prices)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model Performance:")
print(f"  Mean Squared Error: {mse:.2f}")
print(f"  R² Score: {r2:.3f}")
print(f"  Features: Moving Average (Short), Moving Average (Long), Volatility")

# Visualization
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(days, prices, 'b-', alpha=0.7, label='Historical Prices')
plt.title('Historical Stock Price Data')
plt.xlabel('Days')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title(f'Prediction Accuracy (R² = {r2:.3f})')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\\n✅ Machine learning model demonstration complete!")
"""
```

## 💡 Pro Tips for Instructors

1. **Start Simple**: Begin with basic arithmetic before moving to complex visualizations
2. **Emphasize State**: Show how variables persist across multiple executions
3. **Error Practice**: Intentionally introduce errors to demonstrate error handling
4. **Security Awareness**: Discuss the security implications of dynamic code execution
5. **Integration Focus**: Connect this tool to the previous Wikipedia and stock data tools
6. **Real Applications**: Show how this enables AI agents to perform complex analysis

## 🏁 Conclusion

This Python execution tool completes your comprehensive financial analysis toolkit, enabling:

- **Dynamic Code Generation**: AI agents can write and execute Python code
- **Data Visualization**: Create charts and graphs from retrieved data
- **Complex Calculations**: Perform sophisticated financial analysis
- **Flexible Integration**: Seamlessly work with other tools in agent workflows

The combination of Wikipedia research, stock data retrieval, and Python execution creates a powerful foundation for building intelligent financial analysis agents! 🎓📊
