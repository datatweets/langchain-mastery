# 📈 Activity 02: Stock Performance Data Tool - Student Practice Guide

## 🎯 Learning Objectives

By the end of this activity, you will:
- Build a tool that reads and filters CSV data using pandas
- Learn to handle multiple input parameters in LangChain tools
- Master date filtering and data manipulation techniques
- Understand how to convert between different time units (days, weeks, months)
- Create tools that work with local file systems

## 📚 Background Context

**What is this tool for?**
You're building the second component of your financial analysis system:
1. ✅ **Wikipedia tool** → Company information (completed in Activity 01)
2. 🔄 **Stock data tool** → Historical stock performance (this activity)
3. ⭐ **Python execution tool** → Data visualization (next activity)

**Why Stock Data Tool?**
- **Financial Analysis**: Retrieve historical stock prices
- **Data Processing**: Filter data by time periods
- **Flexible Queries**: Handle different time ranges (days, weeks, months)
- **Local Data**: Work with CSV files efficiently

## 🏢 Available Stock Data

Your tool will work with these Fortune 500 companies:

| Company Name | Ticker Symbol | Description |
|--------------|---------------|-------------|
| Apple        | AAPL          | Technology/Consumer Electronics |
| Microsoft    | MSFT          | Technology/Software |
| Amazon       | AMZN          | E-commerce/Cloud Services |
| Meta         | META          | Social Media/Technology |
| Netflix      | NFLX          | Streaming/Entertainment |
| Tesla        | TSLA          | Electric Vehicles/Clean Energy |

**Data Structure:**
Each CSV contains: `Date`, `Close/Last`, `Volume`, `Open`, `High`, `Low`

## 🔧 Setup Instructions

### Step 1: Install Required Libraries
```bash
pip install --quiet langchain-core==0.3.59 pandas
```

### Step 2: Import Dependencies
```python
# TODO: Import the required modules
# Hint: You need os, typing.Annotated, pandas, and langchain_core.tools.tool
import ________
from typing import ________
import pandas as ________
from langchain_core.tools import ________
```

<details>
<summary>💡 Hint for Step 2</summary>

You need to import:
- `os` for file system operations
- `Annotated` from typing for parameter descriptions  
- `pandas as pd` for data manipulation
- `tool` decorator from langchain_core.tools
</details>

## 🏗️ Building the Stock Data Tool

### Step 3: Create the Tool Function Structure

**Your task:** Complete the `stock_data_tool` function below. You need to write about **70%** of the implementation.

```python
# TODO: Step 3a - Add the tool decorator
________
def ________(
    company_ticker: Annotated[str, "The ticker symbol of the company to retrieve their stock performance data."], 
    num_days: Annotated[int, "The number of days of stock data required to respond to the user query."]
) -> str:
    """
    Use this to look-up stock performance data for companies to retrieve a table from a CSV. 
    You may need to convert company names into ticker symbols to call this function, 
    e.g, Apple Inc. -> AAPL, and you may need to convert weeks, months, and years, into days.
    """
    
    # TODO: Step 3b - Build the file path
    # Hint: Use f-string with "data/{company_ticker}.csv"
    file_path = f"________/{________}.csv"

    # TODO: Step 3c - Check if file exists
    # Hint: Use os.path.exists() and check if it's False
    if ________.path.exists(________) is ________:
        return f"Sorry, but data for company {company_ticker} is not available. Please try Apple, Amazon, Meta, Microsoft, Netflix, or Tesla."
    
    # TODO: Step 3d - Read CSV file with proper date parsing
    # Hint: Use pd.read_csv() with index_col='Date' and parse_dates=True
    stock_df = pd.________(________, index_col='________', parse_dates=________)

    # TODO: Step 3e - Ensure index is in date format
    # Hint: Convert stock_df.index to .date
    stock_df.index = stock_df.________.________
    
    # Maximum num_days supported by the dataset
    max_num_days = (stock_df.index.max() - stock_df.index.min()).days
    
    if num_days > max_num_days:
        return "Sorry, but this time period exceeds the data available. Please reduce it to continue."
    
    # TODO: Step 3f - Get the most recent date
    # Hint: Use .max() on the index
    final_date = stock_df.________.________()

    # TODO: Step 3g - Filter DataFrame for last num_days
    # Hint: Filter where index > (final_date - pd.Timedelta(days=num_days))
    filtered_df = stock_df[stock_df.________ > (________ - pd.Timedelta(days=________))]

    # TODO: Step 3h - Return formatted result with markdown table
    # Hint: Use .to_markdown() method on filtered_df
    return f"Successfully executed the stock performance data retrieval tool to retrieve the last *{num_days} days* of data for company **{company_ticker}**:\\n\\n{filtered_df.________()}"
```

<details>
<summary>🔍 Step-by-Step Hints</summary>

**Step 3a:** Use `@tool` decorator
**Step 3b:** `file_path = f"data/{company_ticker}.csv"`
**Step 3c:** `if os.path.exists(file_path) is False:`
**Step 3d:** `pd.read_csv(file_path, index_col='Date', parse_dates=True)`
**Step 3e:** `stock_df.index.date`
**Step 3f:** `stock_df.index.max()`
**Step 3g:** Filter with index and Timedelta
**Step 3h:** Use `to_markdown()` method
</details>

### Step 4: Test Your Tool

Test your stock data tool with different scenarios:

```python
# TODO: Step 4a - Test with META stock for 4 days
# Hint: Use .invoke() with a dictionary containing company_ticker and num_days
retrieved_data = stock_data_tool.invoke({
    "company_ticker": "________", 
    "num_days": ________
})
print(retrieved_data)

# TODO: Step 4b - Test with AAPL for 7 days
aapl_data = stock_data_tool.invoke({
    "________": "AAPL", 
    "________": 7
})
print("\\n--- Apple Stock Data ---")
print(aapl_data)

# TODO: Step 4c - Test error handling with invalid ticker
error_test = stock_data_tool.invoke({
    "company_ticker": "________",  # Use "INVALID" 
    "num_days": 5
})
print("\\n--- Error Test ---")
print(error_test)
```

<details>
<summary>💡 Testing Hints</summary>

- Use `"META"` and `4` for the first test
- Parameter names are `"company_ticker"` and `"num_days"`  
- Use `"INVALID"` to test error handling
- Tools with multiple parameters need dictionary input
</details>

### Step 5: Enhanced Display (Bonus)

```python
# TODO: Step 5 - Display data as formatted table
# Hint: Import and use IPython.display functions
from IPython.display import ________, ________

# Display the retrieved data as a nicely formatted table
________(________(retrieved_data))
```

## ✅ Expected Output

Your tool should return something like:

```
Successfully executed the stock performance data retrieval tool to retrieve the last *4 days* of data for company **META**:

| Date       | Close/Last | Volume   | Open   | High   | Low    |
|------------|------------|----------|--------|--------|--------|
| 2025-05-28 | $515.42   | 15339680 | $515.59| $517.73| $512.90|
| 2025-05-27 | $510.21   | 18288480 | $508.30| $515.74| $507.43|
| 2025-05-23 | $505.27   | 20432920 | $503.65| $507.70| $503.46|
| 2025-05-22 | $511.36   | 16742410 | $510.71| $512.75| $509.70|
```

## 🎓 Understanding Your Code

### Key Concepts Explained:

**1. Multiple Parameters in Tools:**
```python
def stock_data_tool(
    company_ticker: Annotated[str, "Description"], 
    num_days: Annotated[int, "Description"]
) -> str:
```
- Tools can accept multiple parameters
- Each parameter needs its own `Annotated` type hint
- Return type annotation (`-> str`) is good practice

**2. Pandas Date Operations:**
```python
stock_df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
stock_df.index = stock_df.index.date
```
- `parse_dates=True` automatically converts date strings to datetime objects
- `.date` extracts just the date part (removes time)
- Index becomes the Date column for efficient filtering

**3. Date Filtering with Timedelta:**
```python
final_date = stock_df.index.max()
filtered_df = stock_df[stock_df.index > (final_date - pd.Timedelta(days=num_days))]
```
- `pd.Timedelta(days=num_days)` creates a time period
- Subtracting from `final_date` gives us the start date
- Boolean indexing filters the DataFrame

**4. File Path Construction:**
```python
file_path = f"data/{company_ticker}.csv"
```
- F-strings create dynamic file paths
- Each company has its own CSV file
- Consistent naming convention: `TICKER.csv`

## 🔧 Troubleshooting Guide

### Common Issues & Solutions:

**❌ "FileNotFoundError"**
```bash
# Solution: Make sure the data folder exists with CSV files
mkdir -p data
# Copy the provided CSV files to the data folder
```

**❌ "KeyError: 'Date'"**
```python
# Problem: CSV doesn't have expected column name
# Solution: Check the CSV structure - first row should have "Date" column
```

**❌ "Empty DataFrame"**
- **Check:** Your date filtering logic
- **Debug:** Print `final_date` and `num_days` to verify calculation
- **Try:** Smaller `num_days` value

**❌ "Tool invoke error with dictionary"**
```python
# Wrong - single parameter format
stock_data_tool.invoke("AAPL")

# Right - dictionary format for multiple parameters  
stock_data_tool.invoke({"company_ticker": "AAPL", "num_days": 5})
```

## 🧪 Testing Challenges

### Challenge 1: Time Period Conversions
```python
# Test converting different time periods to days
# Hint: 1 week ≈ 7 days, 1 month ≈ 30 days, 1 year ≈ 365 days

test_periods = [
    ("1 week", 7),
    ("2 weeks", 14), 
    ("1 month", 30),
    ("3 months", 90)
]

for period_name, days in test_periods:
    print(f"\\n--- {period_name} of TSLA data ---")
    data = stock_data_tool.invoke({"company_ticker": "TSLA", "num_days": days})
    # Count lines to verify data amount
    lines = len(data.split('\\n'))
    print(f"Retrieved {lines} lines of data")
```

### Challenge 2: Compare Different Companies
```python
# Compare stock data across companies for the same period
companies = ["AAPL", "MSFT", "AMZN", "META", "NFLX", "TSLA"]
days = 5

for ticker in companies:
    data = stock_data_tool.invoke({"company_ticker": ticker, "num_days": days})
    print(f"\\n=== {ticker} - Last {days} days ===")
    print(data)
```

### Challenge 3: Edge Case Testing
```python
# Test edge cases
edge_cases = [
    {"company_ticker": "AAPL", "num_days": 1000},    # Too many days
    {"company_ticker": "FAKE", "num_days": 5},       # Invalid ticker
    {"company_ticker": "AAPL", "num_days": 0},       # Zero days
    {"company_ticker": "", "num_days": 5},           # Empty ticker
]

for test_case in edge_cases:
    result = stock_data_tool.invoke(test_case)
    print(f"\\nTest: {test_case}")
    print(f"Result: {result[:100]}...")
```

## 🚀 Next Steps

After completing this activity:

1. **Activity 03:** Build a Python code execution tool for data visualization
2. **Activity 04:** Combine all three tools in a single agent workflow  
3. **Activity 05:** Add conditional routing for intelligent tool selection
4. **Advanced:** Connect to real-time stock APIs like Yahoo Finance or Alpha Vantage

## 📝 Self-Assessment

**Check your understanding:**

□ I can build tools with multiple parameters  
□ I understand how to use pandas for date filtering  
□ I can handle file system operations in tools  
□ I know how to invoke tools with dictionary inputs  
□ I understand time period conversions (days ↔ weeks ↔ months)  
□ I can debug data filtering issues  
□ I can format output with markdown tables  

## 💡 Real-World Applications

**Where this pattern is used:**
- **Financial APIs:** Retrieving stock, forex, or crypto data
- **Database queries:** Filtering records by date ranges  
- **Log analysis:** Getting recent log entries
- **Analytics dashboards:** Time-series data visualization
- **Report generation:** Automated periodic reports

## 🎉 Congratulations!

You've successfully created a powerful stock data retrieval tool! This tool can:

- ✅ **Read CSV files** efficiently with pandas
- ✅ **Filter data by date ranges** using Timedelta
- ✅ **Handle multiple parameters** in LangChain tools
- ✅ **Provide error handling** for missing data
- ✅ **Return formatted tables** ready for display

**Key Takeaways:**
- Multiple parameter tools need dictionary inputs
- Pandas makes date filtering straightforward
- Proper error handling improves user experience
- Markdown formatting makes data readable

Ready to build that Python execution tool? Let's create some visualizations! 📊🚀