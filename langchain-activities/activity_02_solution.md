# 📊 Activity 02: Stock Performance Data Tool - Master Solution Guide

## 📋 Activity Overview

**Topic:** Building a stock data retrieval tool with pandas and CSV processing  
**Duration:** 45-60 minutes  
**Difficulty:** Intermediate  
**Prerequisites:** Basic Python, pandas fundamentals, file operations

## 🏆 Complete Solution

### Step 1: Environment Setup

```python
# Install required libraries
!pip install --quiet langchain-core==0.3.59 pandas
```

### Step 2: Import Dependencies

```python
import os
from typing import Annotated
import pandas as pd
from langchain_core.tools import tool
```

**Explanation:**
- `os`: File system operations for checking file existence
- `Annotated`: Type hints with metadata for LangChain tool parameters
- `pandas`: Data manipulation and CSV processing
- `tool`: Decorator to convert Python functions into LangChain tools

### Step 3: Complete Stock Data Tool Implementation

```python
@tool
def stock_data_tool(
    company_ticker: Annotated[str, "The ticker symbol of the company to retrieve their stock performance data."], 
    num_days: Annotated[int, "The number of days of stock data required to respond to the user query."]
) -> str:
    """
    Use this to look-up stock performance data for companies to retrieve a table from a CSV. 
    You may need to convert company names into ticker symbols to call this function, 
    e.g, Apple Inc. -> AAPL, and you may need to convert weeks, months, and years, into days.
    """
    
    # Load the CSV for the company requested
    file_path = f"data/{company_ticker}.csv"

    if os.path.exists(file_path) is False:
        return f"Sorry, but data for company {company_ticker} is not available. Please try Apple, Amazon, Meta, Microsoft, Netflix, or Tesla."
    
    stock_df = pd.read_csv(file_path, index_col='Date', parse_dates=True)

    # Ensure the index is in date format
    stock_df.index = stock_df.index.date
    
    # Maximum num_days supported by the dataset
    max_num_days = (stock_df.index.max() - stock_df.index.min()).days
    
    if num_days > max_num_days:
        return "Sorry, but this time period exceeds the data available. Please reduce it to continue."
    
    # Get the most recent date in the DataFrame
    final_date = stock_df.index.max()

    # Filter the DataFrame to get the last num_days of stock data
    filtered_df = stock_df[stock_df.index > (final_date - pd.Timedelta(days=num_days))]

    return f"Successfully executed the stock performance data retrieval tool to retrieve the last *{num_days} days* of data for company **{company_ticker}**:\\n\\n{filtered_df.to_markdown()}"
```

### Step 4: Testing Implementation

```python
# Test with META stock for 4 days
retrieved_data = stock_data_tool.invoke({"company_ticker": "META", "num_days": 4})
print(retrieved_data)

# Test with AAPL for 7 days
aapl_data = stock_data_tool.invoke({"company_ticker": "AAPL", "num_days": 7})
print("\\n--- Apple Stock Data ---")
print(aapl_data)

# Test error handling with invalid ticker
error_test = stock_data_tool.invoke({"company_ticker": "INVALID", "num_days": 5})
print("\\n--- Error Test ---")
print(error_test)
```

### Step 5: Enhanced Display

```python
from IPython.display import display, Markdown

# Display as formatted table
display(Markdown(retrieved_data))
```

## 🧠 Code Breakdown & Best Practices

### 1. Multi-Parameter Tool Design

```python
@tool
def stock_data_tool(
    company_ticker: Annotated[str, "The ticker symbol of the company to retrieve their stock performance data."], 
    num_days: Annotated[int, "The number of days of stock data required to respond to the user query."]
) -> str:
```

**Key Design Decisions:**
- **Two parameters:** Both are required for tool functionality
- **Clear descriptions:** Help AI agents understand parameter purposes
- **Type hints:** `str` for ticker, `int` for days, `-> str` for return
- **Professional docstring:** Provides usage examples and conversion hints

### 2. File Path Construction and Validation

```python
file_path = f"data/{company_ticker}.csv"

if os.path.exists(file_path) is False:
    return f"Sorry, but data for company {company_ticker} is not available. Please try Apple, Amazon, Meta, Microsoft, Netflix, or Tesla."
```

**Best Practices:**
- **Dynamic path construction:** Uses ticker symbol for file identification
- **File existence check:** Prevents FileNotFoundError
- **Helpful error messages:** Lists available companies
- **Explicit comparison:** `is False` is more readable than `not`

### 3. Advanced Pandas Date Processing

```python
stock_df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
stock_df.index = stock_df.index.date
```

**Technical Details:**
- **`index_col='Date'`:** Sets Date column as DataFrame index for efficient filtering
- **`parse_dates=True`:** Automatically converts date strings to datetime objects
- **`.date` conversion:** Strips time components, keeping only date for consistency
- **Index-based operations:** Enable efficient time-series filtering

### 4. Date Range Calculation and Validation

```python
# Maximum num_days supported by the dataset
max_num_days = (stock_df.index.max() - stock_df.index.min()).days

if num_days > max_num_days:
    return "Sorry, but this time period exceeds the data available. Please reduce it to continue."

# Get the most recent date in the DataFrame
final_date = stock_df.index.max()

# Filter the DataFrame to get the last num_days of stock data
filtered_df = stock_df[stock_df.index > (final_date - pd.Timedelta(days=num_days))]
```

**Advanced Techniques:**
- **Data range validation:** Calculates maximum available days
- **Boundary checking:** Prevents impossible queries
- **Recent data focus:** Always gets the most current data
- **Timedelta filtering:** Efficient date arithmetic for range selection
- **Boolean indexing:** Pandas-optimized filtering technique

### 5. Professional Output Formatting

```python
return f"Successfully executed the stock performance data retrieval tool to retrieve the last *{num_days} days* of data for company **{company_ticker}**:\\n\\n{filtered_df.to_markdown()}"
```

**Formatting Features:**
- **Clear status indication:** "Successfully executed" confirms operation
- **Markdown formatting:** Bold and italic text for emphasis
- **Table output:** `to_markdown()` creates readable tables
- **Contextual information:** Includes parameters used in the query

## 🧪 Comprehensive Testing Suite

### Test Suite 1: Basic Functionality

```python
def test_basic_functionality():
    """Test tool with all available companies"""
    companies = ["AAPL", "MSFT", "AMZN", "META", "NFLX", "TSLA"]
    test_days = [1, 5, 10, 30]
    
    for ticker in companies:
        for days in test_days:
            print(f"\\n--- Testing: {ticker} for {days} days ---")
            result = stock_data_tool.invoke({
                "company_ticker": ticker, 
                "num_days": days
            })
            
            # Assertions for automated testing
            assert "Successfully executed" in result
            assert ticker in result
            assert str(days) in result
            assert "| Date" in result  # Check for table header
            
            print("✅ Test passed")
            print(result[:200] + "..." if len(result) > 200 else result)
```

### Test Suite 2: Edge Cases and Error Handling

```python
def test_edge_cases():
    """Test error handling and boundary conditions"""
    
    edge_cases = [
        # Invalid ticker symbols
        ({"company_ticker": "INVALID", "num_days": 5}, "not available"),
        ({"company_ticker": "", "num_days": 5}, "not available"),
        ({"company_ticker": "XYZ123", "num_days": 10}, "not available"),
        
        # Boundary conditions
        ({"company_ticker": "AAPL", "num_days": 0}, "Successfully executed"),  # Should work
        ({"company_ticker": "AAPL", "num_days": 1}, "Successfully executed"),
        ({"company_ticker": "AAPL", "num_days": 1000}, "exceeds the data available"),
        ({"company_ticker": "AAPL", "num_days": 9999}, "exceeds the data available"),
        
        # Case sensitivity
        ({"company_ticker": "aapl", "num_days": 5}, "not available"),  # Should fail
        ({"company_ticker": "Apple", "num_days": 5}, "not available"),  # Should fail
    ]
    
    for test_input, expected_phrase in edge_cases:
        print(f"\\n--- Testing: {test_input} ---")
        result = stock_data_tool.invoke(test_input)
        
        # Check if expected phrase is in result
        assert expected_phrase in result, f"Expected '{expected_phrase}' in result"
        print(f"✅ Expected behavior: Contains '{expected_phrase}'")
        print(f"Result: {result[:100]}...")
```

### Test Suite 3: Data Quality and Format Validation

```python
def test_data_quality():
    """Validate data format and content quality"""
    
    test_cases = [
        {"company_ticker": "AAPL", "num_days": 5},
        {"company_ticker": "TSLA", "num_days": 10},
        {"company_ticker": "META", "num_days": 3}
    ]
    
    for test_case in test_cases:
        result = stock_data_tool.invoke(test_case)
        
        # Check markdown table format
        lines = result.split('\\n')
        table_lines = [line for line in lines if '|' in line]
        
        assert len(table_lines) > 2, "Should have header, separator, and data rows"
        assert "Date" in table_lines[0], "Header should contain Date"
        assert "Close/Last" in table_lines[0], "Header should contain Close/Last"
        assert "Volume" in table_lines[0], "Header should contain Volume"
        
        # Check data rows count (should be approximately num_days)
        data_rows = len(table_lines) - 2  # Subtract header and separator
        expected_days = test_case["num_days"]
        
        # Allow some flexibility for weekends and holidays
        assert data_rows <= expected_days + 2, f"Too many rows: {data_rows} for {expected_days} days"
        assert data_rows >= max(1, expected_days - 5), f"Too few rows: {data_rows} for {expected_days} days"
        
        print(f"✅ {test_case['company_ticker']}: {data_rows} rows for {expected_days} days request")
```

### Test Suite 4: Performance Testing

```python
import time

def test_performance():
    """Test tool response times and memory usage"""
    
    performance_tests = [
        {"company_ticker": "AAPL", "num_days": 1},    # Small query
        {"company_ticker": "AAPL", "num_days": 30},   # Medium query  
        {"company_ticker": "AAPL", "num_days": 100},  # Large query
    ]
    
    for test_case in performance_tests:
        start_time = time.time()
        result = stock_data_tool.invoke(test_case)
        end_time = time.time()
        
        execution_time = end_time - start_time
        data_size = len(result)
        
        print(f"Query: {test_case}")
        print(f"Execution time: {execution_time:.3f}s")
        print(f"Result size: {data_size} characters")
        
        # Performance assertions
        assert execution_time < 5.0, f"Too slow: {execution_time}s"
        assert data_size > 0, "Empty result"
        
        print("✅ Performance test passed\\n")
```

## 🎓 Educational Insights

### Why This Implementation is Production-Ready

1. **Robust Error Handling**
   - File existence checking prevents crashes
   - Data range validation prevents impossible queries
   - Clear error messages guide user behavior
   - Graceful degradation for edge cases

2. **Efficient Data Processing**
   - Index-based filtering for optimal performance
   - Date conversion for consistent operations
   - Memory-efficient pandas operations
   - Minimal data loading (only what's needed)

3. **Flexible Interface Design**
   - Multiple parameter support
   - Clear parameter descriptions for AI agents
   - Professional output formatting
   - Easy integration with display systems

4. **Scalable Architecture**
   - Configurable data directory structure
   - Extensible to additional ticker symbols
   - Compatible with real-time data APIs
   - Modular design for easy modification

### Common Student Mistakes & Solutions

#### ❌ Mistake 1: Incorrect Tool Invocation
```python
# Wrong - single parameter format for multi-parameter tool
stock_data_tool.invoke("AAPL")
```

**✅ Solution:** Use dictionary format for multiple parameters
```python
stock_data_tool.invoke({"company_ticker": "AAPL", "num_days": 5})
```

#### ❌ Mistake 2: Poor Date Filtering
```python
# Wrong - inefficient string-based filtering
filtered_df = stock_df[stock_df['Date'].str.contains("2025-05")]
```

**✅ Solution:** Use proper date indexing and Timedelta
```python
filtered_df = stock_df[stock_df.index > (final_date - pd.Timedelta(days=num_days))]
```

#### ❌ Mistake 3: Missing Error Handling
```python
# Wrong - no file existence check
stock_df = pd.read_csv(file_path)  # Will crash if file doesn't exist
```

**✅ Solution:** Always validate file existence
```python
if os.path.exists(file_path) is False:
    return f"Sorry, but data for company {company_ticker} is not available."
```

#### ❌ Mistake 4: Inconsistent Date Formats
```python
# Wrong - mixing datetime and string operations
stock_df.index = pd.to_datetime(stock_df.index)
# Later: string comparison fails
```

**✅ Solution:** Consistent date format conversion
```python
stock_df.index = stock_df.index.date  # Consistent date objects
```

## 🔧 Tool Variations & Extensions

### Variation 1: Enhanced Stock Metrics

```python
@tool
def enhanced_stock_tool(
    company_ticker: Annotated[str, "Company ticker symbol"],
    num_days: Annotated[int, "Number of days of data"],
    include_metrics: Annotated[bool, "Calculate additional metrics"] = False
):
    """Enhanced stock tool with optional metrics calculation."""
    
    # ... basic implementation ...
    
    if include_metrics:
        # Calculate additional metrics
        filtered_df['Daily_Change'] = filtered_df['Close/Last'] - filtered_df['Open']
        filtered_df['Daily_Change_Pct'] = (filtered_df['Daily_Change'] / filtered_df['Open']) * 100
        filtered_df['Volume_MA'] = filtered_df['Volume'].rolling(window=3).mean()
    
    return f"Enhanced stock data with metrics:\\n\\n{filtered_df.to_markdown()}"
```

### Variation 2: Multi-Company Comparison Tool

```python
@tool
def compare_stocks_tool(
    company_tickers: Annotated[list, "List of ticker symbols to compare"],
    num_days: Annotated[int, "Number of days for comparison"]
):
    """Compare multiple stocks side by side."""
    
    comparison_data = {}
    
    for ticker in company_tickers:
        file_path = f"data/{ticker}.csv"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
            df.index = df.index.date
            final_date = df.index.max()
            filtered_df = df[df.index > (final_date - pd.Timedelta(days=num_days))]
            comparison_data[ticker] = filtered_df['Close/Last'].iloc[-1]  # Latest price
    
    # Create comparison table
    comparison_df = pd.DataFrame(list(comparison_data.items()), 
                                columns=['Ticker', 'Latest_Price'])
    
    return f"Stock comparison:\\n\\n{comparison_df.to_markdown()}"
```

### Variation 3: Time Period Conversion Tool

```python
@tool
def flexible_period_stock_tool(
    company_ticker: Annotated[str, "Company ticker symbol"],
    time_period: Annotated[str, "Time period: '5d', '2w', '1m', '3m', '1y'"]
):
    """Stock tool with flexible time period specification."""
    
    # Convert period to days
    period_mapping = {
        'd': 1, 'w': 7, 'm': 30, 'y': 365
    }
    
    # Parse time period string (e.g., "5d", "2w", "1m")
    if len(time_period) < 2:
        return "Invalid time period format. Use format like '5d', '2w', '1m'."
    
    try:
        number = int(time_period[:-1])
        unit = time_period[-1].lower()
        
        if unit not in period_mapping:
            return "Invalid time unit. Use 'd' (days), 'w' (weeks), 'm' (months), 'y' (years)."
        
        num_days = number * period_mapping[unit]
        
    except ValueError:
        return "Invalid time period format. Use format like '5d', '2w', '1m'."
    
    # Use standard stock_data_tool logic
    return stock_data_tool.invoke({"company_ticker": company_ticker, "num_days": num_days})
```

## 📊 Data Format Understanding

### CSV Structure Analysis

```python
def analyze_csv_structure():
    """Analyze the structure of stock CSV files"""
    
    sample_files = ["AAPL.csv", "TSLA.csv", "META.csv"]
    
    for filename in sample_files:
        filepath = f"data/{filename}"
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            
            print(f"\\n=== {filename} Analysis ===")
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
            print(f"Sample data:\\n{df.head()}")
            
            # Check data types
            print(f"\\nData types:")
            print(df.dtypes)
            
            # Check for missing values
            print(f"\\nMissing values:")
            print(df.isnull().sum())
```

### Expected Data Structure

| Column     | Type    | Description                    | Example      |
|------------|---------|--------------------------------|--------------|
| Date       | String  | Date in MM/DD/YYYY format     | "05/28/2025" |
| Close/Last | String  | Closing price with $ symbol   | "$200.42"    |
| Volume     | Integer | Number of shares traded       | 45339680     |
| Open       | String  | Opening price with $ symbol   | "$200.59"    |
| High       | String  | Highest price with $ symbol   | "$202.73"    |
| Low        | String  | Lowest price with $ symbol    | "$199.90"    |

## 📝 Assessment Rubric

### Functionality (50 points)
- **Tool creation:** Proper `@tool` decorator and function structure (15 pts)
- **File operations:** Correct CSV reading and path handling (10 pts)
- **Date filtering:** Accurate Timedelta-based filtering (15 pts)
- **Error handling:** Robust file existence and range checking (10 pts)

### Code Quality (30 points)
- **Type annotations:** Proper use of `Annotated` parameters (10 pts)
- **Pandas operations:** Efficient DataFrame operations (10 pts)
- **Code structure:** Clean, readable implementation (10 pts)

### Understanding (20 points)
- **Multi-parameter tools:** Understands dictionary-based invocation (10 pts)
- **Date operations:** Explains Timedelta and index filtering (10 pts)

### Total: 100 points

**Grading Scale:**
- 90-100: Excellent implementation with advanced understanding
- 80-89: Good functionality with minor optimization opportunities
- 70-79: Basic requirements met, needs improvement in efficiency
- Below 70: Requires additional practice with pandas and tool design

## 🚀 Integration with Real-Time APIs

### Extending to Live Data

```python
# Example: Yahoo Finance integration (requires yfinance package)
import yfinance as yf

@tool
def live_stock_tool(
    company_ticker: Annotated[str, "Company ticker symbol"],
    period: Annotated[str, "Period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max"] = "1mo"
):
    """Retrieve live stock data from Yahoo Finance."""
    try:
        stock = yf.Ticker(company_ticker)
        data = stock.history(period=period)
        
        if data.empty:
            return f"No data found for ticker {company_ticker}"
        
        # Format similar to CSV tool
        return f"Live stock data for {company_ticker}:\\n\\n{data.to_markdown()}"
        
    except Exception as e:
        return f"Error retrieving live data: {str(e)}"
```

## 💡 Pro Tips for Instructors

1. **Emphasize Data Validation:** Show students why checking file existence prevents crashes
2. **Demonstrate Date Operations:** Walk through Timedelta calculations step by step
3. **Practice Multi-Parameter Syntax:** Students often struggle with dictionary invocation
4. **Show Real-World Extensions:** Connect to live APIs for practical applications
5. **Debug Common Errors:** Practice with missing files and invalid tickers

This activity builds essential data processing skills that are fundamental to financial analysis and time-series applications! 📈