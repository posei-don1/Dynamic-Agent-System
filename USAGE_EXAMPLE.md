# Dynamic DB+Math Pipeline Usage Example

This document shows how to use the new dynamic DB+Math pipeline that works with user-uploaded files for financial queries like moving averages.

## Overview

The pipeline supports queries like:
- "Tell me the Moving Average of AAPL from January to February 2024"
- "What's the 20-day MA for the uploaded stock data?"
- "Calculate RSI for the price data"
- "Show me Bollinger Bands for the uploaded financial data"

## Key Features

✅ **Dynamic File Support**: Works with any user-uploaded CSV/Excel file  
✅ **Flexible Column Detection**: Automatically detects price, date, and symbol columns  
✅ **CSV Metadata Handling**: Properly extracts metadata from first row as specified  
✅ **Python Tools**: Uses pure Python calculations (not LLM-based)  
✅ **Multiple Metrics**: Moving Average, EMA, Bollinger Bands, RSI, and more  
✅ **React-Ready**: Structured JSON responses for frontend integration  

## How It Works

1. **File Upload**: User uploads CSV/Excel file (previous uploads are cleared)
2. **Query Classification**: LLM classifies the query as "math_financial"
3. **Parameter Extraction**: LLM extracts calculation parameters and column hints
4. **Column Detection**: System intelligently detects price, date, symbol columns
5. **Data Retrieval**: Fetches relevant data using flexible column mapping
6. **Python Calculation**: Performs financial calculations using pure Python tools
7. **Response Formatting**: Returns structured JSON for UI consumption

## File Requirements

### CSV Files
- **First row after headers**: Contains metadata (extracted automatically)
- **Supported columns**: Any combination of:
  - **Price columns**: `close`, `price`, `adj_close`, `value`, `amount`
  - **Date columns**: `date`, `time`, `timestamp`
  - **Symbol columns**: `symbol`, `ticker`, `stock` (optional)
  - **Volume columns**: `volume`, `vol` (optional)

### Example CSV Structure
```csv
date,symbol,close,volume
2024-01-01,AAPL,180.00,1000000  # <- This row is metadata
2024-01-02,AAPL,182.50,1200000
2024-01-03,AAPL,185.20,1100000
...
```

## API Usage

### Step 1: Upload File
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@your_stock_data.csv"
```

### Step 2: Query Financial Metrics
```json
{
  "query": "Calculate the 20-day moving average for AAPL",
  "persona": "financial_analyst",
  "context": {}
}
```

### Example Response
```json
{
  "formatted_response": {
    "response": {
      "success": true,
      "file_path": "./data/structured.csv",
      "symbol": "AAPL",
      "calculation_type": "moving_average",
      "parameters": {
        "window": 20,
        "span": 12,
        "num_std": 2.0
      },
      "columns_used": {
        "symbol_column": "symbol",
        "date_column": "date", 
        "price_column": "close"
      },
      "calculation_result": {
        "success": true,
        "window": 20,
        "moving_average": [
          {"date": "2024-01-01", "moving_average": 180.00},
          {"date": "2024-01-02", "moving_average": 181.25},
          ...
        ],
        "calculation_type": "simple_moving_average"
      }
    },
    "file_used": "structured.csv",
    "columns_available": ["date", "symbol", "close", "volume"],
    "parameters_extracted": {
      "symbol": "AAPL",
      "calculation_type": "moving_average",
      "window": 20
    }
  },
  "metadata": {
    "processing_time": 0.45,
    "system_mode": "math_financial_pipeline"
  },
  "status": "success"
}
```

## Supported Calculations

### 1. Moving Average
```
Query: "Calculate 20-day moving average"
Python Tool: moving_average(data, window=20)
```

### 2. Exponential Moving Average (EMA)
```
Query: "Show me EMA with span 12"
Python Tool: exponential_moving_average(data, span=12)
```

### 3. Bollinger Bands
```
Query: "Calculate Bollinger Bands with 2 standard deviations"
Python Tool: bollinger_bands(data, window=20, num_std=2.0)
```

### 4. Relative Strength Index (RSI)
```
Query: "What's the RSI for this data?"
Python Tool: rsi(data, window=14)
```

## Testing

Run the comprehensive test script:
```bash
cd Dynamic-Agent-System
python test_dynamic_pipeline.py
```

This tests:
- ✅ CSV loading with metadata extraction
- ✅ Flexible column detection
- ✅ Multiple file formats (with/without symbol columns)
- ✅ All financial calculations (MA, EMA, Bollinger Bands, RSI)
- ✅ Error handling and edge cases

## Future React Integration

The structured responses make React integration straightforward:

```javascript
// Example React component
const FinancialChart = ({ response }) => {
  const { calculation_result, symbol, calculation_type } = response;
  
  return (
    <div className="financial-chart">
      <h3>{symbol} - {calculation_type}</h3>
      <div className="chart-info">
        <p>File: {response.file_used}</p>
        <p>Columns: {response.columns_available.join(', ')}</p>
      </div>
      <LineChart data={calculation_result.moving_average} />
    </div>
  );
};
```

## Error Handling

The system provides clear error messages:
- **No file uploaded**: "No structured data file found. Please upload a CSV or Excel file first."
- **Column detection failed**: "No price column found. Available columns: [...]"
- **Invalid date range**: "Date filtering failed: [specific error]"
- **Calculation errors**: "Failed to compute moving average: [specific error]"

## File Management

- **Automatic cleanup**: Previous uploads are automatically deleted
- **Single file context**: System works with the most recently uploaded file
- **Metadata preservation**: CSV metadata is extracted and preserved
- **Column flexibility**: Works with various column naming conventions 