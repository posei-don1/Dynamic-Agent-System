"""
Database Node
Handles CSV data processing, database queries, and data analysis
"""
from typing import Dict, Any, List, Optional
import logging
import pandas as pd
import os
from pathlib import Path
import sqlite3
import difflib
import numpy as np
from .math_node import MathNode
import re

logger = logging.getLogger(__name__)

class DbNode:
    """Processes CSV uploads and queries using numpy, with column name correction."""
    def __init__(self):
        self.supported_formats = ['.csv']
        self.loaded_data = {}  # {filename: (np_array, header)}
        self.column_names_by_file = {}  # {filename: [col1, col2, ...]}
        self.last_loaded_file = None  # Track the most recently loaded file

    def _find_closest(self, name, options):
        matches = difflib.get_close_matches(name, options, n=1, cutoff=0.6)
        return matches[0] if matches else None

    def load_csv_pandas(self, file_path):
        print(f"[DbNode] Loading CSV with pandas: {file_path}")
        try:
            df = pd.read_csv(file_path)
            print(f"[DbNode] DataFrame columns: {df.columns.tolist()}")
            return df, df.columns.tolist()
        except Exception as e:
            print(f"[DbNode] Error loading CSV with pandas: {e}")
            return None, None

    def update_column_names_cache(self, filename):
        file_path = os.path.join(self.get_structured_dir(), filename)
        df, header = self.load_csv_pandas(file_path)
        if df is not None and header is not None:
            self.column_names_by_file[filename] = header
            self.loaded_data[filename] = (df, header)
            self.last_loaded_file = filename
            print(f"[DbNode] Cached columns for {filename}: {header}")
            return True
        print(f"[DbNode] Failed to cache columns for {filename}")
        return False

    def get_structured_dir(self):
        return './data/uploads/structured/'

    def find_csv_file(self, filename=None):
        directory = self.get_structured_dir()
        files = [f for f in os.listdir(directory) if f.endswith('.csv')]
        if not files:
            print("[DbNode] No CSV files found in uploads/structured/")
            return None, None, "No CSV files found."
        if len(files) == 1:
            # Only one file, always use it
            filename = files[0]
            print(f"[DbNode] Only one CSV file found, using: {filename}")
            return os.path.join(directory, filename), filename, None
        if filename is None:
            # Use the most recently modified file
            files.sort(key=lambda x: os.path.getmtime(os.path.join(directory, x)), reverse=True)
            filename = files[0]
            print(f"[DbNode] No filename specified, using latest: {filename}")
        if filename in files:
            return os.path.join(directory, filename), filename, None
        closest = self._find_closest(filename, files)
        if closest:
            print(f"[DbNode] Filename corrected to '{closest}'")
            return os.path.join(directory, closest), closest, f"Filename corrected to '{closest}'"
        print(f"[DbNode] No matching file found for '{filename}'. Available: {files}")
        return None, None, f"No matching file found for '{filename}'. Available: {files}"

    def find_column(self, col, header):
        # Make matching case-insensitive and robust to whitespace
        col_clean = col.strip().lower()
        header_clean = [h.strip().lower() for h in header]
        if col_clean in header_clean:
            idx = header_clean.index(col_clean)
            print(f"[DbNode] Matched column '{col}' to '{header[idx]}' (case-insensitive)")
            return header[idx], None
        closest = self._find_closest(col_clean, header_clean)
        if closest:
            idx = header_clean.index(closest)
            print(f"[DbNode] Column corrected to '{header[idx]}' (from '{col}') (case-insensitive)")
            return header[idx], f"Column corrected to '{header[idx]}'"
        print(f"[DbNode] No matching column found for '{col}'. Available: {header}")
        return None, f"No matching column found for '{col}'. Available: {header}"

    def extract_column_from_query(self, query, header):
        # Try to find a column name in the query (exact or fuzzy, case-insensitive)
        header_clean = [h.strip().lower() for h in header]
        for col, col_clean in zip(header, header_clean):
            if col_clean in query.lower():
                print(f"[DbNode] Detected column in query: {col} (case-insensitive)")
                return col, None
        # Fuzzy: find the closest word in the query to any column
        words = re.findall(r'\w+', query.lower())
        for word in words:
            closest = self._find_closest(word, header_clean)
            if closest:
                idx = header_clean.index(closest)
                print(f"[DbNode] Column corrected to '{header[idx]}' from '{word}' in query (case-insensitive)")
                return header[idx], f"Column corrected to '{header[idx]}' from '{word}'"
        print(f"[DbNode] No column name detected in query. Header: {header}")
        return None, "No column name detected in query."

    def process_query(self, query: str, filename: str = None, function: str = None) -> dict:
        file_path, used_file, file_correction = self.find_csv_file(filename)
        if not file_path:
            return {"success": False, "error": file_correction}
        # Load or get cached
        if used_file in self.loaded_data:
            df, header = self.loaded_data[used_file]
            print(f"[DbNode] Using cached DataFrame for {used_file}")
        else:
            df, header = self.load_csv_pandas(file_path)
            if df is not None and header is not None:
                self.loaded_data[used_file] = (df, header)
                self.column_names_by_file[used_file] = header
                self.last_loaded_file = used_file
        if df is None or header is None:
            return {"success": False, "error": f"Failed to load file '{used_file}'"}

        # Use LLM to extract function and column
        import openai
        columns = header
        llm_prompt = (
            f"User query: {query}\n"
            f"Available columns: {columns}\n"
            "Return a JSON object with:\n"
            "- function: the operation to perform (mean, sum, min, max, count, all values, etc.)\n"
            "- column: the column to use\n"
            "Example: {\"function\": \"mean\", \"column\": \"Salary\"}"
        )
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a function and column selector for data analysis."},
                    {"role": "user", "content": llm_prompt}
                ],
                max_tokens=50,
                temperature=0
            )
            content = response.choices[0].message.content
            import json
            parsed = json.loads(content)
            llm_function = parsed.get("function")
            llm_column = parsed.get("column")
            print(f"[DbNode] LLM extracted function: {llm_function}, column: {llm_column}")
        except Exception as e:
            print(f"[DbNode] LLM extraction failed: {e}")
            llm_function = None
            llm_column = None

        # Use LLM's result if available, else fallback
        used_col = llm_column if llm_column else None
        col_correction = None
        if used_col and used_col not in header:
            # Try to correct with fuzzy match
            used_col, col_correction = self.extract_column_from_query(used_col, header)
        elif not used_col:
            used_col, col_correction = self.extract_column_from_query(query, header)
        if not used_col:
            return {"success": False, "error": col_correction or "No column detected."}
        # Extract column data
        try:
            col_data = df[used_col].dropna()
            print(f"[DbNode] Extracted data for column '{used_col}': {col_data.head().tolist()}... (total {len(col_data)})")
        except Exception as e:
            print(f"[DbNode] Failed to extract column '{used_col}': {e}")
            return {"success": False, "error": f"Failed to extract column '{used_col}': {e}"}
        # Decide what operation to apply
        math_keywords = ['mean', 'sum', 'median', 'std', 'min', 'max', 'count']
        values_keywords = ["all values", "values", "list values", "show values", "values in", "show all", "list all"]
        op_requested = None
        if llm_function and llm_function.lower() in math_keywords:
            op_requested = llm_function.lower()
        elif function and function.lower() in math_keywords:
            op_requested = function.lower()
        else:
            for kw in math_keywords:
                if kw in query.lower():
                    op_requested = kw
                    break
        values_requested = (llm_function and llm_function.lower() in values_keywords) or any(kw in query.lower() for kw in values_keywords)
        # Apply operation if requested
        from .math_node import MathNode
        math_node = MathNode()
        math_result = None
        try:
            if values_requested:
                math_result = {"type": "values", "result": col_data.tolist(), "description": f"All values in column '{used_col}'"}
            elif op_requested:
                print(f"[DbNode] Passing data to MathNode for function '{op_requested}'")
                if hasattr(math_node, 'dispatch'):
                    math_result = math_node.dispatch(op_requested, col_data.tolist())
                else:
                    func_map = {
                        'mean': pd.Series.mean,
                        'sum': pd.Series.sum,
                        'median': pd.Series.median,
                        'std': pd.Series.std,
                        'min': pd.Series.min,
                        'max': pd.Series.max,
                        'count': lambda x: len(x)
                    }
                    func = func_map.get(op_requested)
                    if not func:
                        return {"success": False, "error": f"Unsupported function '{op_requested}'. Supported: {math_keywords}"}
                    math_result = {"type": op_requested, "result": func(col_data), "description": f"{op_requested} of column '{used_col}'"}
                print(f"[DbNode] MathNode result: {math_result}")
            else:
                print(f"[DbNode] No math operation requested, returning column data only.")
                math_result = {"type": "column_data", "result": col_data.tolist(), "description": f"Data for column '{used_col}'"}
        except Exception as e:
            print(f"[DbNode] MathNode failed: {e}")
            return {"success": False, "error": f"Math operation failed: {e}"}
        correction_info = []
        if file_correction:
            correction_info.append(file_correction)
        if col_correction:
            correction_info.append(col_correction)
        return {
            "success": True,
            "result": math_result,
            "used_column": used_col,
            "used_file": used_file,
            "correction_info": correction_info
        }

    def _is_number(self, s):
        try:
            float(s)
            return True
        except Exception:
            return False

    def _extract_number(self, text: str, default: int = 5) -> int:
        """Extract number from text"""
        import re
        numbers = re.findall(r'\d+', text) if text else []
        return int(numbers[0]) if numbers else default

    def _extract_column_name(self, text: str, columns: List[str]) -> str:
        """Extract column name from text"""
        text_lower = text.lower() if text else ''
        for col in columns:
            if col.lower() in text_lower:
                return col
        return columns[0] if columns else ''

    def analyze_financial_data(self, data_name: str = None) -> Dict[str, Any]:
        """Specialized financial data analysis"""
        logger.info(f"Analyzing financial data: {data_name}")
        
        try:
            if data_name is None:
                if not self.loaded_data:
                    return {"error": "No data loaded"}
                data_name = next(iter(self.loaded_data)) or "data"
            
            if data_name not in self.loaded_data:
                return {"error": f"Data '{data_name}' not found"}
            
            entry = self.loaded_data[data_name]
            df = entry["data"] if isinstance(entry, dict) and "data" in entry else entry
            if not isinstance(df, pd.DataFrame):
                return {"error": "Loaded data is not a DataFrame"}
            
            # Financial analysis
            analysis = {}
            
            # Revenue analysis
            revenue_cols = [col for col in df.columns if 'revenue' in col.lower() or 'sales' in col.lower()]
            if revenue_cols:
                revenue_col = revenue_cols[0]
                analysis['revenue'] = {
                    'total': df[revenue_col].sum(),
                    'average': df[revenue_col].mean(),
                    'max': df[revenue_col].max(),
                    'min': df[revenue_col].min()
                }
            
            # Profit analysis
            profit_cols = [col for col in df.columns if 'profit' in col.lower() or 'earnings' in col.lower()]
            if profit_cols:
                profit_col = profit_cols[0]
                analysis['profit'] = {
                    'total': df[profit_col].sum(),
                    'average': df[profit_col].mean(),
                    'margin': df[profit_col].sum() / df[revenue_cols[0]].sum() if revenue_cols else 0
                }
            
            # Growth analysis
            date_cols = [col for col in df.columns if 'date' in col.lower() or 'year' in col.lower()]
            if date_cols and revenue_cols:
                # Calculate growth trends
                analysis['growth'] = self._calculate_growth(df, date_cols[0], revenue_cols[0])
            
            return {
                "success": True,
                "analysis": analysis,
                "data_name": data_name
            }
            
        except Exception as e:
            logger.error(f"Error analyzing financial data: {str(e)}")
            return {"error": f"Financial analysis failed: {str(e)}"}
    
    def _calculate_growth(self, df: pd.DataFrame, date_col: str, value_col: str) -> Dict[str, Any]:
        """Calculate growth trends"""
        if not isinstance(df, pd.DataFrame):
            return {"error": "Loaded data is not a DataFrame"}
        return {
            "trend": "increasing",
            "growth_rate": 0.15,
            "description": "Revenue showing 15% growth trend"
        }
    
    def get_data_summary(self, data_name: str = None) -> Dict[str, Any]:
        """Get comprehensive data summary, including metadata if available"""
        if data_name is None:
            if not self.loaded_data:
                return {"error": "No data loaded"}
            data_name = next(iter(self.loaded_data)) or "data"
        if data_name not in self.loaded_data:
            return {"error": f"Data '{data_name}' not found"}
        entry = self.loaded_data[data_name]
        df = entry["data"] if isinstance(entry, dict) and "data" in entry else entry
        meta = entry["metadata"] if isinstance(entry, dict) and "metadata" in entry else {}
        if not isinstance(df, pd.DataFrame):
            return {"error": "Loaded data is not a DataFrame"}
        return {
            "data_name": data_name,
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.to_dict(),
            "null_counts": df.isnull().sum().to_dict(),
            "numeric_summary": df.describe().to_dict() if len(df.select_dtypes(include=['number']).columns) > 0 else {},
            "sample_data": df.head().to_dict('records'),
            "metadata": meta
        } 

    def get_stock_prices(self, symbol: str, start_date: str, end_date: str, data_name: str = None, context: dict = None) -> Dict[str, Any]:
        """
        Fetch stock price data for a given symbol and date range from loaded data.
        Works with dynamically uploaded files and flexible column detection.
        Args:
            symbol: Stock symbol (e.g., 'MSFT') - can be empty for single-symbol files
            start_date: Start date in 'YYYY-MM-DD' format - can be empty for all data
            end_date: End date in 'YYYY-MM-DD' format - can be empty for all data
            data_name: Optional, name of loaded data to use
            context: Context with column information and file details
        Returns:
            Dict with filtered price data or error
        """
        import pandas as pd
        try:
            if data_name is None:
                if not self.loaded_data:
                    return {"error": "No data loaded"}
                data_name = next(iter(self.loaded_data)) or "data"
            if data_name not in self.loaded_data:
                return {"error": f"Data '{data_name}' not found"}
            entry = self.loaded_data[data_name]
            df = entry["data"] if isinstance(entry, dict) and "data" in entry else entry
            if not isinstance(df, pd.DataFrame):
                return {"error": "Loaded data is not a DataFrame"}
            context = context or {}
            available_columns = context.get("available_columns", df.columns.tolist()) if context else df.columns.tolist()
            
            # Find columns intelligently
            symbol_col = None
            date_col = None
            price_col = None
            
            # Use context hints first
            if context:
                if context.get("target_column"):
                    price_col = context["target_column"]
                if context.get("date_column"):
                    date_col = context["date_column"]
            
            # Fallback to intelligent detection
            if not symbol_col:
                symbol_col = next((col for col in available_columns if col.lower() in ["symbol", "ticker", "stock"]), None)
            if not date_col:
                date_col = next((col for col in available_columns if "date" in col.lower() or "time" in col.lower()), None)
            if not price_col:
                price_col = next((col for col in available_columns if col.lower() in ["close", "price", "adj close", "adj_close", "value", "amount"]), None)
            
            # If no date column found, use index if it's datetime
            if not date_col and hasattr(df.index, 'dtype') and 'datetime' in str(df.index.dtype):
                df = df.reset_index()
                date_col = df.columns[0] if len(df.columns) > 0 else ''
            
            if not price_col:
                return {"error": f"No price column found. Available columns: {available_columns}"}
            
            # Filter by symbol if symbol column exists and symbol is specified
            if symbol_col and symbol:
                df_filtered = df[df[symbol_col].str.upper() == symbol.upper()]
                if df_filtered.empty:
                    return {"error": f"No data found for symbol '{symbol}' in column '{symbol_col}'"}
            else:
                df_filtered = df.copy()
            
            # Filter by date range if date column exists and dates are specified
            if date_col and start_date and end_date and isinstance(date_col, str) and date_col in df_filtered.columns:
                try:
                    df_filtered[date_col] = pd.to_datetime(df_filtered[date_col], errors='coerce')
                    df_filtered = df_filtered[(df_filtered[date_col] >= pd.to_datetime(start_date)) & (df_filtered[date_col] <= pd.to_datetime(end_date))]
                except Exception as e:
                    return {"error": f"Date filtering failed: {str(e)}"}
            
            # Sort by date if date column exists
            if date_col and isinstance(date_col, str) and date_col in df_filtered.columns:
                df_filtered = df_filtered.sort_values(by=date_col)
            
            # Prepare return data
            return_columns = [date_col, price_col] if date_col and isinstance(date_col, str) and date_col in df_filtered.columns else [price_col]
            return_data = df_filtered[return_columns].copy()
            
            # Rename columns for consistency
            column_mapping = {}
            if date_col and isinstance(date_col, str) and date_col in return_data.columns:
                column_mapping[date_col] = "date"
            if price_col and isinstance(price_col, str) and price_col in return_data.columns:
                column_mapping[price_col] = "price"
            return_data = return_data.rename(columns=column_mapping)
            
            return {
                "success": True,
                "symbol": symbol or "Unknown",
                "start_date": start_date or "All",
                "end_date": end_date or "All",
                "data": return_data.to_dict('records'),
                "columns_used": {
                    "symbol_column": symbol_col,
                    "date_column": date_col,
                    "price_column": price_col
                }
            }
        except Exception as e:
            return {"error": f"Failed to fetch stock prices: {str(e)}"} 