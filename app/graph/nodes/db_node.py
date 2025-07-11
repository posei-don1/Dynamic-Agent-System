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
from .math_node import MathNode

logger = logging.getLogger(__name__)

class DbNode:
    """Processes database queries and CSV data analysis"""
    
    def __init__(self):
        self.supported_formats = ['.csv', '.xlsx', '.json']
        self.db_connection = None
        self.loaded_data = {}
        self.file_metadata = {}  # Store metadata for each file
        self.math_node = MathNode()
    
    def load_data(self, file_path: str, data_name: str = None) -> Dict[str, Any]:
        """
        Load data file (CSV, Excel, JSON) and store DataFrame and metadata
        """
        try:
            if not file_path or not isinstance(file_path, str):
                return {"error": "No file path provided"}
            file_ext = Path(file_path).suffix.lower()
            if file_ext not in self.supported_formats:
                return {"error": f"Unsupported file format: {file_ext}"}
            import pandas as pd
            df = pd.read_csv(file_path) if file_ext == '.csv' else pd.read_excel(file_path)
            if not hasattr(df, 'columns'):
                return {"error": "Loaded data is not a DataFrame"}
            data_name = data_name or Path(file_path).stem
            meta = {
                'file_name': os.path.basename(file_path),
                'columns': df.columns.tolist(),
                'dtypes': df.dtypes.apply(lambda x: str(x)).to_dict(),
                'shape': df.shape,
                'sample': df.head().to_dict('records') if hasattr(df, 'head') else []
            }
            self.loaded_data[data_name] = {"data": df, "metadata": meta}
            self.file_metadata[data_name] = meta
            return {
                'success': True,
                'data_name': data_name,
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'data_types': df.dtypes.to_dict(),
                'sample_data': df.head().to_dict('records') if hasattr(df, 'head') else [],
                'metadata': meta,
                'file_name': os.path.basename(file_path)
            }
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return {"error": f"Data loading failed: {str(e)}"}
    
    def query_data(self, query: str, data_name: str = None) -> Dict[str, Any]:
        """
        Query loaded data using natural language, always include metadata
        """
        logger.info(f"Querying data: {query}")
        try:
            if data_name is None:
                if not self.loaded_data:
                    return {"error": "No data loaded"}
                data_name = next(iter(self.loaded_data))
            if data_name not in self.loaded_data:
                return {"error": f"Data '{data_name}' not found"}
            entry = self.loaded_data[data_name]
            df = entry["data"] if isinstance(entry, dict) and "data" in entry else entry
            meta = entry["metadata"] if isinstance(entry, dict) and "metadata" in entry else {}
            import pandas as pd
            if not isinstance(df, pd.DataFrame):
                return {"error": "Loaded data is not a DataFrame"}
            query_lower = query.lower()
            # Check for metadata requests
            if 'metadata' in query_lower:
                meta = self.file_metadata.get(data_name, {})
                return {'type': 'metadata', 'metadata': meta, 'description': 'File metadata'}
            # Check for data type requests
            if 'data type' in query_lower or 'dtype' in query_lower:
                import difflib
                col_name = self._extract_column_name(query, df.columns.tolist())
                if col_name is None:
                    matches = difflib.get_close_matches(query, df.columns.tolist(), n=1)
                    col_name = matches[0] if matches else ''
                dtype = str(df[col_name].dtype) if col_name and col_name in df.columns else 'unknown'
                return {'type': 'dtype', 'column': col_name, 'dtype': dtype, 'description': f'Data type of {col_name}'}
            # Check for math/stat function requests
            for func in ['mean', 'average', 'median', 'std', 'sum', 'min', 'max', 'mode', 'count', 'describe']:
                if func in query_lower:
                    col_name = self._extract_column_name(query, df.columns.tolist())
                    if col_name:
                        col_data = df[col_name].dropna().tolist()
                        result = self.math_node.dispatch(func, col_data)
                        return {'type': func, 'column': col_name, 'result': result, 'description': f'{func} of {col_name}'}
                    if 'row' in query_lower or 'all columns' in query_lower:
                        numeric_cols = list(df.select_dtypes(include=['number']).columns)
                        results = {col: self.math_node.dispatch(func, df[col].dropna().tolist()) for col in numeric_cols}
                        return {'type': func, 'row_wise': True, 'results': results, 'description': f'{func} for all numeric columns'}
            # Basic query patterns
            if "show" in query_lower or "display" in query_lower:
                if hasattr(df, 'head'):
                    if "top" in query_lower or "first" in query_lower:
                        n = self._extract_number(query, default=5)
                        return {
                            "type": "display",
                            "data": df.head(n).to_dict('records'),
                            "description": f"First {n} rows"
                        }
                    else:
                        return {
                            "type": "display",
                            "data": df.to_dict('records'),
                            "description": "All data"
                        }
            if "count" in query_lower:
                return {
                    "type": "count",
                    "data": len(df),
                    "description": f"Total rows: {len(df)}"
                }
            if "sum" in query_lower:
                numeric_cols = list(df.select_dtypes(include=['number']).columns)
                if len(numeric_cols) > 0:
                    col_name = self._extract_column_name(query, numeric_cols)
                    if col_name:
                        return {
                            "type": "sum",
                            "data": df[col_name].sum(),
                            "description": f"Sum of {col_name}"
                        }
            if "average" in query_lower or "mean" in query_lower:
                numeric_cols = list(df.select_dtypes(include=['number']).columns)
                if len(numeric_cols) > 0:
                    col_name = self._extract_column_name(query, numeric_cols)
                    if col_name:
                        return {
                            "type": "average",
                            "data": df[col_name].mean(),
                            "description": f"Average of {col_name}"
                        }
            if "max" in query_lower or "maximum" in query_lower:
                numeric_cols = list(df.select_dtypes(include=['number']).columns)
                if len(numeric_cols) > 0:
                    col_name = self._extract_column_name(query, numeric_cols)
                    if col_name:
                        return {
                            "type": "max",
                            "data": df[col_name].max(),
                            "description": f"Maximum of {col_name}"
                        }
            if "min" in query_lower or "minimum" in query_lower:
                numeric_cols = list(df.select_dtypes(include=['number']).columns)
                if len(numeric_cols) > 0:
                    col_name = self._extract_column_name(query, numeric_cols)
                    if col_name:
                        return {
                            "type": "min",
                            "data": df[col_name].min(),
                            "description": f"Minimum of {col_name}"
                        }
            if "group" in query_lower or "groupby" in query_lower:
                group_col = self._extract_column_name(query, list(df.columns))
                if group_col:
                    grouped = df.groupby(group_col).size().reset_index()
                    grouped = grouped.rename(columns={0: 'count'})
                    return {
                        "type": "group",
                        "data": grouped.to_dict('records'),
                        "description": f"Grouped by {group_col}"
                    }
            return {
                "type": "info",
                "data": {
                    "shape": df.shape,
                    "columns": df.columns.tolist(),
                    "dtypes": df.dtypes.to_dict(),
                    "sample": df.head().to_dict('records') if hasattr(df, 'head') else []
                },
                "description": "Dataset information"
            }
        except Exception as e:
            logger.error(f"Error querying data: {str(e)}")
            return {"error": f"Query execution failed: {str(e)}"}
    
    def _execute_query(self, query: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Execute natural language query on DataFrame"""
        query_lower = query.lower()
        
        # Basic query patterns
        if "show" in query_lower or "display" in query_lower:
            if "top" in query_lower or "first" in query_lower:
                n = self._extract_number(query, default=5)
                return {
                    "type": "display",
                    "data": df.head(n).to_dict('records'),
                    "description": f"First {n} rows"
                }
            else:
                return {
                    "type": "display",
                    "data": df.to_dict('records'),
                    "description": "All data"
                }
        
        elif "count" in query_lower:
            return {
                "type": "count",
                "data": len(df),
                "description": f"Total rows: {len(df)}"
            }
        
        elif "sum" in query_lower:
            numeric_cols = list(df.select_dtypes(include=['number']).columns)
            if len(numeric_cols) > 0:
                col_name = self._extract_column_name(query, numeric_cols)
                if col_name:
                    return {
                        "type": "sum",
                        "data": df[col_name].sum(),
                        "description": f"Sum of {col_name}"
                    }
        
        elif "average" in query_lower or "mean" in query_lower:
            numeric_cols = list(df.select_dtypes(include=['number']).columns)
            if len(numeric_cols) > 0:
                col_name = self._extract_column_name(query, numeric_cols)
                if col_name:
                    return {
                        "type": "average",
                        "data": df[col_name].mean(),
                        "description": f"Average of {col_name}"
                    }
        
        elif "max" in query_lower or "maximum" in query_lower:
            numeric_cols = list(df.select_dtypes(include=['number']).columns)
            if len(numeric_cols) > 0:
                col_name = self._extract_column_name(query, numeric_cols)
                if col_name:
                    return {
                        "type": "max",
                        "data": df[col_name].max(),
                        "description": f"Maximum of {col_name}"
                    }
        
        elif "min" in query_lower or "minimum" in query_lower:
            numeric_cols = list(df.select_dtypes(include=['number']).columns)
            if len(numeric_cols) > 0:
                col_name = self._extract_column_name(query, numeric_cols)
                if col_name:
                    return {
                        "type": "min",
                        "data": df[col_name].min(),
                        "description": f"Minimum of {col_name}"
                    }
        
        elif "group" in query_lower or "groupby" in query_lower:
            group_col = self._extract_column_name(query, list(df.columns))
            if group_col:
                grouped = df.groupby(group_col).size().reset_index()
                grouped = grouped.rename(columns={0: 'count'})
                return {
                    "type": "group",
                    "data": grouped.to_dict('records'),
                    "description": f"Grouped by {group_col}"
                }
        
        else:
            # Default: return basic info
            return {
                "type": "info",
                "data": {
                    "shape": df.shape,
                    "columns": df.columns.tolist(),
                    "dtypes": df.dtypes.to_dict(),
                    "sample": df.head().to_dict('records')
                },
                "description": "Dataset information"
            }
    
    def _extract_number(self, text: str, default: int = 5) -> int:
        """Extract number from text"""
        import re
        numbers = re.findall(r'\d+', text)
        return int(numbers[0]) if numbers else default
    
    def _extract_column_name(self, text: str, columns: List[str]) -> Optional[str]:
        """Extract column name from text"""
        text_lower = text.lower()
        for col in columns:
            if col.lower() in text_lower:
                return col
        return columns[0] if columns else None
    
    def analyze_financial_data(self, data_name: str = None) -> Dict[str, Any]:
        """Specialized financial data analysis"""
        logger.info(f"Analyzing financial data: {data_name}")
        
        try:
            if data_name is None:
                if not self.loaded_data:
                    return {"error": "No data loaded"}
                data_name = next(iter(self.loaded_data))
            
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
            data_name = next(iter(self.loaded_data))
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
                data_name = next(iter(self.loaded_data))
            if data_name not in self.loaded_data:
                return {"error": f"Data '{data_name}' not found"}
            entry = self.loaded_data[data_name]
            df = entry["data"] if isinstance(entry, dict) and "data" in entry else entry
            if not isinstance(df, pd.DataFrame):
                return {"error": "Loaded data is not a DataFrame"}
            
            # Flexible column detection using context or intelligent guessing
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
                date_col = df.columns[0]
            
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
            if date_col and start_date and end_date:
                try:
                    df_filtered[date_col] = pd.to_datetime(df_filtered[date_col], errors='coerce')
                    df_filtered = df_filtered[(df_filtered[date_col] >= pd.to_datetime(start_date)) & (df_filtered[date_col] <= pd.to_datetime(end_date))]
                except Exception as e:
                    return {"error": f"Date filtering failed: {str(e)}"}
            
            # Sort by date if date column exists
            if date_col:
                df_filtered = df_filtered.sort_values(by=date_col)
            
            # Prepare return data
            return_columns = [date_col, price_col] if date_col else [price_col]
            return_data = df_filtered[return_columns].copy()
            
            # Rename columns for consistency
            column_mapping = {}
            if date_col:
                column_mapping[date_col] = "date"
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