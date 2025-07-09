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

logger = logging.getLogger(__name__)

class DbNode:
    """Processes database queries and CSV data analysis"""
    
    def __init__(self):
        self.supported_formats = ['.csv', '.xlsx', '.json']
        self.db_connection = None
        self.loaded_data = {}
    
    def load_data(self, file_path: str, data_name: str = None) -> Dict[str, Any]:
        """
        Load data from file into memory
        
        Args:
            file_path: Path to the data file
            data_name: Name to reference the loaded data
            
        Returns:
            Loading results
        """
        logger.info(f"Loading data from: {file_path}")
        
        try:
            if not os.path.exists(file_path):
                return {"error": f"File not found: {file_path}"}
            
            file_ext = Path(file_path).suffix.lower()
            if file_ext not in self.supported_formats:
                return {"error": f"Unsupported file format: {file_ext}"}
            
            # Load data based on file type
            if file_ext == '.csv':
                df = pd.read_csv(file_path)
            elif file_ext == '.xlsx':
                df = pd.read_excel(file_path)
            elif file_ext == '.json':
                df = pd.read_json(file_path)
            else:
                return {"error": f"Unsupported format: {file_ext}"}
            
            # Store loaded data
            data_name = data_name or Path(file_path).stem
            self.loaded_data[data_name] = df
            
            return {
                "success": True,
                "data_name": data_name,
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "data_types": df.dtypes.to_dict(),
                "sample_data": df.head().to_dict('records')
            }
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return {"error": f"Data loading failed: {str(e)}"}
    
    def query_data(self, query: str, data_name: str = None) -> Dict[str, Any]:
        """
        Query loaded data using natural language
        
        Args:
            query: Natural language query
            data_name: Name of the data to query
            
        Returns:
            Query results
        """
        logger.info(f"Querying data: {query}")
        
        try:
            # If no data_name specified, use the first loaded dataset
            if data_name is None:
                if not self.loaded_data:
                    return {"error": "No data loaded"}
                data_name = next(iter(self.loaded_data))
            
            if data_name not in self.loaded_data:
                return {"error": f"Data '{data_name}' not found"}
            
            df = self.loaded_data[data_name]
            
            # Parse query and execute
            result = self._execute_query(query, df)
            
            return {
                "success": True,
                "query": query,
                "data_name": data_name,
                "result": result
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
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                col_name = self._extract_column_name(query, numeric_cols)
                if col_name:
                    return {
                        "type": "sum",
                        "data": df[col_name].sum(),
                        "description": f"Sum of {col_name}"
                    }
        
        elif "average" in query_lower or "mean" in query_lower:
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                col_name = self._extract_column_name(query, numeric_cols)
                if col_name:
                    return {
                        "type": "average",
                        "data": df[col_name].mean(),
                        "description": f"Average of {col_name}"
                    }
        
        elif "max" in query_lower or "maximum" in query_lower:
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                col_name = self._extract_column_name(query, numeric_cols)
                if col_name:
                    return {
                        "type": "max",
                        "data": df[col_name].max(),
                        "description": f"Maximum of {col_name}"
                    }
        
        elif "min" in query_lower or "minimum" in query_lower:
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                col_name = self._extract_column_name(query, numeric_cols)
                if col_name:
                    return {
                        "type": "min",
                        "data": df[col_name].min(),
                        "description": f"Minimum of {col_name}"
                    }
        
        elif "group" in query_lower or "groupby" in query_lower:
            group_col = self._extract_column_name(query, df.columns)
            if group_col:
                grouped = df.groupby(group_col).size().reset_index(name='count')
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
            
            df = self.loaded_data[data_name]
            
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
        # Placeholder for growth calculation
        return {
            "trend": "increasing",
            "growth_rate": 0.15,
            "description": "Revenue showing 15% growth trend"
        }
    
    def get_data_summary(self, data_name: str = None) -> Dict[str, Any]:
        """Get comprehensive data summary"""
        if data_name is None:
            if not self.loaded_data:
                return {"error": "No data loaded"}
            data_name = next(iter(self.loaded_data))
        
        if data_name not in self.loaded_data:
            return {"error": f"Data '{data_name}' not found"}
        
        df = self.loaded_data[data_name]
        
        return {
            "data_name": data_name,
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.to_dict(),
            "null_counts": df.isnull().sum().to_dict(),
            "numeric_summary": df.describe().to_dict() if len(df.select_dtypes(include=['number']).columns) > 0 else {},
            "sample_data": df.head().to_dict('records')
        } 