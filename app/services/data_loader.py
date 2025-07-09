"""
Data Loader Service
Handles CSV file loading, database connections, and data processing
"""
from typing import Dict, Any, List, Optional, Union
import logging
import pandas as pd
import os
from pathlib import Path
import sqlite3
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class DataLoader:
    """Handles data loading from various sources"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.supported_formats = ['.csv', '.xlsx', '.json', '.parquet']
        self.data_cache = {}
        self.db_connections = {}
        
        # Default configuration
        self.default_csv_params = {
            'encoding': 'utf-8',
            'na_values': ['', 'NULL', 'null', 'NA', 'na', 'N/A', 'n/a'],
            'keep_default_na': True
        }
    
    def load_csv(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """
        Load CSV file into DataFrame
        
        Args:
            file_path: Path to CSV file
            **kwargs: Additional parameters for pandas.read_csv
            
        Returns:
            Loading result with data and metadata
        """
        logger.info(f"Loading CSV file: {file_path}")
        
        try:
            if not os.path.exists(file_path):
                return {"error": f"File not found: {file_path}"}
            
            # Merge default parameters with provided kwargs
            csv_params = {**self.default_csv_params, **kwargs}
            
            # Load CSV file
            df = pd.read_csv(file_path, **csv_params)
            
            # Generate data summary
            data_summary = self._generate_data_summary(df, file_path)
            
            # Cache the data
            cache_key = Path(file_path).stem
            self.data_cache[cache_key] = df
            
            return {
                'success': True,
                'data': df,
                'data_name': cache_key,
                'file_path': file_path,
                'summary': data_summary
            }
            
        except Exception as e:
            logger.error(f"Error loading CSV: {str(e)}")
            return {"error": f"CSV loading failed: {str(e)}"}
    
    def load_excel(self, file_path: str, sheet_name: str = None, **kwargs) -> Dict[str, Any]:
        """
        Load Excel file into DataFrame
        
        Args:
            file_path: Path to Excel file
            sheet_name: Name of sheet to load (default: first sheet)
            **kwargs: Additional parameters for pandas.read_excel
            
        Returns:
            Loading result with data and metadata
        """
        logger.info(f"Loading Excel file: {file_path}")
        
        try:
            if not os.path.exists(file_path):
                return {"error": f"File not found: {file_path}"}
            
            # Load Excel file
            if sheet_name:
                df = pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
            else:
                df = pd.read_excel(file_path, **kwargs)
            
            # Generate data summary
            data_summary = self._generate_data_summary(df, file_path)
            
            # Cache the data
            cache_key = f"{Path(file_path).stem}_{sheet_name or 'sheet1'}"
            self.data_cache[cache_key] = df
            
            return {
                'success': True,
                'data': df,
                'data_name': cache_key,
                'file_path': file_path,
                'sheet_name': sheet_name,
                'summary': data_summary
            }
            
        except Exception as e:
            logger.error(f"Error loading Excel: {str(e)}")
            return {"error": f"Excel loading failed: {str(e)}"}
    
    def load_json(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """
        Load JSON file into DataFrame
        
        Args:
            file_path: Path to JSON file
            **kwargs: Additional parameters for pandas.read_json
            
        Returns:
            Loading result with data and metadata
        """
        logger.info(f"Loading JSON file: {file_path}")
        
        try:
            if not os.path.exists(file_path):
                return {"error": f"File not found: {file_path}"}
            
            # Load JSON file
            df = pd.read_json(file_path, **kwargs)
            
            # Generate data summary
            data_summary = self._generate_data_summary(df, file_path)
            
            # Cache the data
            cache_key = Path(file_path).stem
            self.data_cache[cache_key] = df
            
            return {
                'success': True,
                'data': df,
                'data_name': cache_key,
                'file_path': file_path,
                'summary': data_summary
            }
            
        except Exception as e:
            logger.error(f"Error loading JSON: {str(e)}")
            return {"error": f"JSON loading failed: {str(e)}"}
    
    def load_data_file(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """
        Load data file (auto-detect format)
        
        Args:
            file_path: Path to data file
            **kwargs: Additional parameters for loading
            
        Returns:
            Loading result with data and metadata
        """
        logger.info(f"Loading data file: {file_path}")
        
        try:
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext == '.csv':
                return self.load_csv(file_path, **kwargs)
            elif file_ext in ['.xlsx', '.xls']:
                return self.load_excel(file_path, **kwargs)
            elif file_ext == '.json':
                return self.load_json(file_path, **kwargs)
            elif file_ext == '.parquet':
                return self.load_parquet(file_path, **kwargs)
            else:
                return {"error": f"Unsupported file format: {file_ext}"}
                
        except Exception as e:
            logger.error(f"Error loading data file: {str(e)}")
            return {"error": f"Data file loading failed: {str(e)}"}
    
    def load_parquet(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """
        Load Parquet file into DataFrame
        
        Args:
            file_path: Path to Parquet file
            **kwargs: Additional parameters for pandas.read_parquet
            
        Returns:
            Loading result with data and metadata
        """
        logger.info(f"Loading Parquet file: {file_path}")
        
        try:
            if not os.path.exists(file_path):
                return {"error": f"File not found: {file_path}"}
            
            # Load Parquet file
            df = pd.read_parquet(file_path, **kwargs)
            
            # Generate data summary
            data_summary = self._generate_data_summary(df, file_path)
            
            # Cache the data
            cache_key = Path(file_path).stem
            self.data_cache[cache_key] = df
            
            return {
                'success': True,
                'data': df,
                'data_name': cache_key,
                'file_path': file_path,
                'summary': data_summary
            }
            
        except Exception as e:
            logger.error(f"Error loading Parquet: {str(e)}")
            return {"error": f"Parquet loading failed: {str(e)}"}
    
    def connect_to_database(self, connection_string: str, db_type: str = 'sqlite') -> Dict[str, Any]:
        """
        Connect to database
        
        Args:
            connection_string: Database connection string
            db_type: Type of database (sqlite, postgresql, mysql)
            
        Returns:
            Connection result
        """
        logger.info(f"Connecting to {db_type} database")
        
        try:
            if db_type == 'sqlite':
                conn = sqlite3.connect(connection_string)
            else:
                # For other databases, would need appropriate connectors
                return {"error": f"Database type '{db_type}' not implemented"}
            
            # Store connection
            self.db_connections[db_type] = conn
            
            return {
                'success': True,
                'db_type': db_type,
                'connection_string': connection_string
            }
            
        except Exception as e:
            logger.error(f"Error connecting to database: {str(e)}")
            return {"error": f"Database connection failed: {str(e)}"}
    
    def query_database(self, query: str, db_type: str = 'sqlite') -> Dict[str, Any]:
        """
        Execute SQL query on database
        
        Args:
            query: SQL query string
            db_type: Type of database to query
            
        Returns:
            Query result
        """
        logger.info(f"Executing database query: {query[:100]}...")
        
        try:
            if db_type not in self.db_connections:
                return {"error": f"No connection to {db_type} database"}
            
            conn = self.db_connections[db_type]
            
            # Execute query
            df = pd.read_sql_query(query, conn)
            
            # Generate data summary
            data_summary = self._generate_data_summary(df, f"query_{db_type}")
            
            return {
                'success': True,
                'data': df,
                'query': query,
                'db_type': db_type,
                'summary': data_summary
            }
            
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            return {"error": f"Database query failed: {str(e)}"}
    
    def _generate_data_summary(self, df: pd.DataFrame, source: str) -> Dict[str, Any]:
        """Generate comprehensive data summary"""
        try:
            # Basic info
            summary = {
                'source': source,
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'dtypes': df.dtypes.to_dict(),
                'memory_usage': df.memory_usage(deep=True).sum(),
                'created_at': datetime.now().isoformat()
            }
            
            # Missing values
            summary['missing_values'] = df.isnull().sum().to_dict()
            summary['missing_percentage'] = (df.isnull().sum() / len(df) * 100).to_dict()
            
            # Numeric columns summary
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                summary['numeric_summary'] = df[numeric_cols].describe().to_dict()
            
            # Categorical columns summary
            categorical_cols = df.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                summary['categorical_summary'] = {}
                for col in categorical_cols:
                    summary['categorical_summary'][col] = {
                        'unique_values': df[col].nunique(),
                        'top_values': df[col].value_counts().head().to_dict()
                    }
            
            # Date columns
            date_cols = df.select_dtypes(include=['datetime']).columns
            if len(date_cols) > 0:
                summary['date_summary'] = {}
                for col in date_cols:
                    summary['date_summary'][col] = {
                        'min_date': df[col].min(),
                        'max_date': df[col].max(),
                        'date_range': (df[col].max() - df[col].min()).days
                    }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating data summary: {str(e)}")
            return {'error': f'Summary generation failed: {str(e)}'}
    
    def get_cached_data(self, data_name: str) -> Optional[pd.DataFrame]:
        """Get cached data by name"""
        return self.data_cache.get(data_name)
    
    def list_cached_data(self) -> List[str]:
        """List all cached data names"""
        return list(self.data_cache.keys())
    
    def clear_cache(self, data_name: str = None) -> Dict[str, Any]:
        """Clear data cache"""
        try:
            if data_name:
                if data_name in self.data_cache:
                    del self.data_cache[data_name]
                    return {'success': True, 'message': f'Cleared cache for {data_name}'}
                else:
                    return {'error': f'Data {data_name} not found in cache'}
            else:
                self.data_cache.clear()
                return {'success': True, 'message': 'Cleared all cache'}
                
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            return {'error': f'Cache clearing failed: {str(e)}'}
    
    def export_data(self, data_name: str, output_path: str, format: str = 'csv') -> Dict[str, Any]:
        """
        Export cached data to file
        
        Args:
            data_name: Name of cached data
            output_path: Output file path
            format: Export format (csv, excel, json, parquet)
            
        Returns:
            Export result
        """
        logger.info(f"Exporting data {data_name} to {output_path}")
        
        try:
            if data_name not in self.data_cache:
                return {'error': f'Data {data_name} not found in cache'}
            
            df = self.data_cache[data_name]
            
            if format.lower() == 'csv':
                df.to_csv(output_path, index=False)
            elif format.lower() == 'excel':
                df.to_excel(output_path, index=False)
            elif format.lower() == 'json':
                df.to_json(output_path, orient='records', indent=2)
            elif format.lower() == 'parquet':
                df.to_parquet(output_path, index=False)
            else:
                return {'error': f'Unsupported export format: {format}'}
            
            return {
                'success': True,
                'output_path': output_path,
                'format': format,
                'rows_exported': len(df)
            }
            
        except Exception as e:
            logger.error(f"Error exporting data: {str(e)}")
            return {'error': f'Data export failed: {str(e)}'}
    
    def merge_datasets(self, left_data: str, right_data: str, 
                      on: Union[str, List[str]], how: str = 'inner') -> Dict[str, Any]:
        """
        Merge two cached datasets
        
        Args:
            left_data: Name of left dataset
            right_data: Name of right dataset
            on: Column(s) to merge on
            how: Type of merge (inner, left, right, outer)
            
        Returns:
            Merge result
        """
        logger.info(f"Merging datasets: {left_data} and {right_data}")
        
        try:
            if left_data not in self.data_cache:
                return {'error': f'Left dataset {left_data} not found in cache'}
            
            if right_data not in self.data_cache:
                return {'error': f'Right dataset {right_data} not found in cache'}
            
            left_df = self.data_cache[left_data]
            right_df = self.data_cache[right_data]
            
            # Perform merge
            merged_df = pd.merge(left_df, right_df, on=on, how=how)
            
            # Cache merged result
            merge_name = f"{left_data}_{right_data}_merged"
            self.data_cache[merge_name] = merged_df
            
            # Generate summary
            summary = self._generate_data_summary(merged_df, f"merge_{left_data}_{right_data}")
            
            return {
                'success': True,
                'merged_data_name': merge_name,
                'left_shape': left_df.shape,
                'right_shape': right_df.shape,
                'merged_shape': merged_df.shape,
                'merge_type': how,
                'merge_columns': on,
                'summary': summary
            }
            
        except Exception as e:
            logger.error(f"Error merging datasets: {str(e)}")
            return {'error': f'Dataset merge failed: {str(e)}'}
    
    def filter_data(self, data_name: str, filter_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter cached data based on conditions
        
        Args:
            data_name: Name of cached data
            filter_conditions: Dictionary of filter conditions
            
        Returns:
            Filter result
        """
        logger.info(f"Filtering data: {data_name}")
        
        try:
            if data_name not in self.data_cache:
                return {'error': f'Data {data_name} not found in cache'}
            
            df = self.data_cache[data_name]
            filtered_df = df.copy()
            
            # Apply filters
            for column, condition in filter_conditions.items():
                if column not in df.columns:
                    return {'error': f'Column {column} not found in data'}
                
                if isinstance(condition, dict):
                    # Handle complex conditions
                    if 'operator' in condition and 'value' in condition:
                        operator = condition['operator']
                        value = condition['value']
                        
                        if operator == 'eq':
                            filtered_df = filtered_df[filtered_df[column] == value]
                        elif operator == 'ne':
                            filtered_df = filtered_df[filtered_df[column] != value]
                        elif operator == 'gt':
                            filtered_df = filtered_df[filtered_df[column] > value]
                        elif operator == 'lt':
                            filtered_df = filtered_df[filtered_df[column] < value]
                        elif operator == 'gte':
                            filtered_df = filtered_df[filtered_df[column] >= value]
                        elif operator == 'lte':
                            filtered_df = filtered_df[filtered_df[column] <= value]
                        elif operator == 'in':
                            filtered_df = filtered_df[filtered_df[column].isin(value)]
                        elif operator == 'contains':
                            filtered_df = filtered_df[filtered_df[column].str.contains(value, na=False)]
                else:
                    # Simple equality filter
                    filtered_df = filtered_df[filtered_df[column] == condition]
            
            # Cache filtered result
            filtered_name = f"{data_name}_filtered"
            self.data_cache[filtered_name] = filtered_df
            
            return {
                'success': True,
                'filtered_data_name': filtered_name,
                'original_shape': df.shape,
                'filtered_shape': filtered_df.shape,
                'rows_removed': len(df) - len(filtered_df),
                'filter_conditions': filter_conditions
            }
            
        except Exception as e:
            logger.error(f"Error filtering data: {str(e)}")
            return {'error': f'Data filtering failed: {str(e)}'}
    
    def close_database_connections(self):
        """Close all database connections"""
        try:
            for db_type, conn in self.db_connections.items():
                conn.close()
                logger.info(f"Closed {db_type} database connection")
            
            self.db_connections.clear()
            
        except Exception as e:
            logger.error(f"Error closing database connections: {str(e)}")
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.close_database_connections() 