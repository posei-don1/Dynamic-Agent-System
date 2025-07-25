"""
Math Node
Handles mathematical calculations, formula evaluation, and statistical computations
"""
from typing import Dict, Any, List, Optional, Union
import logging
import re
import math
import statistics
from decimal import Decimal, getcontext

logger = logging.getLogger(__name__)

class MathNode:
    """Processes mathematical queries and calculations"""
    
    def __init__(self):
        # Set decimal precision for accurate calculations
        getcontext().prec = 28
        
        self.supported_operations = {
            'add': ['+', 'plus', 'add', 'sum'],
            'subtract': ['-', 'minus', 'subtract', 'difference'],
            'multiply': ['*', 'times', 'multiply', 'product'],
            'divide': ['/', 'divided by', 'divide', 'quotient'],
            'power': ['^', '**', 'power', 'exponent'],
            'sqrt': ['sqrt', 'square root'],
            'percentage': ['%', 'percent', 'percentage'],
            'average': ['average', 'mean', 'avg'],
            'median': ['median'],
            'mode': ['mode'],
            'std': ['standard deviation', 'std dev', 'stddev']
        }
    
    def calculate(self, expression: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Calculate mathematical expressions
        
        Args:
            expression: Mathematical expression or query
            context: Additional context information
            
        Returns:
            Calculation results
        """
        logger.info(f"Calculating: {expression}")
        
        try:
            # Clean and normalize the expression
            cleaned_expr = self._clean_expression(expression)
            
            # Identify calculation type
            calc_type = self._identify_calculation_type(cleaned_expr)
            
            # Perform calculation based on type
            if calc_type == 'basic_arithmetic':
                result = self._calculate_basic_arithmetic(cleaned_expr)
            elif calc_type == 'statistical':
                result = self._calculate_statistics(cleaned_expr)
            elif calc_type == 'percentage':
                result = self._calculate_percentage(cleaned_expr)
            elif calc_type == 'formula':
                result = self._evaluate_formula(cleaned_expr)
            else:
                result = self._calculate_basic_arithmetic(cleaned_expr)
            
            return {
                "success": True,
                "expression": expression,
                "result": result,
                "calculation_type": calc_type,
                "steps": self._get_calculation_steps(cleaned_expr, result)
            }
            
        except Exception as e:
            logger.error(f"Calculation error: {str(e)}")
            return {"error": f"Calculation failed: {str(e)}"}
    
    def _clean_expression(self, expression: str) -> str:
        """Clean and normalize mathematical expression"""
        # Remove extra spaces
        cleaned = re.sub(r'\s+', ' ', expression.strip())
        
        # Replace common text representations with symbols
        replacements = {
            'plus': '+',
            'add': '+',
            'minus': '-',
            'subtract': '-',
            'times': '*',
            'multiply': '*',
            'divided by': '/',
            'divide': '/',
            'power': '**',
            'squared': '**2',
            'cubed': '**3'
        }
        
        for word, symbol in replacements.items():
            cleaned = cleaned.replace(word, symbol)
        
        return cleaned
    
    def _identify_calculation_type(self, expression: str) -> str:
        """Identify the type of calculation needed"""
        expr_lower = expression.lower()
        
        if any(stat in expr_lower for stat in ['average', 'mean', 'median', 'mode', 'std', 'standard deviation']):
            return 'statistical'
        elif any(perc in expr_lower for perc in ['percent', 'percentage', '%']):
            return 'percentage'
        elif any(func in expr_lower for func in ['sin', 'cos', 'tan', 'log', 'ln', 'sqrt']):
            return 'formula'
        else:
            return 'basic_arithmetic'
    
    def _calculate_basic_arithmetic(self, expression: str) -> Union[float, int, str]:
        """Calculate basic arithmetic expressions"""
        try:
            # Extract numbers and operations
            numbers = self._extract_numbers(expression)
            
            if not numbers:
                return "No numbers found in expression"
            
            # For simple expressions, try direct evaluation
            if self._is_safe_expression(expression):
                result = eval(expression)
                return round(result, 10) if isinstance(result, float) else result
            
            # For complex expressions, parse step by step
            return self._parse_complex_expression(expression)
            
        except Exception as e:
            return f"Arithmetic calculation error: {str(e)}"
    
    def _calculate_statistics(self, expression: str) -> Dict[str, Any]:
        """Calculate statistical measures"""
        numbers = self._extract_numbers(expression)
        
        if not numbers:
            return {"error": "No numbers found for statistical calculation"}
        
        stats_result = {}
        expr_lower = expression.lower()
        
        if 'average' in expr_lower or 'mean' in expr_lower:
            stats_result['mean'] = statistics.mean(numbers)
        
        if 'median' in expr_lower:
            stats_result['median'] = statistics.median(numbers)
        
        if 'mode' in expr_lower:
            try:
                stats_result['mode'] = statistics.mode(numbers)
            except statistics.StatisticsError:
                stats_result['mode'] = "No unique mode found"
        
        if 'std' in expr_lower or 'standard deviation' in expr_lower:
            if len(numbers) > 1:
                stats_result['std_dev'] = statistics.stdev(numbers)
            else:
                stats_result['std_dev'] = "Need at least 2 values for standard deviation"
        
        # If no specific stat requested, calculate all
        if not stats_result:
            stats_result = {
                'mean': statistics.mean(numbers),
                'median': statistics.median(numbers),
                'min': min(numbers),
                'max': max(numbers),
                'count': len(numbers)
            }
            
            if len(numbers) > 1:
                stats_result['std_dev'] = statistics.stdev(numbers)
        
        return stats_result
    
    def _calculate_percentage(self, expression: str) -> Dict[str, Any]:
        """Calculate percentage operations"""
        numbers = self._extract_numbers(expression)
        
        if len(numbers) < 2:
            return {"error": "Need at least 2 numbers for percentage calculation"}
        
        expr_lower = expression.lower()
        
        if 'of' in expr_lower:
            # e.g., "20% of 100" or "what is 20% of 100"
            percentage = numbers[0]
            total = numbers[1]
            result = (percentage / 100) * total
            return {
                "calculation": f"{percentage}% of {total}",
                "result": result,
                "explanation": f"{percentage}% of {total} = {result}"
            }
        
        elif 'increase' in expr_lower or 'decrease' in expr_lower:
            # e.g., "100 increased by 20%" or "100 decreased by 20%"
            base = numbers[0]
            percentage = numbers[1]
            
            if 'increase' in expr_lower:
                result = base * (1 + percentage / 100)
                operation = "increased"
            else:
                result = base * (1 - percentage / 100)
                operation = "decreased"
            
            return {
                "calculation": f"{base} {operation} by {percentage}%",
                "result": result,
                "explanation": f"{base} {operation} by {percentage}% = {result}"
            }
        
        else:
            # Default: calculate what percentage the first number is of the second
            part = numbers[0]
            total = numbers[1]
            percentage = (part / total) * 100
            
            return {
                "calculation": f"{part} as percentage of {total}",
                "result": f"{percentage}%",
                "explanation": f"{part} is {percentage}% of {total}"
            }
    
    def _evaluate_formula(self, expression: str) -> Union[float, str]:
        """Evaluate mathematical formulas with functions"""
        try:
            # Replace common mathematical functions
            expr = expression.lower()
            
            # Handle square root
            if 'sqrt' in expr:
                numbers = self._extract_numbers(expression)
                if numbers:
                    return math.sqrt(numbers[0])
            
            # Handle trigonometric functions
            if any(func in expr for func in ['sin', 'cos', 'tan']):
                numbers = self._extract_numbers(expression)
                if numbers:
                    angle = numbers[0]
                    if 'sin' in expr:
                        return math.sin(math.radians(angle))
                    elif 'cos' in expr:
                        return math.cos(math.radians(angle))
                    elif 'tan' in expr:
                        return math.tan(math.radians(angle))
            
            # Handle logarithmic functions
            if 'log' in expr or 'ln' in expr:
                numbers = self._extract_numbers(expression)
                if numbers:
                    if 'ln' in expr:
                        return math.log(numbers[0])
                    else:
                        return math.log10(numbers[0])
            
            # Default to basic arithmetic
            return self._calculate_basic_arithmetic(expression)
            
        except Exception as e:
            return f"Formula evaluation error: {str(e)}"
    
    def _extract_numbers(self, text: str) -> List[float]:
        """Extract all numbers from text"""
        # Pattern to match integers, floats, and scientific notation
        pattern = r'-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?'
        matches = re.findall(pattern, text)
        return [float(match) for match in matches]
    
    def _is_safe_expression(self, expression: str) -> bool:
        """Check if expression is safe for eval"""
        # Only allow numbers, basic operators, and parentheses
        safe_chars = set('0123456789+-*/.()^ ')
        return all(c in safe_chars for c in expression)
    
    def _parse_complex_expression(self, expression: str) -> str:
        """Parse complex mathematical expressions step by step"""
        # Placeholder for complex expression parsing
        # In a real implementation, you might use a proper math parser
        return f"Complex expression parsing not implemented for: {expression}"
    
    def _get_calculation_steps(self, expression: str, result: Any) -> List[str]:
        """Generate step-by-step calculation explanation"""
        steps = []
        
        if isinstance(result, dict) and 'explanation' in result:
            steps.append(result['explanation'])
        else:
            steps.append(f"Expression: {expression}")
            steps.append(f"Result: {result}")
        
        return steps
    
    def solve_equation(self, equation: str) -> Dict[str, Any]:
        """Solve simple equations"""
        logger.info(f"Solving equation: {equation}")
        
        try:
            # Simple equation solver placeholder
            # In a real implementation, you might use sympy
            return {
                "success": True,
                "equation": equation,
                "solution": "Equation solving not implemented yet",
                "steps": ["This is a placeholder for equation solving"]
            }
            
        except Exception as e:
            logger.error(f"Equation solving error: {str(e)}")
            return {"error": f"Equation solving failed: {str(e)}"}
    
    def financial_calculation(self, calc_type: str, **kwargs) -> Dict[str, Any]:
        """Perform financial calculations"""
        logger.info(f"Financial calculation: {calc_type}")
        
        try:
            if calc_type == 'compound_interest':
                principal = kwargs.get('principal', 0)
                rate = kwargs.get('rate', 0) / 100  # Convert percentage to decimal
                time = kwargs.get('time', 0)
                compound_freq = kwargs.get('compound_frequency', 1)
                
                amount = principal * (1 + rate/compound_freq) ** (compound_freq * time)
                interest = amount - principal
                
                return {
                    "success": True,
                    "calculation_type": "compound_interest",
                    "principal": principal,
                    "rate": kwargs.get('rate', 0),
                    "time": time,
                    "final_amount": round(amount, 2),
                    "interest_earned": round(interest, 2)
                }
            
            elif calc_type == 'simple_interest':
                principal = kwargs.get('principal', 0)
                rate = kwargs.get('rate', 0) / 100
                time = kwargs.get('time', 0)
                
                interest = principal * rate * time
                amount = principal + interest
                
                return {
                    "success": True,
                    "calculation_type": "simple_interest",
                    "principal": principal,
                    "rate": kwargs.get('rate', 0),
                    "time": time,
                    "final_amount": round(amount, 2),
                    "interest_earned": round(interest, 2)
                }
            
            else:
                return {"error": f"Unsupported financial calculation type: {calc_type}"}
                
        except Exception as e:
            logger.error(f"Financial calculation error: {str(e)}")
            return {"error": f"Financial calculation failed: {str(e)}"} 

    def moving_average(self, data: list, window: int = 5) -> dict:
        """
        Compute moving average for a list of price dicts (with 'date' and 'price' keys).
        Pure Python calculation - no LLM involved.
        Args:
            data: List of dicts with 'date' and 'price' keys
            window: Window size for moving average
        Returns:
            Dict with moving average results
        """
        import pandas as pd
        try:
            if not data or not isinstance(data, list):
                return {"error": "No data provided for moving average calculation"}
            df = pd.DataFrame(data)
            if 'price' not in df.columns:
                return {"error": "Data must contain 'price' column"}
            
            # Sort by date if date column exists
            if 'date' in df.columns:
                df = df.sort_values('date')
            
            # Calculate moving average using pandas rolling window
            df['moving_average'] = df['price'].rolling(window=window, min_periods=1).mean()
            
            # Prepare result
            result_columns = ['moving_average']
            if 'date' in df.columns:
                result_columns = ['date'] + result_columns
            
            result = df[result_columns].to_dict('records')
            return {
                "success": True, 
                "window": window, 
                "moving_average": result,
                "calculation_type": "simple_moving_average"
            }
        except Exception as e:
            return {"error": f"Failed to compute moving average: {str(e)}"}
    
    def exponential_moving_average(self, data: list, span: int = 12) -> dict:
        """
        Compute exponential moving average - pure Python calculation.
        Args:
            data: List of dicts with 'date' and 'price' keys
            span: Span for EMA calculation
        Returns:
            Dict with EMA results
        """
        import pandas as pd
        try:
            if not data or not isinstance(data, list):
                return {"error": "No data provided for EMA calculation"}
            df = pd.DataFrame(data)
            if 'price' not in df.columns:
                return {"error": "Data must contain 'price' column"}
            
            if 'date' in df.columns:
                df = df.sort_values('date')
            
            # Calculate EMA using pandas
            df['ema'] = df['price'].ewm(span=span).mean()
            
            result_columns = ['ema']
            if 'date' in df.columns:
                result_columns = ['date'] + result_columns
            
            result = df[result_columns].to_dict('records')
            return {
                "success": True,
                "span": span,
                "ema": result,
                "calculation_type": "exponential_moving_average"
            }
        except Exception as e:
            return {"error": f"Failed to compute EMA: {str(e)}"}
    
    def bollinger_bands(self, data: list, window: int = 20, num_std: float = 2.0) -> dict:
        """
        Compute Bollinger Bands - pure Python calculation.
        Args:
            data: List of dicts with 'date' and 'price' keys
            window: Window size for moving average
            num_std: Number of standard deviations for bands
        Returns:
            Dict with Bollinger Bands results
        """
        import pandas as pd
        try:
            if not data or not isinstance(data, list):
                return {"error": "No data provided for Bollinger Bands calculation"}
            df = pd.DataFrame(data)
            if 'price' not in df.columns:
                return {"error": "Data must contain 'price' column"}
            
            if 'date' in df.columns:
                df = df.sort_values('date')
            
            # Calculate Bollinger Bands
            df['sma'] = df['price'].rolling(window=window).mean()
            df['std'] = df['price'].rolling(window=window).std()
            df['upper_band'] = df['sma'] + (df['std'] * num_std)
            df['lower_band'] = df['sma'] - (df['std'] * num_std)
            
            result_columns = ['sma', 'upper_band', 'lower_band']
            if 'date' in df.columns:
                result_columns = ['date'] + result_columns
            
            result = df[result_columns].to_dict('records')
            return {
                "success": True,
                "window": window,
                "num_std": num_std,
                "bollinger_bands": result,
                "calculation_type": "bollinger_bands"
            }
        except Exception as e:
            return {"error": f"Failed to compute Bollinger Bands: {str(e)}"}
    
    def rsi(self, data: list, window: int = 14) -> dict:
        """
        Compute Relative Strength Index (RSI) - pure Python calculation.
        Args:
            data: List of dicts with 'date' and 'price' keys
            window: Window size for RSI calculation
        Returns:
            Dict with RSI results
        """
        import pandas as pd
        try:
            if not data or not isinstance(data, list):
                return {"error": "No data provided for RSI calculation"}
            df = pd.DataFrame(data)
            if 'price' not in df.columns:
                return {"error": "Data must contain 'price' column"}
            
            if 'date' in df.columns:
                df = df.sort_values('date')
            
            # Calculate RSI
            delta = df['price'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            result_columns = ['rsi']
            if 'date' in df.columns:
                result_columns = ['date'] + result_columns
            
            result = df[result_columns].to_dict('records')
            return {
                "success": True,
                "window": window,
                "rsi": result,
                "calculation_type": "relative_strength_index"
            }
        except Exception as e:
            return {"error": f"Failed to compute RSI: {str(e)}"}
    
    def calculate_financial_metric(self, data: list, metric_type: str, **kwargs) -> dict:
        """
        Calculate financial metrics using pure Python functions as tools.
        Args:
            data: List of dicts with price data
            metric_type: Type of calculation to perform
            **kwargs: Additional parameters for the calculation
        Returns:
            Dict with calculation results
        """
        try:
            if metric_type == "moving_average":
                return self.moving_average(data, kwargs.get("window", 20))
            elif metric_type == "exponential_moving_average":
                return self.exponential_moving_average(data, kwargs.get("span", 12))
            elif metric_type == "bollinger_bands":
                return self.bollinger_bands(data, kwargs.get("window", 20), kwargs.get("num_std", 2.0))
            elif metric_type == "rsi":
                return self.rsi(data, kwargs.get("window", 14))
            else:
                return {"error": f"Unsupported metric type: {metric_type}"}
        except Exception as e:
            return {"error": f"Financial metric calculation failed: {str(e)}"} 

    def mean(self, data: list) -> float:
        import statistics
        return statistics.mean(data) if data else 0.0

    def median(self, data: list) -> float:
        import statistics
        return statistics.median(data) if data else 0.0

    def std(self, data: list) -> float:
        import statistics
        return statistics.stdev(data) if len(data) > 1 else 0.0

    def sum(self, data: list) -> float:
        return sum(data) if data else 0.0

    def min(self, data: list) -> float:
        return min(data) if data else 0.0

    def max(self, data: list) -> float:
        return max(data) if data else 0.0

    def mode(self, data: list) -> float:
        import statistics
        try:
            return statistics.mode(data)
        except Exception:
            return 0.0

    def count(self, data: list) -> int:
        return len(data)

    def describe(self, data: list) -> dict:
        import statistics
        if not data:
            return {}
        return {
            'count': len(data),
            'mean': statistics.mean(data),
            'std': statistics.stdev(data) if len(data) > 1 else 0.0,
            'min': min(data),
            '25%': statistics.quantiles(data, n=4)[0] if len(data) > 3 else None,
            '50%': statistics.median(data),
            '75%': statistics.quantiles(data, n=4)[-1] if len(data) > 3 else None,
            'max': max(data)
        }

    def head(self, data: list, n: int = 5) -> list:
        import pandas as pd
        if not data:
            return []
        df = pd.DataFrame(data)
        return df.head(n).to_dict('records')  # type: ignore

    def tail(self, data: list, n: int = 5) -> list:
        import pandas as pd
        if not data:
            return []
        df = pd.DataFrame(data)
        return df.tail(n).to_dict('records')  # type: ignore

    def sample(self, data: list, n: int = 5) -> list:
        import pandas as pd
        if not data:
            return []
        df = pd.DataFrame(data)
        return df.sample(n=min(n, len(df))).to_dict('records')  # type: ignore

    def shape(self, data: list) -> tuple:
        import pandas as pd
        if not data:
            return (0, 0)
        df = pd.DataFrame(data)
        return df.shape

    def columns(self, data: list) -> list:
        import pandas as pd
        if not data:
            return []
        df = pd.DataFrame(data)
        return list(df.columns)

    def info(self, data: list) -> dict:
        import pandas as pd
        import io
        if not data:
            return {"info": "No data loaded."}
        df = pd.DataFrame(data)
        buf = io.StringIO()
        df.info(buf=buf)
        info_str = buf.getvalue()
        return {"info": info_str}

    def dispatch(self, func_name: str, data: list, **kwargs) -> any:
        """Dynamically call the math/stat or DataFrame function by name."""
        func_map = {
            'mean': self.mean,
            'average': self.mean,
            'median': self.median,
            'std': self.std,
            'stdev': self.std,
            'sum': self.sum,
            'min': self.min,
            'max': self.max,
            'mode': self.mode,
            'count': self.count,
            'describe': self.describe,
            'head': self.head,
            'tail': self.tail,
            'sample': self.sample,
            'shape': self.shape,
            'columns': self.columns,
            'info': self.info
        }
        func = func_map.get(func_name.lower())
        if func:
            # Pass n if needed for head/tail/sample
            if func_name.lower() in ['head', 'tail', 'sample']:
                n = kwargs.get('n', 5)
                return func(data, n)
            return func(data)
        return None 