"""
Answer Formatter Node
Formats responses in a user-friendly, structured way with proper presentation
"""
from typing import Dict, Any, List, Optional, Union
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class AnswerFormatter:
    """Formats responses for better user presentation"""
    
    def __init__(self):
        self.format_templates = {
            'data_summary': self._format_data_summary,
            'calculation': self._format_calculation,
            'document_analysis': self._format_document_analysis,
            'suggestion': self._format_suggestion,
            'error': self._format_error,
            'general': self._format_general
        }
    
    def format_response(self, response_data: Dict[str, Any], format_type: str = 'general', 
                       persona: str = None, query: str = None) -> Dict[str, Any]:
        """
        Format response data for user presentation
        
        Args:
            response_data: Raw response data to format
            format_type: Type of formatting to apply
            persona: Selected persona for context
            query: Original user query
            
        Returns:
            Formatted response
        """
        logger.info(f"Formatting response: {format_type}")
        
        try:
            # Get appropriate formatter
            formatter = self.format_templates.get(format_type, self._format_general)
            
            # Format the response
            formatted_response = formatter(response_data)
            
            # Add metadata
            formatted_response['metadata'] = {
                'format_type': format_type,
                'persona': persona,
                'query': query,
                'timestamp': datetime.now().isoformat(),
                'formatted_at': 'answer_formatter'
            }
            
            # Add persona context if available
            if persona:
                formatted_response['persona_context'] = self._add_persona_context(persona, formatted_response)
            
            return formatted_response
            
        except Exception as e:
            logger.error(f"Error formatting response: {str(e)}")
            return self._format_error({"error": f"Formatting failed: {str(e)}"})
    
    def _format_data_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format data analysis results"""
        if 'error' in data:
            return self._format_error(data)
        formatted = {
            'type': 'data_summary',
            'title': 'Data Analysis Results',
            'sections': []
        }
        # Dataset overview
        if 'shape' in data:
            overview = {
                'title': 'Dataset Overview',
                'content': f"Dataset contains {data['shape'][0]} rows and {data['shape'][1]} columns",
                'details': {
                    'rows': data['shape'][0],
                    'columns': data['shape'][1],
                    'column_names': data.get('columns', [])
                }
            }
            formatted['sections'].append(overview)
        # Query results
        if 'result' in data:
            result = data['result']
            # Handle 'values' type (list of values for a column)
            if isinstance(result, dict) and result.get('type') == 'values':
                values_section = {
                    'title': f"Values for column '{data.get('used_column', result.get('column', ''))}",
                    'content': result.get('description', 'List of values'),
                    'values': result.get('result', []),
                    'column': data.get('used_column', result.get('column', '')),
                    'type': 'values',
                    'source_file': data.get('used_file', '')
                }
                formatted['sections'].append(values_section)
            # Fallback for generic column data
            elif isinstance(result, dict) and result.get('type') == 'column_data':
                col_section = {
                    'title': f"Data for column '{data.get('used_column', result.get('column', ''))}",
                    'content': result.get('description', 'Column data'),
                    'values': result.get('result', []),
                    'column': data.get('used_column', result.get('column', '')),
                    'type': 'column_data',
                    'source_file': data.get('used_file', '')
                }
                formatted['sections'].append(col_section)
            # Handle math/stat results
            elif isinstance(result, dict):
                result_section = {
                    'title': 'Query Results',
                    'content': result.get('description', 'Analysis completed'),
                    'data': result.get('result'),
                    'type': result.get('type', 'general')
                }
                formatted['sections'].append(result_section)
        # Financial analysis
        if 'analysis' in data:
            analysis = data['analysis']
            for key, value in analysis.items():
                section = {
                    'title': f'{key.title()} Analysis',
                    'content': self._format_financial_metrics(value),
                    'data': value
                }
                formatted['sections'].append(section)
        return formatted
    
    def _format_calculation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format mathematical calculation results"""
        if 'error' in data:
            return self._format_error(data)
        
        formatted = {
            'type': 'calculation',
            'title': 'Calculation Results',
            'sections': []
        }
        
        # Main result
        if 'result' in data:
            result_section = {
                'title': 'Result',
                'content': f"Result: {data['result']}",
                'data': data['result'],
                'expression': data.get('expression', '')
            }
            formatted['sections'].append(result_section)
        
        # Calculation steps
        if 'steps' in data:
            steps_section = {
                'title': 'Calculation Steps',
                'content': 'Step-by-step calculation:',
                'steps': data['steps']
            }
            formatted['sections'].append(steps_section)
        
        # Statistical results
        if isinstance(data.get('result'), dict):
            for stat, value in data['result'].items():
                section = {
                    'title': f'{stat.replace("_", " ").title()}',
                    'content': f"{stat}: {value}",
                    'value': value
                }
                formatted['sections'].append(section)
        
        return formatted
    
    def _format_document_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format document analysis results"""
        if 'error' in data:
            return self._format_error(data)
        
        formatted = {
            'type': 'document_analysis',
            'title': 'Document Analysis Results',
            'sections': []
        }
        
        # Document info
        if 'file_path' in data:
            doc_info = {
                'title': 'Document Information',
                'content': f"Analyzed document: {data['file_path']}",
                'details': {
                    'file_path': data['file_path'],
                    'text_length': data.get('text_length', 0),
                    'chunks_count': data.get('chunks_count', 0)
                }
            }
            formatted['sections'].append(doc_info)
        
        # Summary
        if 'summary' in data:
            summary_section = {
                'title': 'Summary',
                'content': data['summary'],
                'type': 'summary'
            }
            formatted['sections'].append(summary_section)
        
        # Relevant chunks
        if 'relevant_chunks' in data:
            chunks_section = {
                'title': 'Relevant Sections',
                'content': f"Found {len(data['relevant_chunks'])} relevant sections",
                'chunks': data['relevant_chunks']
            }
            formatted['sections'].append(chunks_section)
        
        # Contract analysis
        if 'results' in data and 'analysis_type' in data:
            results = data['results']
            for key, value in results.items():
                section = {
                    'title': f'{key.replace("_", " ").title()}',
                    'content': self._format_list_content(value),
                    'data': value
                }
                formatted['sections'].append(section)
        
        return formatted
    
    def _format_suggestion(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format suggestion and recommendation results"""
        if 'error' in data:
            return self._format_error(data)
        
        formatted = {
            'type': 'suggestion',
            'title': 'Recommendations',
            'sections': []
        }
        
        # Main suggestions
        if 'suggestions' in data:
            suggestions_section = {
                'title': 'Suggestions',
                'content': 'Based on the analysis, here are the recommendations:',
                'suggestions': data['suggestions']
            }
            formatted['sections'].append(suggestions_section)
        
        # Action items
        if 'action_items' in data:
            actions_section = {
                'title': 'Action Items',
                'content': 'Recommended actions to take:',
                'actions': data['action_items']
            }
            formatted['sections'].append(actions_section)
        
        return formatted
    
    def _format_error(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format error messages"""
        return {
            'type': 'error',
            'title': 'Error',
            'sections': [{
                'title': 'Error Details',
                'content': data.get('error', 'An unknown error occurred'),
                'error_type': 'processing_error'
            }],
            'metadata': {
                'is_error': True,
                'timestamp': datetime.now().isoformat()
            }
        }
    
    def _format_general(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format general responses"""
        if 'error' in data:
            return self._format_error(data)
        
        formatted = {
            'type': 'general',
            'title': 'Response',
            'sections': []
        }
        
        # Convert data to readable format
        if isinstance(data, dict):
            for key, value in data.items():
                if key not in ['metadata', 'type', 'format_type']:
                    section = {
                        'title': key.replace('_', ' ').title(),
                        'content': self._format_value(value),
                        'data': value
                    }
                    formatted['sections'].append(section)
        else:
            formatted['sections'].append({
                'title': 'Result',
                'content': str(data),
                'data': data
            })
        
        return formatted
    
    def _format_financial_metrics(self, metrics: Dict[str, Any]) -> str:
        """Format financial metrics for display"""
        if not isinstance(metrics, dict):
            return str(metrics)
        
        formatted_lines = []
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                if key in ['total', 'average', 'max', 'min'] and value > 1000:
                    formatted_value = f"${value:,.2f}"
                elif key == 'margin':
                    formatted_value = f"{value:.2%}"
                else:
                    formatted_value = f"{value:,.2f}"
            else:
                formatted_value = str(value)
            
            formatted_lines.append(f"{key.replace('_', ' ').title()}: {formatted_value}")
        
        return "\n".join(formatted_lines)
    
    def _format_list_content(self, content: List[Any]) -> str:
        """Format list content for display"""
        if not isinstance(content, list):
            return str(content)
        
        if not content:
            return "No items found"
        
        return "\n".join(f"• {item}" for item in content)
    
    def _format_value(self, value: Any) -> str:
        """Format individual values for display"""
        if isinstance(value, dict):
            return json.dumps(value, indent=2)
        elif isinstance(value, list):
            return self._format_list_content(value)
        elif isinstance(value, (int, float)):
            return f"{value:,.2f}" if value > 1000 else str(value)
        else:
            return str(value)
    
    def _add_persona_context(self, persona: str, formatted_response: Dict[str, Any]) -> Dict[str, Any]:
        """Add persona-specific context to response"""
        persona_contexts = {
            'financial_analyst': {
                'perspective': 'Financial Analysis Perspective',
                'focus': 'This analysis focuses on financial metrics, trends, and investment implications.'
            },
            'legal_advisor': {
                'perspective': 'Legal Analysis Perspective',
                'focus': 'This analysis examines legal implications, compliance, and contractual aspects.'
            },
            'data_scientist': {
                'perspective': 'Data Science Perspective',
                'focus': 'This analysis applies statistical methods and data modeling techniques.'
            },
            'business_consultant': {
                'perspective': 'Business Strategy Perspective',
                'focus': 'This analysis considers business impact, operational efficiency, and strategic implications.'
            }
        }
        
        return persona_contexts.get(persona, {
            'perspective': 'General Analysis Perspective',
            'focus': 'This analysis provides a comprehensive view of the topic.'
        })
    
    def create_summary_report(self, responses: List[Dict[str, Any]], title: str = "Analysis Report") -> Dict[str, Any]:
        """Create a comprehensive summary report from multiple responses"""
        logger.info(f"Creating summary report: {title}")
        
        try:
            report = {
                'type': 'summary_report',
                'title': title,
                'sections': [],
                'metadata': {
                    'total_responses': len(responses),
                    'created_at': datetime.now().isoformat(),
                    'report_type': 'comprehensive'
                }
            }
            
            # Executive summary
            executive_summary = {
                'title': 'Executive Summary',
                'content': f"This report summarizes findings from {len(responses)} analyses.",
                'highlights': []
            }
            
            # Process each response
            for i, response in enumerate(responses):
                section = {
                    'title': f"Analysis {i+1}",
                    'content': self._extract_key_insights(response),
                    'source': response.get('metadata', {}).get('format_type', 'unknown')
                }
                report['sections'].append(section)
                
                # Add to highlights
                if response.get('type') != 'error':
                    executive_summary['highlights'].append(
                        self._extract_highlight(response)
                    )
            
            # Add executive summary at the beginning
            report['sections'].insert(0, executive_summary)
            
            return report
            
        except Exception as e:
            logger.error(f"Error creating summary report: {str(e)}")
            return self._format_error({"error": f"Report creation failed: {str(e)}"})
    
    def _extract_key_insights(self, response: Dict[str, Any]) -> str:
        """Extract key insights from a response"""
        if 'sections' in response:
            insights = []
            for section in response['sections']:
                if section.get('title') != 'Error Details':
                    insights.append(f"{section['title']}: {section.get('content', '')[:100]}...")
            return "\n".join(insights)
        return "No key insights available"
    
    def _extract_highlight(self, response: Dict[str, Any]) -> str:
        """Extract a highlight from a response"""
        if 'sections' in response and response['sections']:
            first_section = response['sections'][0]
            return f"{first_section.get('title', 'Finding')}: {first_section.get('content', '')[:50]}..."
        return "Analysis completed" 

def format_llm_answer(llm_json: dict) -> str:
    """
    Format the LLM's JSON answer (with sources, page/line info) into a beautiful chat message.
    Args:
        llm_json: dict with keys 'answer' and 'sources' (list of dicts with 'page', 'lines', 'text')
    Returns:
        str: Markdown/HTML formatted string for chat display
    """
    if not llm_json or 'answer' not in llm_json:
        return "No answer found."
    answer = llm_json['answer']
    sources = llm_json.get('sources', [])
    msg = f"<div><b>Answer:</b><br>{answer}</div>"
    if sources:
        msg += "<div style='margin-top: 1em;'><b>Source Details:</b><ul>"
        for src in sources:
            page = str(src.get('page') if src.get('page') is not None else '?')
            lines = src.get('lines', [])
            text = str(src.get('text') if src.get('text') is not None else '')
            lines_str = f"Lines {lines[0]}-{lines[1]}" if lines and len(lines) == 2 else ""
            msg += f"<li><b>Page:</b> {page} {lines_str}<br><pre style='background:#f6f8fa;padding:0.5em;border-radius:4px;'>{text}</pre></li>"
        msg += "</ul></div>"
    return msg 