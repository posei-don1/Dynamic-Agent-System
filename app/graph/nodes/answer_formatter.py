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
            'combined_response': self._format_combined_response,
            'suggestion': self._format_suggestion,
            'error': self._format_error,
            'general': self._format_general
        }
    
    def format_response(self, response_data: Dict[str, Any], format_type: str = 'general', 
                       persona: str = None, query: str = None) -> Dict[str, Any]:
        """
        Format response data for user presentation with persona-specific enhancements
        
        Args:
            response_data: Raw response data to format
            format_type: Type of formatting to apply
            persona: Selected persona for context
            query: Original user query
            
        Returns:
            Formatted response
        """
        logger.info(f"Formatting response: {format_type} with persona: {persona}")
        
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
                persona_context = self._add_persona_context(persona, formatted_response)
                formatted_response['persona_context'] = persona_context
                
                # Add persona-specific introduction to the response
                if 'sections' in formatted_response and formatted_response['sections']:
                    # Add persona introduction as the first section
                    persona_intro = {
                        'title': f'Analysis by {persona.replace("_", " ").title()}',
                        'content': persona_context['signature'],
                        'type': 'persona_introduction',
                        'expertise': persona_context['expertise'],
                        'approach': persona_context['approach']
                    }
                    formatted_response['sections'].insert(0, persona_intro)
                
                # Update the title to reflect the persona
                if 'title' in formatted_response:
                    formatted_response['title'] = f"{formatted_response['title']} - {persona.replace('_', ' ').title()} Perspective"
            
            return formatted_response
            
        except Exception as e:
            logger.error(f"Error formatting response: {str(e)}")
            return self._format_error({"error": f"Formatting failed: {str(e)}"})
    
    def _format_data_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format data analysis results"""
        # Handle nested structure from graph builder
        if 'processed_data' in data:
            actual_data = data['processed_data']
        else:
            actual_data = data
            
        logger.info(f"Formatting data summary with actual_data: {actual_data}")
            
        if 'error' in actual_data:
            return self._format_error(actual_data)
            
        formatted = {
            'type': 'data_summary',
            'title': 'Data Analysis Results',
            'sections': []
        }
        # Dataset overview
        if 'shape' in actual_data:
            overview = {
                'title': 'Dataset Overview',
                'content': f"Dataset contains {actual_data['shape'][0]} rows and {actual_data['shape'][1]} columns",
                'details': {
                    'rows': actual_data['shape'][0],
                    'columns': actual_data['shape'][1],
                    'column_names': actual_data.get('columns', [])
                }
            }
            formatted['sections'].append(overview)
        # Query results
        if 'result' in actual_data:
            result = actual_data['result']
            # Handle 'values' type (list of values for a column)
            if isinstance(result, dict) and result.get('type') == 'values':
                values_section = {
                    'title': f"Values for column '{actual_data.get('used_column', result.get('column', ''))}",
                    'content': result.get('description', 'List of values'),
                    'values': result.get('result', []),
                    'column': actual_data.get('used_column', result.get('column', '')),
                    'type': 'values',
                    'source_file': actual_data.get('used_file', '')
                }
                formatted['sections'].append(values_section)
            # Fallback for generic column data
            elif isinstance(result, dict) and result.get('type') == 'column_data':
                col_section = {
                    'title': f"Data for column '{actual_data.get('used_column', result.get('column', ''))}",
                    'content': result.get('description', 'Column data'),
                    'values': result.get('result', []),
                    'column': actual_data.get('used_column', result.get('column', '')),
                    'type': 'column_data',
                    'source_file': actual_data.get('used_file', '')
                }
                formatted['sections'].append(col_section)
            # Handle math/stat results
            elif isinstance(result, dict):
                # Check if it has formatted_result for better display
                if 'formatted_result' in result:
                    content = result['formatted_result']
                else:
                    content = result.get('description', 'Analysis completed')
                
                result_section = {
                    'title': 'Query Results',
                    'content': content,
                    'data': result.get('result'),
                    'type': result.get('type', 'general'),
                    'raw_result': result.get('result'),
                    'formatted_result': result.get('formatted_result', ''),
                    'description': result.get('description', '')
                }
                formatted['sections'].append(result_section)
        # Financial analysis
        if 'analysis' in actual_data:
            analysis = actual_data['analysis']
            for key, value in analysis.items():
                section = {
                    'title': f'{key.title()} Analysis',
                    'content': self._format_financial_metrics(value),
                    'data': value
                }
                formatted['sections'].append(section)
        
        # Add suggestions if available from graph builder
        if 'suggestions' in data and data['suggestions']:
            suggestions_data = data['suggestions']
            if suggestions_data.get('suggestions'):
                suggestions_section = {
                    'title': 'Suggestions',
                    'content': f"Generated {len(suggestions_data['suggestions'])} follow-up questions",
                    'suggestions': suggestions_data['suggestions'],
                    'type': 'suggestions'
                }
                formatted['sections'].append(suggestions_section)
        
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
        
        # Main answer from RAG processing - THIS IS THE KEY PART!
        if 'answer' in data:
            answer_section = {
                'title': 'Answer',
                'content': data['answer'],
                'type': 'answer'
            }
            formatted['sections'].append(answer_section)
        
        # Document info
        if 'document_used' in data:
            doc_info = {
                'title': 'Document Information',
                'content': f"Analyzed document: {data['document_used']}",
                'details': {
                    'document_used': data['document_used'],
                    'relevant_chunks_found': data.get('relevant_chunks_found', 0),
                    'query': data.get('query', '')
                }
            }
            formatted['sections'].append(doc_info)
        
        # References from RAG
        if 'references' in data and data['references']:
            references_section = {
                'title': 'References',
                'content': f"Found {len(data['references'])} relevant document sections",
                'references': data['references'],
                'type': 'references'
            }
            formatted['sections'].append(references_section)
        
        # Legacy support for other document analysis types
        if 'file_path' in data and 'document_used' not in data:
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
    
    def _format_combined_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format responses that include both actual answer and suggestions"""
        if 'error' in data:
            return self._format_error(data)
        
        formatted = {
            'type': 'combined_response',
            'title': 'Analysis Results with Recommendations',
            'sections': []
        }
        
        # Extract processed data and suggestions
        processed_data = data.get("processed_data", {})
        suggestions = data.get("suggestions", {})
        
        # Add the actual answer first (most important)
        if processed_data.get("answer"):
            answer_section = {
                'title': 'Answer',
                'content': processed_data['answer'],
                'type': 'answer'
            }
            formatted['sections'].append(answer_section)
        
        # Add document information
        if processed_data.get("document_used"):
            doc_info = {
                'title': 'Document Information',
                'content': f"Analyzed document: {processed_data['document_used']}",
                'details': {
                    'document_used': processed_data['document_used'],
                    'relevant_chunks_found': processed_data.get('relevant_chunks_found', 0),
                    'query': processed_data.get('query', '')
                }
            }
            formatted['sections'].append(doc_info)
        
        # Add references from RAG
        if processed_data.get("references") and processed_data["references"]:
            references_section = {
                'title': 'References',
                'content': f"Found {len(processed_data['references'])} relevant document sections",
                'references': processed_data['references'],
                'type': 'references'
            }
            formatted['sections'].append(references_section)
        
        # Add suggestions
        if suggestions.get("suggestions"):
            suggestions_section = {
                'title': 'Recommendations',
                'content': 'Based on the analysis, here are additional recommendations:',
                'suggestions': suggestions['suggestions'],
                'type': 'suggestions'
            }
            formatted['sections'].append(suggestions_section)
        
        # Add action plan if available
        if suggestions.get("action_plan"):
            action_plan = suggestions['action_plan']
            if action_plan.get("short_term_goals"):
                action_section = {
                    'title': 'Action Plan',
                    'content': 'Recommended next steps:',
                    'actions': action_plan['short_term_goals'],
                    'type': 'action_plan'
                }
                formatted['sections'].append(action_section)
        
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
    
    def _add_persona_context(self, persona: Optional[str], formatted_response: Dict[str, Any]) -> Dict[str, Any]:
        """Add persona-specific context and signature for the answer"""
        # Ensure persona is a string and default to 'general' if None or invalid
        if not isinstance(persona, str) or not persona:
            persona = 'general'
        # Strong, explicit system prompts for each persona
        persona_prompts = {
            'legal_advisor': (
                """
                ROLE: You are a highly experienced legal advisor specializing in contracts, compliance, regulatory frameworks, and risk assessment.
                EXPERTISE: Provide clear, precise, and actionable legal advice. Reference relevant laws, statutes, or regulations by name and jurisdiction when appropriate. Use formal, professional, and unambiguous language.
                STRUCTURE: Organize your responses logically, outlining the legal reasoning behind your advice. When possible, cite specific legal precedents, articles, or sections of law to support your statements. Summarize key points at the end.
                DISCLAIMER: Always include a disclaimer that your response does not constitute formal legal counsel and recommend consulting a qualified attorney for binding advice or jurisdiction-specific matters.
                BOUNDARIES: Do NOT provide financial, business, or general advice—focus strictly on legal matters and compliance. If a question is outside your legal domain, politely state your limitations and recommend consulting a relevant expert.
                TONE: Maintain a clear, authoritative, and objective tone. If the question is ambiguous, request clarification or additional details before providing substantive advice. Stay in character as a legal advisor at all times and never speculate beyond your legal expertise.
                """
            ),
            'financial_analyst': (
                """
                ROLE: You are a professional financial analyst with expertise in financial modeling, market trends, investment strategies, and risk assessment.
                EXPERTISE: Provide detailed, data-driven, and insightful financial analysis. Use appropriate financial terminology and reference current market trends, historical data, or relevant financial models. Offer actionable insights and risk assessments where possible.
                STRUCTURE: Organize your responses with clear sections (e.g., Analysis, Insights, Recommendations). Use bullet points or tables for clarity when appropriate. Summarize key findings at the end.
                DISCLAIMER: Do NOT provide legal, business, or general advice—focus strictly on financial analysis. If a question is outside your financial domain, politely state your limitations and recommend consulting a relevant expert.
                TONE: Maintain a professional, objective, and analytical tone. If the question is ambiguous, request clarification or additional details before providing substantive analysis. Stay in character as a financial analyst at all times and never speculate beyond your financial expertise.
                """
            ),
            'general': (
                """
                ROLE: You are a helpful, knowledgeable general assistant with broad expertise across many domains.
                EXPERTISE: Answer clearly, concisely, and in a friendly, accessible tone. Provide accurate, well-structured, and easy-to-understand information. If a question requires legal or financial expertise, recommend consulting a specialist and do not attempt to provide professional advice in those areas.
                STRUCTURE: Organize your responses with clear sections or bullet points for readability. Summarize key points at the end if appropriate.
                DISCLAIMER: Avoid speculation and always strive for accuracy and clarity. If you are unsure or the question is ambiguous, ask for clarification or additional details before answering.
                TONE: Be polite, approachable, and neutral. Stay in character as a general-purpose assistant at all times.
                """
            )
        }
        # Default to general if not found
        system_prompt = persona_prompts.get(persona, persona_prompts['general'])
        # Existing context (signature, expertise, approach)
        context = {
            'signature': f"This answer is provided by a {persona.replace('_', ' ').title()}.",
            'expertise': f"Expertise: {persona.replace('_', ' ').title()} domain.",
            'approach': f"Approach: Answers are tailored to the {persona.replace('_', ' ').title()} perspective.",
            'system_prompt': system_prompt
        }
        return context
    
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