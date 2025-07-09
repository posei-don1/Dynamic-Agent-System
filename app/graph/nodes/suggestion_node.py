"""
Suggestion Node
Generates recommendations, advice, and actionable suggestions based on analysis
"""
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

class SuggestionNode:
    """Generates suggestions and recommendations based on analysis results"""
    
    def __init__(self):
        self.suggestion_templates = {
            'financial': self._generate_financial_suggestions,
            'business': self._generate_business_suggestions,
            'data': self._generate_data_suggestions,
            'document': self._generate_document_suggestions,
            'general': self._generate_general_suggestions
        }
        
        self.priority_levels = {
            'high': {'weight': 3, 'label': 'High Priority'},
            'medium': {'weight': 2, 'label': 'Medium Priority'},
            'low': {'weight': 1, 'label': 'Low Priority'}
        }
    
    def generate_suggestions(self, analysis_results: Dict[str, Any], 
                           suggestion_type: str = 'general', 
                           context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate suggestions based on analysis results
        
        Args:
            analysis_results: Results from previous analysis
            suggestion_type: Type of suggestions to generate
            context: Additional context information
            
        Returns:
            Generated suggestions and recommendations
        """
        logger.info(f"Generating suggestions: {suggestion_type}")
        
        try:
            # Get appropriate suggestion generator
            generator = self.suggestion_templates.get(suggestion_type, self._generate_general_suggestions)
            
            # Generate suggestions
            suggestions = generator(analysis_results, context or {})
            
            # Prioritize suggestions
            prioritized_suggestions = self._prioritize_suggestions(suggestions)
            
            # Create action plan
            action_plan = self._create_action_plan(prioritized_suggestions)
            
            return {
                'success': True,
                'suggestion_type': suggestion_type,
                'suggestions': prioritized_suggestions,
                'action_plan': action_plan,
                'total_suggestions': len(prioritized_suggestions),
                'context': context or {}
            }
            
        except Exception as e:
            logger.error(f"Error generating suggestions: {str(e)}")
            return {'error': f'Suggestion generation failed: {str(e)}'}
    
    def _generate_financial_suggestions(self, results: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate financial-specific suggestions"""
        suggestions = []
        
        # Check for financial analysis results
        if 'analysis' in results:
            analysis = results['analysis']
            
            # Revenue suggestions
            if 'revenue' in analysis:
                revenue_data = analysis['revenue']
                
                if revenue_data.get('total', 0) > 0:
                    suggestions.append({
                        'title': 'Revenue Optimization',
                        'description': f'Current total revenue is ${revenue_data["total"]:,.2f}. Consider strategies to increase revenue streams.',
                        'priority': 'high',
                        'category': 'revenue',
                        'actions': [
                            'Analyze top-performing products/services',
                            'Identify new market opportunities',
                            'Optimize pricing strategy'
                        ]
                    })
            
            # Profit margin suggestions
            if 'profit' in analysis:
                profit_data = analysis['profit']
                margin = profit_data.get('margin', 0)
                
                if margin < 0.15:  # Less than 15% margin
                    suggestions.append({
                        'title': 'Improve Profit Margins',
                        'description': f'Current profit margin is {margin:.2%}. Consider cost reduction and efficiency improvements.',
                        'priority': 'high',
                        'category': 'profitability',
                        'actions': [
                            'Review operational costs',
                            'Negotiate better supplier terms',
                            'Improve operational efficiency'
                        ]
                    })
                elif margin > 0.25:  # Good margin
                    suggestions.append({
                        'title': 'Leverage Strong Margins',
                        'description': f'Strong profit margin of {margin:.2%}. Consider expansion opportunities.',
                        'priority': 'medium',
                        'category': 'growth',
                        'actions': [
                            'Invest in marketing and sales',
                            'Explore new markets',
                            'Increase production capacity'
                        ]
                    })
            
            # Growth suggestions
            if 'growth' in analysis:
                growth_data = analysis['growth']
                
                if growth_data.get('trend') == 'increasing':
                    suggestions.append({
                        'title': 'Sustain Growth Momentum',
                        'description': 'Positive growth trend detected. Focus on maintaining and accelerating growth.',
                        'priority': 'medium',
                        'category': 'growth',
                        'actions': [
                            'Scale successful initiatives',
                            'Invest in growth enablers',
                            'Monitor key performance indicators'
                        ]
                    })
        
        # Default financial suggestions if no specific data
        if not suggestions:
            suggestions.extend([
                {
                    'title': 'Financial Health Check',
                    'description': 'Conduct comprehensive financial analysis to identify opportunities.',
                    'priority': 'medium',
                    'category': 'analysis',
                    'actions': [
                        'Review financial statements',
                        'Analyze cash flow patterns',
                        'Benchmark against industry standards'
                    ]
                }
            ])
        
        return suggestions
    
    def _generate_business_suggestions(self, results: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate business strategy suggestions"""
        suggestions = []
        
        # Data-driven suggestions
        if 'result' in results:
            result = results['result']
            
            if isinstance(result, dict) and 'data' in result:
                data = result['data']
                
                if isinstance(data, list) and len(data) > 0:
                    suggestions.append({
                        'title': 'Data-Driven Decision Making',
                        'description': f'Dataset contains {len(data)} records. Leverage this data for strategic decisions.',
                        'priority': 'high',
                        'category': 'strategy',
                        'actions': [
                            'Identify key performance indicators',
                            'Create data visualization dashboards',
                            'Implement regular data reviews'
                        ]
                    })
        
        # Process optimization suggestions
        suggestions.extend([
            {
                'title': 'Process Optimization',
                'description': 'Streamline operations to improve efficiency and reduce costs.',
                'priority': 'high',
                'category': 'operations',
                'actions': [
                    'Map current business processes',
                    'Identify bottlenecks and inefficiencies',
                    'Implement automation where possible'
                ]
            },
            {
                'title': 'Technology Integration',
                'description': 'Leverage technology to enhance business capabilities.',
                'priority': 'medium',
                'category': 'technology',
                'actions': [
                    'Assess current technology stack',
                    'Identify digital transformation opportunities',
                    'Implement scalable solutions'
                ]
            }
        ])
        
        return suggestions
    
    def _generate_data_suggestions(self, results: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate data analysis suggestions"""
        suggestions = []
        
        # Data quality suggestions
        if 'null_counts' in results:
            null_counts = results['null_counts']
            high_null_cols = [col for col, count in null_counts.items() if count > 0]
            
            if high_null_cols:
                suggestions.append({
                    'title': 'Data Quality Improvement',
                    'description': f'Found missing data in {len(high_null_cols)} columns. Address data quality issues.',
                    'priority': 'high',
                    'category': 'data_quality',
                    'actions': [
                        'Investigate sources of missing data',
                        'Implement data validation rules',
                        'Consider data imputation strategies'
                    ]
                })
        
        # Analysis suggestions
        if 'columns' in results:
            columns = results['columns']
            
            suggestions.append({
                'title': 'Advanced Analytics',
                'description': f'Dataset has {len(columns)} columns. Explore advanced analytical techniques.',
                'priority': 'medium',
                'category': 'analytics',
                'actions': [
                    'Perform correlation analysis',
                    'Apply machine learning models',
                    'Create predictive analytics'
                ]
            })
        
        # Visualization suggestions
        suggestions.append({
            'title': 'Data Visualization',
            'description': 'Create visualizations to better understand data patterns.',
            'priority': 'medium',
            'category': 'visualization',
            'actions': [
                'Create interactive dashboards',
                'Develop key metric visualizations',
                'Implement real-time monitoring'
            ]
        })
        
        return suggestions
    
    def _generate_document_suggestions(self, results: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate document analysis suggestions"""
        suggestions = []
        
        # Document processing suggestions
        if 'chunks_count' in results:
            chunks = results['chunks_count']
            
            if chunks > 10:
                suggestions.append({
                    'title': 'Document Structure Optimization',
                    'description': f'Document contains {chunks} sections. Consider improving structure for better readability.',
                    'priority': 'medium',
                    'category': 'document_management',
                    'actions': [
                        'Create clear section headers',
                        'Add executive summary',
                        'Implement consistent formatting'
                    ]
                })
        
        # Content suggestions
        if 'relevant_chunks' in results:
            relevant = results['relevant_chunks']
            
            if len(relevant) > 0:
                suggestions.append({
                    'title': 'Content Enhancement',
                    'description': f'Found {len(relevant)} relevant sections. Enhance content based on analysis.',
                    'priority': 'medium',
                    'category': 'content',
                    'actions': [
                        'Expand on key topics',
                        'Add supporting examples',
                        'Include visual aids'
                    ]
                })
        
        return suggestions
    
    def _generate_general_suggestions(self, results: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate general suggestions"""
        suggestions = [
            {
                'title': 'Regular Monitoring',
                'description': 'Implement regular monitoring and review processes.',
                'priority': 'medium',
                'category': 'monitoring',
                'actions': [
                    'Set up periodic reviews',
                    'Define key success metrics',
                    'Create alerting systems'
                ]
            },
            {
                'title': 'Continuous Improvement',
                'description': 'Establish continuous improvement processes.',
                'priority': 'medium',
                'category': 'improvement',
                'actions': [
                    'Collect feedback regularly',
                    'Identify improvement opportunities',
                    'Implement iterative changes'
                ]
            }
        ]
        
        return suggestions
    
    def _prioritize_suggestions(self, suggestions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritize suggestions based on priority levels"""
        # Add priority weights
        for suggestion in suggestions:
            priority = suggestion.get('priority', 'medium')
            suggestion['priority_weight'] = self.priority_levels[priority]['weight']
            suggestion['priority_label'] = self.priority_levels[priority]['label']
        
        # Sort by priority weight (descending)
        return sorted(suggestions, key=lambda x: x['priority_weight'], reverse=True)
    
    def _create_action_plan(self, suggestions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create an action plan from suggestions"""
        action_plan = {
            'immediate_actions': [],
            'short_term_goals': [],
            'long_term_objectives': [],
            'success_metrics': []
        }
        
        for suggestion in suggestions:
            if suggestion['priority'] == 'high':
                action_plan['immediate_actions'].extend(suggestion.get('actions', []))
            elif suggestion['priority'] == 'medium':
                action_plan['short_term_goals'].extend(suggestion.get('actions', []))
            else:
                action_plan['long_term_objectives'].extend(suggestion.get('actions', []))
        
        # Add success metrics
        action_plan['success_metrics'] = [
            'Track implementation progress',
            'Monitor key performance indicators',
            'Measure impact of changes',
            'Collect stakeholder feedback'
        ]
        
        return action_plan
    
    def create_recommendation_report(self, suggestions: List[Dict[str, Any]], 
                                   title: str = "Recommendations Report") -> Dict[str, Any]:
        """Create a comprehensive recommendation report"""
        logger.info(f"Creating recommendation report: {title}")
        
        try:
            report = {
                'title': title,
                'summary': f"Generated {len(suggestions)} recommendations based on analysis",
                'priority_breakdown': self._get_priority_breakdown(suggestions),
                'category_breakdown': self._get_category_breakdown(suggestions),
                'recommendations': suggestions,
                'implementation_timeline': self._create_implementation_timeline(suggestions)
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error creating recommendation report: {str(e)}")
            return {'error': f'Report creation failed: {str(e)}'}
    
    def _get_priority_breakdown(self, suggestions: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get breakdown of suggestions by priority"""
        breakdown = {'high': 0, 'medium': 0, 'low': 0}
        
        for suggestion in suggestions:
            priority = suggestion.get('priority', 'medium')
            breakdown[priority] += 1
        
        return breakdown
    
    def _get_category_breakdown(self, suggestions: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get breakdown of suggestions by category"""
        breakdown = {}
        
        for suggestion in suggestions:
            category = suggestion.get('category', 'general')
            breakdown[category] = breakdown.get(category, 0) + 1
        
        return breakdown
    
    def _create_implementation_timeline(self, suggestions: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Create implementation timeline"""
        timeline = {
            'week_1': [],
            'month_1': [],
            'quarter_1': [],
            'ongoing': []
        }
        
        for suggestion in suggestions:
            title = suggestion.get('title', 'Suggestion')
            
            if suggestion.get('priority') == 'high':
                timeline['week_1'].append(title)
            elif suggestion.get('priority') == 'medium':
                timeline['month_1'].append(title)
            else:
                timeline['quarter_1'].append(title)
        
        timeline['ongoing'] = ['Monitor progress', 'Collect feedback', 'Adjust strategies']
        
        return timeline 