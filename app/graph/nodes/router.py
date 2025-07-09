"""
Router Node
Routes queries to appropriate processing nodes based on content analysis
"""
from typing import Dict, Any, List
import logging
import re

logger = logging.getLogger(__name__)

class Router:
    """Routes queries to appropriate processing nodes"""
    
    def __init__(self):
        self.route_patterns = {
            "doc_node": {
                "keywords": ["document", "pdf", "contract", "agreement", "review", "analyze document"],
                "patterns": [r".*analyze.*document.*", r".*review.*pdf.*", r".*contract.*terms.*"]
            },
            "db_node": {
                "keywords": ["database", "table", "sql", "query", "data", "csv", "records"],
                "patterns": [r".*query.*data.*", r".*from.*table.*", r".*csv.*analysis.*"]
            },
            "math_node": {
                "keywords": ["calculate", "math", "formula", "compute", "equation", "statistics"],
                "patterns": [r".*calculate.*", r".*what.*is.*\d+.*", r".*formula.*for.*"]
            },
            "suggestion_node": {
                "keywords": ["suggest", "recommend", "advice", "what should", "how to"],
                "patterns": [r".*suggest.*", r".*recommend.*", r".*what.*should.*", r".*how.*to.*"]
            }
        }
    
    def route_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Route query to appropriate processing node
        
        Args:
            query: User query text
            context: Additional context information
            
        Returns:
            Routing decision with node and confidence
        """
        logger.info(f"Routing query: {query[:100]}...")
        
        query_lower = query.lower()
        node_scores = {}
        
        # Score each node based on keyword matches and patterns
        for node_name, node_info in self.route_patterns.items():
            score = 0
            
            # Check keywords
            for keyword in node_info["keywords"]:
                if keyword in query_lower:
                    score += 1
            
            # Check patterns
            for pattern in node_info["patterns"]:
                if re.search(pattern, query_lower):
                    score += 2  # Patterns have higher weight
            
            node_scores[node_name] = score
        
        # Select node with highest score
        if max(node_scores.values()) == 0:
            # Default to db_node for general queries
            selected_node = "db_node"
            confidence = 0.5
        else:
            selected_node = max(node_scores, key=node_scores.get)
            max_possible_score = len(self.route_patterns[selected_node]["keywords"]) + \
                              len(self.route_patterns[selected_node]["patterns"]) * 2
            confidence = node_scores[selected_node] / max_possible_score
        
        logger.info(f"Routed to: {selected_node} (confidence: {confidence:.2f})")
        
        return {
            "node": selected_node,
            "confidence": confidence,
            "scores": node_scores
        }
    
    def get_processing_strategy(self, node: str, query: str) -> Dict[str, Any]:
        """Get processing strategy for the selected node"""
        strategies = {
            "doc_node": {
                "approach": "document_analysis",
                "steps": ["extract_text", "chunk_content", "semantic_search", "summarize"]
            },
            "db_node": {
                "approach": "data_query",
                "steps": ["parse_query", "generate_sql", "execute_query", "format_results"]
            },
            "math_node": {
                "approach": "calculation",
                "steps": ["extract_numbers", "identify_operation", "calculate", "format_result"]
            },
            "suggestion_node": {
                "approach": "recommendation",
                "steps": ["analyze_context", "generate_options", "rank_suggestions", "format_advice"]
            }
        }
        
        return strategies.get(node, strategies["db_node"]) 