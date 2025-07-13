"""
Router Node
Routes queries to appropriate processing nodes based on LLM-powered content analysis
"""
from typing import Dict, Any, List
import logging
import re
import openai

logger = logging.getLogger(__name__)

class Router:
    """Routes queries to appropriate processing nodes"""
    
    def __init__(self):
        self.route_patterns = {
            "doc_node": {
                "keywords": ["document", "pdf", "contract", "agreement", "review", "analyze document", "clause", "terms"],
                "patterns": [r".*analyze.*document.*", r".*review.*pdf.*", r".*contract.*terms.*", r".*clause.*"]
            },
            "db_node": {
                "keywords": ["database", "table", "sql", "query", "data", "csv", "records", "show", "display"],
                "patterns": [r".*query.*data.*", r".*from.*table.*", r".*csv.*analysis.*", r".*show.*data.*"]
            },
            "math_node": {
                "keywords": ["calculate", "math", "formula", "compute", "equation", "statistics", "moving average", "ma", "stock", "price", "msft", "aapl", "financial", "average", "trend", "analysis"],
                "patterns": [r".*calculate.*", r".*what.*is.*\d+.*", r".*formula.*for.*", r".*moving.*average.*", r".*\b(ma|MA)\b.*", r".*stock.*price.*", r".*from.*\d{4}.*to.*\d{4}.*", r".*march.*may.*", r".*\b(msft|MSFT|aapl|AAPL|googl|GOOGL)\b.*"]
            },
            "suggestion_node": {
                "keywords": ["suggest", "recommend", "advice", "what should", "how to"],
                "patterns": [r".*suggest.*", r".*recommend.*", r".*what.*should.*", r".*how.*to.*"]
            }
        }
    
    def route_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Route query to appropriate processing node with structured/unstructured classification
        
        Args:
            query: User query text
            context: Additional context information
            
        Returns:
            Routing decision with node, confidence, and query type
        """
        logger.info(f"Routing query: {query[:100]}...")
        
        # First, classify as structured or unstructured
        query_type = self._classify_query_type(query)
        logger.info(f"Query classified as: {query_type}")
        
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
        
        # Use LLM to select the best processing node
        selected_node, confidence = self._llm_select_node(query, query_type, node_scores)
        
        logger.info(f"Routed to: {selected_node} (confidence: {confidence:.2f}) for {query_type} query")
        
        return {
            "node": selected_node,
            "confidence": confidence,
            "query_type": query_type,
            "scores": node_scores
        }
    
    def _classify_query_type(self, query: str) -> str:
        """
        Classify query as structured or unstructured using LLM
        
        Args:
            query: User query text
            
        Returns:
            Query type: 'structured' or 'unstructured'
        """
        try:
            # Create LLM classification prompt
            classification_prompt = f"""
            Classify this question as either 'structured' or 'unstructured':
            
            - 'structured': Questions about tables, columns, numbers, data analysis, CSV files, Excel spreadsheets, 
              statistical calculations (mean, sum, median, etc.), or requests to show/display specific data values
            
            - 'unstructured': Questions about document content, text analysis, explanations, interpretations, 
              general knowledge, concepts, or requests to analyze/explain content from PDFs, documents, or text
            
            Question: "{query}"
            
            Answer with only 'structured' or 'unstructured'.
            """
            
            # Use OpenAI to classify the query
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a query classifier. Only respond with 'structured' or 'unstructured'."},
                    {"role": "user", "content": classification_prompt}
                ],
                max_tokens=10,
                temperature=0.1
            )
            
            classification = response.choices[0].message.content.strip().lower()
            
            # Validate the response
            if classification in ['structured', 'unstructured']:
                logger.info(f"LLM classified query as: {classification}")
                return classification
            else:
                logger.warning(f"Invalid LLM classification: {classification}, defaulting to structured")
                return "structured"
                
        except Exception as e:
            logger.error(f"LLM classification failed: {str(e)}, using fallback classification")
            return self._fallback_classify_query_type(query)
    
    def _fallback_classify_query_type(self, query: str) -> str:
        """
        Fallback classification using keyword matching when LLM fails
        
        Args:
            query: User query text
            
        Returns:
            Query type: 'structured' or 'unstructured'
        """
        query_lower = query.lower()
        
        # Structured query indicators
        structured_indicators = [
            "mean", "average", "sum", "median", "std", "min", "max", "count",
            "column", "row", "table", "data", "csv", "excel", "spreadsheet",
            "calculate", "compute", "statistics", "numbers", "values",
            "show", "display", "list", "find", "get", "retrieve"
        ]
        
        # Unstructured query indicators
        unstructured_indicators = [
            "analyze", "explain", "describe", "what is", "how does", "why",
            "document", "pdf", "text", "content", "meaning", "interpret",
            "understand", "context", "background", "information", "details"
        ]
        
        structured_score = sum(1 for indicator in structured_indicators if indicator in query_lower)
        unstructured_score = sum(1 for indicator in unstructured_indicators if indicator in query_lower)
        
        # Determine query type
        if structured_score > unstructured_score:
            return "structured"
        elif unstructured_score > structured_score:
            return "unstructured"
        else:
            # If scores are equal, check for specific patterns
            if any(word in query_lower for word in ["mean", "sum", "average", "column", "data"]):
                return "structured"
            elif any(word in query_lower for word in ["analyze", "explain", "what", "how", "why"]):
                return "unstructured"
            else:
                # Default to structured for ambiguous queries
                return "structured"
    
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
    
    def _llm_select_node(self, query: str, query_type: str, node_scores: Dict[str, int]) -> tuple[str, float]:
        """
        Use LLM to select the best processing node
        
        Args:
            query: User query text
            query_type: 'structured' or 'unstructured'
            node_scores: Scores for each node from keyword matching
            
        Returns:
            Tuple of (selected_node, confidence)
        """
        try:
            # Create LLM node selection prompt
            node_selection_prompt = f"""
            Select the best processing node for this query:
            
            Query: "{query}"
            Query Type: {query_type}
            
            Available nodes and their purposes:
            - db_node: For data queries, CSV analysis, statistical calculations (mean, sum, etc.), column operations, database queries
            - math_node: For mathematical calculations, financial analysis, complex formulas, moving averages, numerical computations
            - doc_node: For document analysis, PDF processing, text content analysis, RAG-based Q&A, questions about uploaded documents, content explanations
            - suggestion_node: For generating follow-up questions and conversation continuations AFTER getting an answer from other nodes
            
            IMPORTANT: The suggestion_node is ONLY for generating follow-up questions after processing. For actual content questions, use doc_node.
            
            Node scores from keyword matching: {node_scores}
            
            Based on the query and query type, select the most appropriate node.
            Answer with only the node name: 'db_node', 'math_node', 'doc_node', or 'suggestion_node'.
            """
            
            # Use OpenAI to select the node
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a node selector. Only respond with the node name: db_node, math_node, doc_node, or suggestion_node."},
                    {"role": "user", "content": node_selection_prompt}
                ],
                max_tokens=15,
                temperature=0.1
            )
            
            selected_node = response.choices[0].message.content.strip().lower()
            
            # Validate the response
            valid_nodes = ['db_node', 'math_node', 'doc_node', 'suggestion_node']
            if selected_node in valid_nodes:
                # Calculate confidence based on node scores and LLM selection
                max_score = max(node_scores.values()) if node_scores else 0
                selected_score = node_scores.get(selected_node, 0)
                confidence = min(0.9, (selected_score / max_score) if max_score > 0 else 0.7)
                
                logger.info(f"LLM selected node: {selected_node} with confidence: {confidence:.2f}")
                return selected_node, confidence
            else:
                logger.warning(f"Invalid LLM node selection: {selected_node}, using fallback")
                return self._fallback_select_node(query_type, node_scores)
                
        except Exception as e:
            logger.error(f"LLM node selection failed: {str(e)}, using fallback")
            return self._fallback_select_node(query_type, node_scores)
    
    def _fallback_select_node(self, query_type: str, node_scores: Dict[str, int]) -> tuple[str, float]:
        """
        Fallback node selection when LLM fails
        
        Args:
            query_type: 'structured' or 'unstructured'
            node_scores: Scores for each node
            
        Returns:
            Tuple of (selected_node, confidence)
        """
        if query_type == "structured":
            # For structured queries, prefer db_node or math_node
            if node_scores.get("math_node", 0) > node_scores.get("db_node", 0):
                selected_node = "math_node"
            else:
                selected_node = "db_node"
            confidence = max(node_scores.get("math_node", 0), node_scores.get("db_node", 0)) / 10
        elif query_type == "unstructured":
            # For unstructured queries, prefer doc_node or suggestion_node
            if node_scores.get("doc_node", 0) > node_scores.get("suggestion_node", 0):
                selected_node = "doc_node"
            else:
                selected_node = "suggestion_node"
            confidence = max(node_scores.get("doc_node", 0), node_scores.get("suggestion_node", 0)) / 10
        else:
            # Fallback based on highest score
            if max(node_scores.values()) == 0:
                selected_node = "db_node"
                confidence = 0.5
            else:
                selected_node = max(node_scores, key=node_scores.get)
                max_possible_score = len(self.route_patterns[selected_node]["keywords"]) + \
                                  len(self.route_patterns[selected_node]["patterns"]) * 2
                confidence = node_scores[selected_node] / max_possible_score
        
        return selected_node, confidence 