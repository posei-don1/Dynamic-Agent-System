"""
Persona Selector Node
Selects appropriate persona based on query context
"""
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class PersonaSelector:
    """Selects appropriate persona based on query analysis"""
    
    def __init__(self):
        self.personas = {
            "general": {
                "description": "General assistant with broad knowledge and analytical capabilities",
                "keywords": ["general", "help", "assist", "explain", "what", "how", "why"],
                "capabilities": ["general analysis", "information synthesis", "problem solving"]
            },
            "financial_analyst": {
                "description": "Expert in financial analysis, market trends, and investment strategies",
                "keywords": ["financial", "investment", "market", "stock", "revenue", "profit", "analysis"],
                "capabilities": ["data analysis", "financial modeling", "market research"]
            },
            "legal_advisor": {
                "description": "Legal expert specializing in contracts, compliance, and regulations",
                "keywords": ["contract", "legal", "compliance", "regulation", "law", "agreement"],
                "capabilities": ["document review", "legal analysis", "compliance checking"]
            },
            "data_scientist": {
                "description": "Expert in data analysis, machine learning, and statistical modeling",
                "keywords": ["data", "analysis", "statistics", "ml", "model", "prediction"],
                "capabilities": ["data processing", "statistical analysis", "predictive modeling"]
            },
            "business_consultant": {
                "description": "Business strategy and operations expert",
                "keywords": ["strategy", "business", "operations", "consulting", "growth", "optimization"],
                "capabilities": ["strategic planning", "process optimization", "business analysis"]
            }
        }
    
    def select_persona(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Select the most appropriate persona based on query content and context
        
        Args:
            query: User query text
            context: Additional context information
            
        Returns:
            Selected persona information
        """
        logger.info(f"Selecting persona for query: {query[:100]}...")
        
        # Check if a specific persona was requested in context
        if context and "requested_persona" in context:
            requested_persona = context["requested_persona"]
            if requested_persona in self.personas:
                logger.info(f"Using requested persona: {requested_persona}")
                return {
                    "persona": requested_persona,
                    "persona_info": self.personas[requested_persona],
                    "confidence": 1.0,
                    "selection_method": "user_requested"
                }
            else:
                logger.warning(f"Requested persona '{requested_persona}' not found, falling back to automatic selection")
        
        # Automatic persona selection based on query content
        query_lower = query.lower()
        persona_scores = {}
        
        # Score each persona based on keyword matches
        for persona_name, persona_info in self.personas.items():
            score = 0
            for keyword in persona_info["keywords"]:
                if keyword in query_lower:
                    score += 1
            persona_scores[persona_name] = score
        
        # Select persona with highest score
        selected_persona = max(persona_scores, key=persona_scores.get)
        
        # Default to general if no clear match
        if persona_scores[selected_persona] == 0:
            selected_persona = "general"
        
        logger.info(f"Auto-selected persona: {selected_persona}")
        
        return {
            "persona": selected_persona,
            "persona_info": self.personas[selected_persona],
            "confidence": persona_scores[selected_persona] / len(self.personas[selected_persona]["keywords"]),
            "selection_method": "automatic"
        }
    
    def get_persona_prompt(self, persona_name: str) -> str:
        """Get the system prompt for a specific persona"""
        if persona_name not in self.personas:
            persona_name = "business_consultant"
        
        persona_info = self.personas[persona_name]
        
        return f"""
        You are a {persona_info['description']}.
        Your capabilities include: {', '.join(persona_info['capabilities'])}.
        
        Always respond in character as this persona, providing expert insights 
        and analysis relevant to your domain of expertise.
        """ 