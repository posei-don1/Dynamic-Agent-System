"""
Suggestion Node
Generates follow-up questions based on the answer content to help users continue the conversation
"""
from typing import Dict, Any, List, Optional
import logging
import openai

logger = logging.getLogger(__name__)

class SuggestionNode:
    """Generates follow-up questions based on answer content using OpenAI"""
    
    def __init__(self):
        self.system_prompt = """You are an expert at generating follow-up questions that help users continue meaningful conversations.

Your task is to analyze the given answer text and generate 2-4 follow-up questions that are:
1. **Directly related** to the specific content and topics mentioned in the provided answer
2. **Based on the actual text** - questions should reference specific concepts, examples, or details from the answer
3. **Functional and actionable** - questions that lead to deeper understanding of the topics discussed
4. **Varied in depth** - mix of basic understanding and advanced exploration
5. **Natural conversation flow** - questions that feel like a natural continuation of the discussion

CRITICAL REQUIREMENTS:
- Generate questions ONLY from the content provided in the answer text
- Do NOT generate generic questions that could apply to any topic
- Do NOT ask about topics not mentioned in the answer
- Focus on specific concepts, examples, functions, or scenarios mentioned in the answer
- Make questions that would help someone dive deeper into the specific topics discussed

Guidelines:
- Look for specific terms, concepts, or examples mentioned in the answer
- Generate questions about those specific elements
- Ensure questions are specific enough to be meaningful
- Avoid overly generic questions like "Tell me more" or "What else"
- Make questions that would genuinely help someone learn more about the specific topics covered

Format your response as a JSON array of question strings only, like:
["Question 1?", "Question 2?", "Question 3?", "Question 4?"]

Do not include any explanations, just the questions."""
    
    def generate_suggestions(self, answer_content: str, 
                           context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate follow-up questions based on the answer content using OpenAI
        
        Args:
            answer_content: The main answer text from doc_node, db_node, etc.
            context: Additional context (query, persona, etc.)
            
        Returns:
            List of suggested follow-up questions
        """
        logger.info("Generating follow-up questions using OpenAI")
        
        try:
            # Get the original question from context
            original_question = context.get("query", "") if context else ""
            
            # Create the user prompt with both original question and answer content
            user_prompt = f"""Based on this original question and answer, generate 2-4 follow-up questions:

Original Question: "{original_question}"

Answer: {answer_content}

Generate follow-up questions that:
1. Are directly related to BOTH the original question AND the answer content
2. Help users dive deeper into the specific topics discussed
3. Continue the conversation naturally from where it left off
4. Reference specific concepts, examples, or details from the answer
5. Would lead to meaningful responses when asked"""

            # Use OpenAI to generate suggestions
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=200,
                temperature=0.7
            )
            
            # Parse the response
            suggestions = self._parse_llm_response(response.choices[0].message.content)
            
            # Format suggestions for frontend
            formatted_suggestions = []
            for i, question in enumerate(suggestions):
                formatted_suggestions.append({
                    'id': i + 1,
                    'title': question,
                    'type': 'follow_up',
                    'category': 'openai_generated',
                    'context': {
                        'original_question': original_question,
                        'related_to_answer': True
                    }
                })
            
            return {
                'success': True,
                'suggestions': formatted_suggestions,
                'total_suggestions': len(formatted_suggestions),
                'method': 'openai_generated',
                'context': {
                    'original_question': original_question
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating suggestions with OpenAI: {str(e)}")
            # Fallback to simple suggestions
            fallback_suggestions = self.create_simple_suggestions(answer_content, context)
            formatted_fallback = []
            for i, question in enumerate(fallback_suggestions):
                formatted_fallback.append({
                    'id': i + 1,
                    'title': question,
                    'type': 'follow_up',
                    'category': 'fallback',
                    'context': {
                        'original_question': context.get("query", "") if context else "",
                        'related_to_answer': True
                    }
                })
            
            return {
                'success': True,
                'suggestions': formatted_fallback,
                'total_suggestions': len(formatted_fallback),
                'method': 'fallback',
                'error': f'OpenAI generation failed, using fallback: {str(e)}',
                'context': {
                    'original_question': context.get("query", "") if context else ""
                }
            }
    
    def _parse_llm_response(self, response_content: str) -> List[str]:
        """Parse LLM response to extract questions"""
        import json
        import re
        
        try:
            # Try to parse as JSON first
            response_clean = response_content.strip()
            if response_clean.startswith('[') and response_clean.endswith(']'):
                questions = json.loads(response_clean)
                if isinstance(questions, list):
                    return [str(q).strip() for q in questions if q.strip()]
            
            # If JSON parsing fails, try to extract questions using regex
            # Look for patterns like "Question?", "What is...?", "How does...?", etc.
            question_patterns = [
                r'["\']([^"\']+\?)["\']',  # Quoted questions
                r'([A-Z][^.!?]*\?)',       # Questions starting with capital letter
                r'([^.!?]*\?[^.!?]*)',     # Any text ending with ?
            ]
            
            questions = []
            for pattern in question_patterns:
                matches = re.findall(pattern, response_content)
                for match in matches:
                    if isinstance(match, tuple):
                        match = match[0]
                    question = match.strip()
                    if len(question) > 10 and question not in questions:
                        questions.append(question)
            
            # If still no questions found, split by lines and look for question marks
            if not questions:
                lines = response_content.split('\n')
                for line in lines:
                    line = line.strip()
                    if '?' in line and len(line) > 10:
                        # Clean up the line
                        question = re.sub(r'^\d+\.\s*', '', line)  # Remove numbering
                        question = re.sub(r'^["\']|["\']$', '', question)  # Remove quotes
                        if question and question not in questions:
                            questions.append(question)
            
            return questions[:4]  # Return max 4 questions
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {str(e)}")
            return []
    
    def create_simple_suggestions(self, answer_content: str, context: Dict[str, Any] = None) -> List[str]:
        """Create simple follow-up questions without complex processing"""
        # Extract a few key terms
        import re
        
        # Get original question for context
        original_question = context.get("query", "") if context else ""
        
        # Find capitalized terms and acronyms from both answer and original question
        answer_terms = re.findall(r'\b[A-Z][a-zA-Z]{2,}\b|\b[A-Z]{2,}\b', answer_content)
        question_terms = re.findall(r'\b[A-Z][a-zA-Z]{2,}\b|\b[A-Z]{2,}\b', original_question)
        
        # Combine and prioritize terms that appear in both
        all_terms = list(set(answer_terms + question_terms))
        common_terms = [term for term in answer_terms if term in question_terms]
        
        # Use common terms first, then other terms
        terms = common_terms + [term for term in all_terms if term not in common_terms]
        terms = terms[:3]  # Get unique terms, limit to 3
        
        if not terms:
            # Fallback suggestions that relate to the original question
            if "mpi" in original_question.lower():
                return [
                    "How does MPI handle communication between processes?",
                    "What are the main MPI functions for data exchange?",
                    "Can you show me a simple MPI example?"
                ]
            else:
                return [
                    "Can you explain this in more detail?",
                    "What are the key points to remember?",
                    "How does this relate to other topics?"
                ]
        
        suggestions = []
        for term in terms:
            suggestions.extend([
                f"What is {term}?",
                f"How does {term} work?",
                f"Can you provide examples of {term}?"
            ])
        
        return suggestions[:5]  # Return max 5 suggestions 