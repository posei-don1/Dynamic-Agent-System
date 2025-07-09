"""
LangGraph Builder for Dynamic Agent System
Orchestrates the flow between different processing nodes
"""
from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict
import logging

from .nodes.persona_selector import PersonaSelector
from .nodes.router import Router
from .nodes.doc_node import DocNode
from .nodes.db_node import DbNode
from .nodes.math_node import MathNode
from .nodes.answer_formatter import AnswerFormatter
from .nodes.suggestion_node import SuggestionNode
from ..services.pdf_utils import PDFProcessor
from ..services.pinecone_service import PineconeService
from ..services.data_loader import DataLoader

logger = logging.getLogger(__name__)

class GraphState(TypedDict):
    """State passed between nodes in the graph"""
    query: str
    context: Dict[str, Any]
    persona: Dict[str, Any]
    route: Dict[str, Any]
    processed_data: Dict[str, Any]
    suggestions: Dict[str, Any]
    formatted_response: Dict[str, Any]
    error: Optional[str]
    metadata: Dict[str, Any]

class DynamicAgentGraph:
    """Main graph orchestrator for the dynamic agent system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize processing nodes
        self.persona_selector = PersonaSelector()
        self.router = Router()
        self.doc_node = DocNode()
        self.db_node = DbNode()
        self.math_node = MathNode()
        self.answer_formatter = AnswerFormatter()
        self.suggestion_node = SuggestionNode()
        
        # Initialize services
        self.pdf_processor = PDFProcessor(self.config.get('pdf_config', {}))
        self.pinecone_service = PineconeService(self.config.get('pinecone_config', {}))
        self.data_loader = DataLoader(self.config.get('data_config', {}))
        
        # Build the graph
        self.graph = self._build_graph()
        
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        logger.info("Building dynamic agent graph")
        
        # Create the graph
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("persona_selector", self._persona_selector_node)
        workflow.add_node("router", self._router_node)
        workflow.add_node("doc_processor", self._doc_processor_node)
        workflow.add_node("db_processor", self._db_processor_node)
        workflow.add_node("math_processor", self._math_processor_node)
        workflow.add_node("suggestion_generator", self._suggestion_generator_node)
        workflow.add_node("answer_formatter", self._answer_formatter_node)
        
        # Set entry point
        workflow.set_entry_point("persona_selector")
        
        # Define the flow
        workflow.add_edge("persona_selector", "router")
        workflow.add_conditional_edges(
            "router",
            self._route_decision,
            {
                "doc_processor": "doc_processor",
                "db_processor": "db_processor", 
                "math_processor": "math_processor",
                "suggestion_generator": "suggestion_generator"
            }
        )
        
        # All processing nodes go to suggestion generator
        workflow.add_edge("doc_processor", "suggestion_generator")
        workflow.add_edge("db_processor", "suggestion_generator")
        workflow.add_edge("math_processor", "suggestion_generator")
        
        # Suggestion generator goes to answer formatter
        workflow.add_edge("suggestion_generator", "answer_formatter")
        
        # Answer formatter is the end
        workflow.add_edge("answer_formatter", END)
        
        return workflow.compile()
    
    def _persona_selector_node(self, state: GraphState) -> GraphState:
        """Select appropriate persona for the query"""
        logger.info("Executing persona selector node")
        
        try:
            query = state["query"]
            context = state.get("context", {})
            
            # Select persona
            persona_result = self.persona_selector.select_persona(query, context)
            
            # Update state
            state["persona"] = persona_result
            state["metadata"] = state.get("metadata", {})
            state["metadata"]["persona_selected"] = persona_result.get("persona", "unknown")
            
            logger.info(f"Selected persona: {persona_result.get('persona', 'unknown')}")
            
        except Exception as e:
            logger.error(f"Error in persona selector: {str(e)}")
            state["error"] = f"Persona selection failed: {str(e)}"
        
        return state
    
    def _router_node(self, state: GraphState) -> GraphState:
        """Route query to appropriate processing node"""
        logger.info("Executing router node")
        
        try:
            query = state["query"]
            context = state.get("context", {})
            
            # Route the query
            route_result = self.router.route_query(query, context)
            
            # Update state
            state["route"] = route_result
            state["metadata"]["route_selected"] = route_result.get("node", "unknown")
            state["metadata"]["route_confidence"] = route_result.get("confidence", 0.0)
            
            logger.info(f"Routed to: {route_result.get('node', 'unknown')}")
            
        except Exception as e:
            logger.error(f"Error in router: {str(e)}")
            state["error"] = f"Routing failed: {str(e)}"
        
        return state
    
    def _route_decision(self, state: GraphState) -> str:
        """Decide which processing node to use"""
        if state.get("error"):
            return "suggestion_generator"
        
        route = state.get("route", {})
        selected_node = route.get("node", "db_processor")
        
        # Map router decisions to actual node names
        node_mapping = {
            "doc_node": "doc_processor",
            "db_node": "db_processor",
            "math_node": "math_processor",
            "suggestion_node": "suggestion_generator"
        }
        
        return node_mapping.get(selected_node, "db_processor")
    
    def _doc_processor_node(self, state: GraphState) -> GraphState:
        """Process document-related queries"""
        logger.info("Executing document processor node")
        
        try:
            query = state["query"]
            context = state.get("context", {})
            
            # Check if we have a document to process
            if "document_path" in context:
                doc_path = context["document_path"]
                result = self.doc_node.process_document(doc_path, query, context)
            elif "contract_path" in context:
                contract_path = context["contract_path"]
                result = self.doc_node.analyze_contract(contract_path)
            else:
                # If no document specified, use sample contract
                result = self.doc_node.analyze_contract("./data/sample_contract.pdf")
            
            state["processed_data"] = result
            
        except Exception as e:
            logger.error(f"Error in document processor: {str(e)}")
            state["error"] = f"Document processing failed: {str(e)}"
            state["processed_data"] = {"error": str(e)}
        
        return state
    
    def _db_processor_node(self, state: GraphState) -> GraphState:
        """Process data-related queries"""
        logger.info("Executing database processor node")
        
        try:
            query = state["query"]
            context = state.get("context", {})
            
            # Load data if specified
            if "data_source" in context:
                data_source = context["data_source"]
                if data_source.endswith('.csv'):
                    # Load CSV data
                    load_result = self.data_loader.load_csv(f"./data/{data_source}")
                    if load_result.get("success"):
                        data_name = load_result["data_name"]
                        # Query the loaded data
                        result = self.db_node.query_data(query, data_name)
                        result["load_info"] = load_result["summary"]
                    else:
                        result = load_result
                else:
                    result = {"error": f"Unsupported data source: {data_source}"}
            else:
                # Try to load default data
                load_result = self.data_loader.load_csv("./data/msft_2024.csv")
                if load_result.get("success"):
                    data_name = load_result["data_name"]
                    result = self.db_node.query_data(query, data_name)
                    result["load_info"] = load_result["summary"]
                else:
                    result = {"error": "No data source specified and default data not available"}
            
            state["processed_data"] = result
            
        except Exception as e:
            logger.error(f"Error in database processor: {str(e)}")
            state["error"] = f"Database processing failed: {str(e)}"
            state["processed_data"] = {"error": str(e)}
        
        return state
    
    def _math_processor_node(self, state: GraphState) -> GraphState:
        """Process mathematical queries"""
        logger.info("Executing math processor node")
        
        try:
            query = state["query"]
            context = state.get("context", {})
            
            # Process mathematical calculation
            result = self.math_node.calculate(query, context)
            
            state["processed_data"] = result
            
        except Exception as e:
            logger.error(f"Error in math processor: {str(e)}")
            state["error"] = f"Math processing failed: {str(e)}"
            state["processed_data"] = {"error": str(e)}
        
        return state
    
    def _suggestion_generator_node(self, state: GraphState) -> GraphState:
        """Generate suggestions based on processed data"""
        logger.info("Executing suggestion generator node")
        
        try:
            processed_data = state.get("processed_data", {})
            persona = state.get("persona", {})
            context = state.get("context", {})
            
            # Determine suggestion type based on persona
            persona_name = persona.get("persona", "general")
            suggestion_type_mapping = {
                "financial_analyst": "financial",
                "legal_advisor": "document",
                "data_scientist": "data",
                "business_consultant": "business"
            }
            
            suggestion_type = suggestion_type_mapping.get(persona_name, "general")
            
            # Generate suggestions
            suggestions = self.suggestion_node.generate_suggestions(
                processed_data, 
                suggestion_type, 
                context
            )
            
            state["suggestions"] = suggestions
            
        except Exception as e:
            logger.error(f"Error in suggestion generator: {str(e)}")
            state["error"] = f"Suggestion generation failed: {str(e)}"
            state["suggestions"] = {"error": str(e)}
        
        return state
    
    def _answer_formatter_node(self, state: GraphState) -> GraphState:
        """Format the final response"""
        logger.info("Executing answer formatter node")
        
        try:
            processed_data = state.get("processed_data", {})
            suggestions = state.get("suggestions", {})
            persona = state.get("persona", {})
            query = state["query"]
            
            # Determine format type based on processed data
            if processed_data.get("analysis_type") == "contract":
                format_type = "document_analysis"
            elif "result" in processed_data and processed_data["result"].get("type") in ["display", "count", "sum", "average"]:
                format_type = "data_summary"
            elif processed_data.get("calculation_type"):
                format_type = "calculation"
            elif suggestions.get("suggestions"):
                format_type = "suggestion"
            else:
                format_type = "general"
            
            # Format the response
            formatted_response = self.answer_formatter.format_response(
                response_data={
                    "processed_data": processed_data,
                    "suggestions": suggestions,
                    "metadata": state.get("metadata", {})
                },
                format_type=format_type,
                persona=persona.get("persona"),
                query=query
            )
            
            state["formatted_response"] = formatted_response
            
        except Exception as e:
            logger.error(f"Error in answer formatter: {str(e)}")
            state["error"] = f"Answer formatting failed: {str(e)}"
            state["formatted_response"] = {"error": str(e)}
        
        return state
    
    async def process_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a query through the agent system
        
        Args:
            query: User query
            context: Additional context information
            
        Returns:
            Processed response
        """
        logger.info(f"Processing query: {query[:100]}...")
        
        try:
            # Initialize state
            initial_state = GraphState(
                query=query,
                context=context or {},
                persona={},
                route={},
                processed_data={},
                suggestions={},
                formatted_response={},
                error=None,
                metadata={}
            )
            
            # Run the graph
            result = await self.graph.ainvoke(initial_state)
            
            # Return the final formatted response
            return {
                "success": True,
                "response": result["formatted_response"],
                "metadata": result.get("metadata", {}),
                "error": result.get("error")
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "success": False,
                "error": f"Query processing failed: {str(e)}"
            }
    
    def process_query_sync(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a query through the agent system (synchronous version)
        
        Args:
            query: User query
            context: Additional context information
            
        Returns:
            Processed response
        """
        logger.info(f"Processing query (sync): {query[:100]}...")
        
        try:
            # Initialize state
            initial_state = GraphState(
                query=query,
                context=context or {},
                persona={},
                route={},
                processed_data={},
                suggestions={},
                formatted_response={},
                error=None,
                metadata={}
            )
            
            # Run the graph
            result = self.graph.invoke(initial_state)
            
            # Return the final formatted response
            return {
                "success": True,
                "response": result["formatted_response"],
                "metadata": result.get("metadata", {}),
                "error": result.get("error")
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "success": False,
                "error": f"Query processing failed: {str(e)}"
            }
    
    def get_graph_visualization(self) -> str:
        """Get a visual representation of the graph"""
        try:
            # Return a text representation of the graph structure
            return """
            Dynamic Agent System Graph Structure:
            
            Entry Point: persona_selector
            
            persona_selector → router → [doc_processor | db_processor | math_processor | suggestion_generator]
                                ↓
            suggestion_generator → answer_formatter → END
            
            Node Functions:
            - persona_selector: Selects appropriate AI persona
            - router: Routes to specialized processing node
            - doc_processor: Handles document analysis
            - db_processor: Handles data queries
            - math_processor: Handles calculations
            - suggestion_generator: Generates recommendations
            - answer_formatter: Formats final response
            """
        except Exception as e:
            logger.error(f"Error generating graph visualization: {str(e)}")
            return f"Error generating visualization: {str(e)}"
