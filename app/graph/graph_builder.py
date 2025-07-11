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
import os  # Add os import for file operations
import numpy as np  # Add numpy import for CSV operations

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
            if state.get("metadata") is None:
                state["metadata"] = {}
            state["metadata"]["persona_selected"] = persona_result.get("persona", "unknown")
            
            logger.info(f"Selected persona: {persona_result.get('persona', 'unknown')}")
            
        except Exception as e:
            logger.error(f"Error in persona selector: {str(e)}")
            state["error"] = f"Persona selection failed: {str(e)}"
            if state.get("metadata") is None:
                state["metadata"] = {}
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
            if state.get("metadata") is None:
                state["metadata"] = {}
            state["metadata"]["route_selected"] = route_result.get("node", "unknown")
            state["metadata"]["route_confidence"] = route_result.get("confidence", 0.0)
            
            logger.info(f"Routed to: {route_result.get('node', 'unknown')}")
            
        except Exception as e:
            logger.error(f"Error in router: {str(e)}")
            state["error"] = f"Routing failed: {str(e)}"
            if state.get("route") is None:
                state["route"] = {}
            if state.get("metadata") is None:
                state["metadata"] = {}
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
        """Process data-related queries with LLM tool-calling prompt engineering"""
        logger.info("Executing database processor node")
        try:
            query = state["query"]
            context = state.get("context", {})

            # --- LLM Tool-Calling Prompt Engineering ---
            # If the query is ambiguous or could use a tool, prompt the LLM to select the tool and column.
            # Example prompt:
            # "Given the user query: '{query}', select the most appropriate tool from [mean, sum, median, std, min, max, count, describe, head, tail, sample, shape, columns, info] and the column to apply it to (if applicable)."
            # The LLM should return a JSON: {"tool": "mean", "column": "price"} or {"tool": "head", "n": 5}
            # The backend will then call: result = math_node.dispatch(tool, df[column].dropna().tolist()) or result = math_node.dispatch(tool, df.to_dict('records'), n=n)
            # This enables dynamic, robust tool selection for structured data analysis.

            # Load data if specified
            if "data_source" in context:
                data_source = context["data_source"]
                if data_source.endswith('.csv'):
                    # Always load CSV from uploads/structured
                    file_path = f"./data/uploads/structured/{data_source}"
                    load_result = self.data_loader.load_csv(file_path)
                    if load_result.get("success"):
                        data_name = load_result["data_name"]
                        # Query the loaded data (which now supports LLM tool-calling logic)
                        result = self.db_node.query_data(query, data_name)
                        result["load_info"] = load_result["summary"]
                    else:
                        result = load_result
                else:
                    result = {"error": f"Unsupported data source: {data_source}"}
            else:
                # Try to load the most recent file from uploads/structured
                structured_dir = "./data/uploads/structured/"
                try:
                    files = [f for f in os.listdir(structured_dir) if f.endswith('.csv')]
                    if not files:
                        raise FileNotFoundError("No CSV files found in uploads/structured.")
                    # Use the most recently modified file
                    files.sort(key=lambda x: os.path.getmtime(os.path.join(structured_dir, x)), reverse=True)
                    latest_file = files[0]
                    file_path = os.path.join(structured_dir, latest_file)
                    load_result = self.data_loader.load_csv(file_path)
                    if load_result.get("success"):
                        data_name = load_result["data_name"]
                        result = self.db_node.query_data(query, data_name)
                        result["load_info"] = load_result["summary"]
                    else:
                        result = load_result
                except Exception as e:
                    result = {"error": f"No data source specified and no CSV in uploads/structured: {str(e)}"}

            state["processed_data"] = result
        except Exception as e:
            logger.error(f"Error in database processor: {str(e)}")
            state["error"] = f"Database processing failed: {str(e)}"
            if state.get("processed_data") is None:
                state["processed_data"] = {}
            state["processed_data"] = {"error": str(e)}
        return state
    
    def _math_processor_node(self, state: GraphState) -> GraphState:
        """Process mathematical queries, including DB+Math pipeline with dynamic files"""
        logger.info("Executing math processor node")
        try:
            query = state["query"]
            context = state.get("context", {})
            
            # If context signals a DB+Math pipeline, work with uploaded file
            if context.get("db_math_pipeline"):
                db_node = self.db_node
                math_node = self.math_node
                
                # Get file path from context
                file_path = context.get("file_path")
                if not file_path:
                    state["error"] = "No file path provided in context"
                    state["processed_data"] = {"error": state["error"]}
                    return state
                
                # Load the file using db_node
                data_name = "uploaded_file"
                load_result = db_node.load_data(file_path, data_name)
                if not load_result.get("success"):
                    state["error"] = load_result.get("error", "Failed to load file")
                    state["processed_data"] = load_result
                    return state
                
                # Extract parameters from context
                symbol = context.get("symbol", "")
                start_date = context.get("start_date", "")
                end_date = context.get("end_date", "")
                window = context.get("window", 20)
                calculation_type = context.get("calculation_type", "moving_average")
                
                # Fetch price data using flexible column detection
                db_result = db_node.get_stock_prices(symbol, start_date, end_date, data_name, context)
                if not db_result.get("success"):
                    state["error"] = db_result.get("error", "Failed to fetch price data")
                    state["processed_data"] = db_result
                    return state
                
                # Perform the financial calculation using Python tools
                if calculation_type == "moving_average":
                    calc_result = math_node.moving_average(db_result["data"], window)
                elif calculation_type == "exponential_moving_average":
                    calc_result = math_node.exponential_moving_average(db_result["data"], context.get("span", 12))
                elif calculation_type == "bollinger_bands":
                    calc_result = math_node.bollinger_bands(db_result["data"], window, context.get("num_std", 2.0))
                elif calculation_type == "rsi":
                    calc_result = math_node.rsi(db_result["data"], context.get("rsi_window", 14))
                else:
                    # Use the generic financial metric calculator
                    calc_result = math_node.calculate_financial_metric(db_result["data"], calculation_type, **context)
                
                if not calc_result.get("success"):
                    state["error"] = calc_result.get("error", "Failed to compute financial metric")
                    state["processed_data"] = calc_result
                    return state
                
                # Combine results with metadata
                state["processed_data"] = {
                    "success": True,
                    "file_path": file_path,
                    "symbol": symbol or "All data",
                    "start_date": start_date or "All dates",
                    "end_date": end_date or "All dates",
                    "calculation_type": calculation_type,
                    "parameters": {
                        "window": window,
                        "span": context.get("span", 12),
                        "num_std": context.get("num_std", 2.0),
                        "rsi_window": context.get("rsi_window", 14)
                    },
                    "columns_used": db_result.get("columns_used", {}),
                    "calculation_result": calc_result,
                    "raw_data": db_result["data"]
                }
            else:
                # Process regular mathematical calculation as before
                result = self.math_node.calculate(query, context)
                state["processed_data"] = result
                
        except Exception as e:
            logger.error(f"Error in math processor: {str(e)}")
            state["error"] = f"Math processing failed: {str(e)}"
            if state.get("processed_data") is None:
                state["processed_data"] = {}
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
            if state.get("suggestions") is None:
                state["suggestions"] = {}
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
            if state.get("metadata") is None:
                state["metadata"] = {}
            formatted_response = self.answer_formatter.format_response(
                response_data={
                    "processed_data": processed_data,
                    "suggestions": suggestions,
                    "metadata": state["metadata"]
                },
                format_type=format_type,
                query=query,
                persona=persona.get("persona", "")
            )
            
            state["formatted_response"] = formatted_response
            
        except Exception as e:
            logger.error(f"Error in answer formatter: {str(e)}")
            state["error"] = f"Answer formatting failed: {str(e)}"
            if state.get("formatted_response") is None:
                state["formatted_response"] = {}
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
