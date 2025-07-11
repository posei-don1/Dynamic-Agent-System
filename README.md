# Dynamic Agent System

A powerful, multi-agent system built with LangGraph that intelligently routes queries to specialized processing nodes based on context and content analysis.

## Features

- **Intelligent Routing**: Automatically routes queries to the most appropriate processing node
- **Multiple Personas**: Financial analyst, legal advisor, data scientist, and business consultant personas
- **Document Processing**: Advanced PDF processing with OCR support
- **Data Analysis**: CSV/Excel data loading and analysis capabilities
- **Vector Search**: Semantic search using Pinecone vector database
- **Mathematical Computing**: Complex calculations and statistical analysis
- **Suggestion Engine**: Intelligent recommendations based on analysis results
- **Response Formatting**: User-friendly response formatting with structured output

## Architecture

The system is built using LangGraph and consists of several key components:

- **Graph Builder**: Orchestrates the flow between different nodes
- **Persona Selector**: Chooses the appropriate AI persona for the query
- **Router**: Intelligently routes queries to specialized processing nodes
- **Processing Nodes**: 
  - Document Node: PDF analysis and processing
  - Database Node: Data query and analysis
  - Math Node: Mathematical calculations
  - Suggestion Node: Recommendation generation
- **Answer Formatter**: Formats responses for optimal user experience

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd dynamic-agent-system
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

## Configuration

To get **actual responses** instead of mock responses, create a `.env` file in the project root and configure your API keys:

```bash
# API Keys
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here

# Pinecone Configuration
PINECONE_ENVIRONMENT=your_pinecone_environment_here
PINECONE_INDEX_NAME=your_pinecone_index_name_here
```

**Without these API keys, the system will use mock responses for testing.**

### Getting API Keys

1. **OpenAI API Key**: Get from [OpenAI Platform](https://platform.openai.com/api-keys)
2. **Pinecone API Key**: Get from [Pinecone Console](https://app.pinecone.io/)

### Testing Your Setup

Run the setup test script to verify everything is working:

```bash
python test_setup.py
```

This will check:
- Environment variables are set
- Backend is running
- Query processing is working
- Whether you're using actual or mock responses

## Usage

### Starting the Application

Start the FastAPI backend:
```bash
python run_backend.py
```
The backend will start on `http://localhost:8000`

#### Option 2: Direct API Usage

```bash
python app/main.py
```

### API Endpoints

- `POST /query`: Process a query through the agent system
- `POST /upload`: Upload documents for processing
- `GET /health`: Health check endpoint
- `GET /docs`: Interactive API documentation

### Example Usage

```python
import requests

# Process a query
response = requests.post("http://localhost:8000/query", json={
    "query": "Analyze the revenue trends in the Microsoft data",
    "persona": "financial_analyst",
    "context": {"data_source": "msft_2024.csv"}
})

print(response.json())
```

## File Structure

```
dynamic-agent-system/
│
├── app/
│   ├── __init__.py
│   ├── main.py               # FastAPI entrypoint
│   │
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── graph_builder.py  # LangGraph setup
│   │   ├── nodes/
│   │   │   ├── __init__.py
│   │   │   ├── persona_selector.py
│   │   │   ├── router.py
│   │   │   ├── doc_node.py
│   │   │   ├── db_node.py
│   │   │   ├── math_node.py
│   │   │   ├── answer_formatter.py
│   │   │   └── suggestion_node.py
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── pdf_utils.py      # PDF OCR + chunking
│   │   ├── pinecone_service.py
│   │   └── data_loader.py    # CSV/DB loader
│
├── data/                     # Sample CSVs, PDFs
│   ├── msft_2024.csv
│   └── sample_contract.pdf
│
├── streamlit_app.py          # Streamlit UI for testing
├── run_backend.py            # Script to run FastAPI backend
├── run_streamlit.py          # Script to run Streamlit UI
├── test_setup.py             # Setup verification script
├── requirements.txt
├── SETUP.md                  # Detailed setup instructions
├── .env.example              # Environment variables template
└── README.md
```

## Dependencies

- **LangGraph**: For building the agent workflow
- **FastAPI**: Web framework for API endpoints
- **Streamlit**: Interactive web UI for testing
- **Pandas**: Data manipulation and analysis
- **OpenAI**: Language model integration
- **Pinecone**: Vector database for semantic search
- **PDFplumber**: PDF processing
- **Requests**: HTTP client library
- **Uvicorn**: ASGI server

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black app/
isort app/
```

### Linting

```bash
flake8 app/
```

## Processing Nodes

### Document Node
- Extracts text from PDFs using OCR when necessary
- Chunks documents for better processing
- Performs semantic search on document content
- Analyzes contracts for key terms and conditions

### Database Node
- Loads CSV, Excel, and JSON files
- Performs natural language queries on data
- Provides statistical analysis and aggregations
- Supports data filtering and manipulation

### Math Node
- Handles mathematical expressions and calculations
- Supports statistical computations
- Provides financial calculations (compound interest, etc.)
- Evaluates formulas and equations

### Suggestion Node
- Generates contextual recommendations
- Creates action plans based on analysis
- Prioritizes suggestions by importance
- Provides implementation timelines

## Personas

The system supports multiple AI personas:

- **Financial Analyst**: Expert in financial data analysis and market trends
- **Legal Advisor**: Specializes in contract review and legal analysis
- **Data Scientist**: Focuses on statistical analysis and data modeling
- **Business Consultant**: Provides strategic business insights

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## System Modes

The system operates in two modes:

### 🚀 Actual Graph Mode
- **When**: API keys are properly configured in `.env` file
- **Features**: Real LangGraph processing, intelligent routing, persona-based responses
- **Indicator**: Responses don't have `[MOCK]` prefix, metadata shows `system_mode: "actual_graph"`

### 🧪 Mock Mode (Fallback)
- **When**: API keys are missing or invalid
- **Features**: Simulated responses for testing the UI and API structure
- **Indicator**: Responses have `[MOCK]` prefix, metadata shows `system_mode: "mock_fallback"`

You can switch between modes by configuring your `.env` file and restarting the backend. The system will automatically detect which mode to use based on available API keys.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions or issues, please open an issue on the GitHub repository or contact the development team.

## Roadmap

- [ ] Add support for more document formats
- [ ] Implement real-time collaboration features
- [ ] Add more sophisticated AI reasoning capabilities
- [ ] Expand database connector support
- [ ] Add visualization capabilities
- [ ] Implement user authentication and permissions 

## How to Run the Dynamic Agent System

Here are the step-by-step instructions to run your Dynamic Agent System:

### Prerequisites

1. **Navigate to the project directory**:
```bash
cd Dynamic-Agent-System
```

2. **Install dependencies** (if not already installed):
```bash
pip install -r requirements.txt
```

### Running the System

Start the FastAPI backend:
```bash
python run_backend.py
```

**OR** alternatively:
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

✅ **Expected Output**: 
- Backend will start on `http://localhost:8000`
- You should see FastAPI startup messages
- API documentation will be available at `http://localhost:8000/docs`

### Testing the System

You can test the backend API using curl, Postman, or your frontend integration:

```bash
# Test with curl
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "Test query", "persona": "financial_analyst"}'

# Or visit the interactive docs
# http://localhost:8000/docs
```

### Stopping the System

- Press `Ctrl+C` in the terminal to stop the backend
- Or close the terminal window

That's it! The system should now be running with a beautiful Streamlit interface for testing your Dynamic Agent System. 