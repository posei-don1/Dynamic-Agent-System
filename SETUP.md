# Setup Instructions for Dynamic Agent System

## Environment Variables Setup

To get actual responses instead of mock responses, you need to configure the following environment variables:

### 1. Create `.env` file

Create a `.env` file in the root directory:

```bash
# API Keys
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here

# Pinecone Configuration
PINECONE_ENVIRONMENT=your_pinecone_environment_here
PINECONE_INDEX_NAME=your_pinecone_index_name_here

# FastAPI Configuration
FASTAPI_HOST=0.0.0.0
FASTAPI_PORT=8000

# File Paths
DATA_PATH=./data/
UPLOAD_PATH=./uploads/

# Logging
LOG_LEVEL=INFO
```

### 2. Get API Keys

#### OpenAI API Key
1. Go to [OpenAI Platform](https://platform.openai.com/api-keys)
2. Create a new API key
3. Copy the key to your `.env` file

#### Pinecone API Key
1. Go to [Pinecone Console](https://app.pinecone.io/)
2. Create a new project or use existing
3. Go to API Keys section
4. Copy the API key and environment to your `.env` file

### 3. Load Environment Variables

The system will automatically load these variables on startup. You can also set them manually:

```bash
export OPENAI_API_KEY="your_key_here"
export PINECONE_API_KEY="your_key_here"
```

## System Modes

The system operates in two modes:

### 1. Actual Graph Mode (when properly configured)
- Uses the real LangGraph system
- Processes queries through specialized nodes
- Provides intelligent routing and persona-based responses
- **Indicator**: Responses show `system_mode: "actual_graph"` in metadata

### 2. Mock Mode (fallback)
- Uses mock responses when API keys are missing
- Responses are prefixed with `[MOCK]`
- **Indicator**: Responses show `system_mode: "mock_fallback"` in metadata

## Testing the Setup

### 1. Check System Status
When you start the backend, look for these log messages:
- ✅ `Dynamic Agent Graph initialized successfully` - Real system active
- ❌ `Failed to initialize graph system` - Mock mode active

### 2. Test Query Response
Send a test query and check the metadata:
```json
{
  "metadata": {
    "system_mode": "actual_graph"  // or "mock_fallback"
  }
}
```

### 3. Verify API Endpoints
- Use curl, Postman, or your frontend to test the backend endpoints.
- Visit `http://localhost:8000/docs` for interactive API documentation.

## Troubleshooting

### Common Issues

1. **Mock mode despite having API keys**
   - Check `.env` file is in the correct location
   - Restart the backend server
   - Check logs for initialization errors

2. **Graph initialization fails**
   - Verify API keys are correct
   - Check Pinecone index exists
   - Ensure all dependencies are installed

3. **Import errors**
   - Make sure you're running from the correct directory
   - Check all files are in place
   - Reinstall requirements: `pip install -r requirements.txt`

### Debug Commands

```bash
# Check if environment variables are loaded
python -c "import os; print(os.getenv('OPENAI_API_KEY', 'NOT_SET'))"

# Test graph system directly
python -c "from app.graph.graph_builder import DynamicAgentGraph; print('Graph import successful')"

# Check backend logs
python run_backend.py
```

## Next Steps

1. Set up your `.env` file with real API keys
2. Restart the backend server
3. Test queries using the backend API
4. Check that responses no longer have `[MOCK]` prefix
5. Verify `system_mode: "actual_graph"` in metadata

The system will automatically switch to actual graph processing once properly configured! 