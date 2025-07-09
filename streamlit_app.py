import streamlit as st
import requests
import json
import time
from typing import Dict, Any, Optional

# Configure Streamlit page
st.set_page_config(
    page_title="Dynamic Agent System - Test UI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .response-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .error-container {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #f44336;
        margin: 1rem 0;
    }
    
    .success-container {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4caf50;
        margin: 1rem 0;
    }
    
    .stButton > button {
        background-color: #667eea;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #5a6fd8;
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown("""
<div class="main-header">
    <h1>🤖 Dynamic Agent System</h1>
    <p>Test your LangGraph-based multi-agent system</p>
</div>
""", unsafe_allow_html=True)

# Backend configuration
BACKEND_URL = "http://localhost:8000"

# Initialize session state
if 'response_history' not in st.session_state:
    st.session_state.response_history = []

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Backend URL configuration
    backend_url = st.text_input("Backend URL", value=BACKEND_URL)
    
    # Test backend connection
    if st.button("🔗 Test Connection"):
        try:
            response = requests.get(f"{backend_url}/health", timeout=5)
            if response.status_code == 200:
                st.success("✅ Backend is connected!")
            else:
                st.error(f"❌ Backend returned status: {response.status_code}")
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Connection failed: {str(e)}")
    
    st.divider()
    
    # Response history
    st.header("📜 Response History")
    if st.session_state.response_history:
        for i, (query, persona, timestamp) in enumerate(st.session_state.response_history[-5:]):
            st.write(f"**{i+1}.** *{persona}*")
            st.write(f"└─ {query[:50]}...")
            st.caption(f"⏰ {timestamp}")
    else:
        st.write("No queries yet")
    
    if st.button("🗑️ Clear History"):
        st.session_state.response_history = []
        st.rerun()

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("💬 Query Interface")
    
    # Query input
    query = st.text_area(
        "Enter your query:",
        placeholder="e.g., 'Analyze the Microsoft financial data for Q4 trends' or 'Review the contract terms'",
        height=100,
        help="Enter any question or request for the AI agent system"
    )
    
    # Persona selection
    persona_options = {
        "financial_analyst": "💼 Financial Analyst",
        "legal_advisor": "⚖️ Legal Advisor", 
        "data_scientist": "📊 Data Scientist",
        "business_consultant": "🏢 Business Consultant"
    }
    
    selected_persona = st.selectbox(
        "Select AI Persona:",
        options=list(persona_options.keys()),
        format_func=lambda x: persona_options[x],
        help="Choose the AI persona that best matches your query type"
    )
    
    # Additional context options
    with st.expander("🔧 Advanced Options"):
        include_suggestions = st.checkbox("Include suggestions", value=True)
        detailed_analysis = st.checkbox("Detailed analysis", value=False)
        
        # Context input
        context = st.text_area(
            "Additional context (optional):",
            placeholder="Any additional information or constraints...",
            height=80
        )

with col2:
    st.header("📁 File Upload")
    
    # File upload section
    uploaded_file = st.file_uploader(
        "Upload a file (PDF or CSV):",
        type=['pdf', 'csv', 'xlsx'],
        help="Upload documents or data files for analysis"
    )
    
    if uploaded_file is not None:
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        
        # File details
        st.write(f"**File type:** {uploaded_file.type}")
        st.write(f"**File size:** {uploaded_file.size} bytes")
        
        # Upload file to backend (stub for now)
        if st.button("📤 Upload to Backend"):
            try:
                files = {"file": uploaded_file.getvalue()}
                response = requests.post(f"{backend_url}/upload", files=files)
                if response.status_code == 200:
                    st.success("File uploaded successfully!")
                else:
                    st.error(f"Upload failed: {response.status_code}")
            except Exception as e:
                st.error(f"Upload error: {str(e)}")

# Query execution
st.divider()

col1, col2, col3 = st.columns([1, 1, 1])

with col2:
    ask_button = st.button("🚀 Ask Agent", use_container_width=True, type="primary")

# Process query when button is clicked
if ask_button and query.strip():
    # Prepare payload
    payload = {
        "query": query.strip(),
        "persona": selected_persona,
        "context": {
            "include_suggestions": include_suggestions,
            "detailed_analysis": detailed_analysis,
            "additional_context": context.strip() if context.strip() else None
        }
    }
    
    # Show loading spinner
    with st.spinner("🤔 Agent is thinking..."):
        try:
            # Make API request
            response = requests.post(
                f"{backend_url}/query",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Add to history
                st.session_state.response_history.append(
                    (query, selected_persona, time.strftime("%H:%M:%S"))
                )
                
                # Display response
                st.markdown('<div class="success-container">', unsafe_allow_html=True)
                st.success("✅ Response received successfully!")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Response tabs
                tab1, tab2, tab3 = st.tabs(["📝 Formatted Response", "🔍 Raw JSON", "📊 Metadata"])
                
                with tab1:
                    st.markdown('<div class="response-container">', unsafe_allow_html=True)
                    
                    # Display formatted response
                    if "formatted_response" in result:
                        formatted = result["formatted_response"]
                        st.markdown(f"**Response:** {formatted.get('response', 'No response')}")
                        
                        if "suggestions" in formatted:
                            st.markdown("**Suggestions:**")
                            for suggestion in formatted["suggestions"]:
                                st.markdown(f"• {suggestion}")
                    else:
                        st.json(result)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with tab2:
                    st.json(result)
                
                with tab3:
                    # Display processing metadata
                    if "metadata" in result:
                        metadata = result["metadata"]
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Selected Persona", metadata.get("persona_selected", "Unknown"))
                            st.metric("Processing Route", metadata.get("route_selected", "Unknown"))
                        
                        with col2:
                            st.metric("Route Confidence", f"{metadata.get('route_confidence', 0):.2f}")
                            st.metric("Processing Time", f"{metadata.get('processing_time', 0):.2f}s")
                    else:
                        st.info("No metadata available")
                
            else:
                st.markdown('<div class="error-container">', unsafe_allow_html=True)
                st.error(f"❌ Error {response.status_code}: {response.text}")
                st.markdown('</div>', unsafe_allow_html=True)
                
        except requests.exceptions.RequestException as e:
            st.markdown('<div class="error-container">', unsafe_allow_html=True)
            st.error(f"❌ Request failed: {str(e)}")
            st.markdown('</div>', unsafe_allow_html=True)
            
        except Exception as e:
            st.markdown('<div class="error-container">', unsafe_allow_html=True)
            st.error(f"❌ Unexpected error: {str(e)}")
            st.markdown('</div>', unsafe_allow_html=True)

elif ask_button and not query.strip():
    st.warning("⚠️ Please enter a query before asking the agent.")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    🤖 Dynamic Agent System Test UI | Built with Streamlit
</div>
""", unsafe_allow_html=True)

# Example queries section
with st.expander("💡 Example Queries"):
    st.markdown("""
    **Financial Analysis Examples:**
    - "Analyze the Microsoft financial data for revenue trends"
    - "Calculate the year-over-year growth rate"
    - "What's the profit margin for Q4?"
    
    **Legal Analysis Examples:**
    - "Review the contract terms and highlight risks"
    - "Summarize the key obligations in the agreement"
    - "Check for any unusual clauses"
    
    **Data Science Examples:**
    - "Perform statistical analysis on the dataset"
    - "Find correlations between variables"
    - "Generate a data summary report"
    
    **Business Consulting Examples:**
    - "Provide strategic recommendations based on the data"
    - "Analyze market trends and opportunities"
    - "Suggest process improvements"
    """) 