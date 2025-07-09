#!/usr/bin/env python3
"""
Test script to verify Dynamic Agent System setup
"""
import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_environment_variables():
    """Test if environment variables are set"""
    print("🔍 Testing Environment Variables...")
    
    required_vars = [
        "OPENAI_API_KEY",
        "PINECONE_API_KEY",
        "PINECONE_ENVIRONMENT",
        "PINECONE_INDEX_NAME"
    ]
    
    missing_vars = []
    for var in required_vars:
        value = os.getenv(var)
        if not value or value == "your_" + var.lower() + "_here":
            missing_vars.append(var)
            print(f"❌ {var}: Not set or using placeholder")
        else:
            print(f"✅ {var}: Set (length: {len(value)})")
    
    if missing_vars:
        print(f"\n⚠️  Missing environment variables: {', '.join(missing_vars)}")
        print("📝 Please update your .env file with actual API keys")
        return False
    else:
        print("\n✅ All environment variables are set!")
        return True

def test_backend_health():
    """Test backend health endpoint"""
    print("\n🏥 Testing Backend Health...")
    
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print("✅ Backend is healthy!")
            print(f"   📊 System Mode: {health_data.get('system_mode')}")
            print(f"   🤖 Graph System Available: {health_data.get('graph_system_available')}")
            print(f"   📅 Version: {health_data.get('version')}")
            return health_data
        else:
            print(f"❌ Backend returned status: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"❌ Backend connection failed: {str(e)}")
        print("💡 Make sure to start the backend: python run_backend.py")
        return None

def test_query_processing():
    """Test query processing"""
    print("\n🧠 Testing Query Processing...")
    
    test_query = {
        "query": "Test query for system verification",
        "persona": "financial_analyst",
        "context": {"test": True}
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/query",
            json=test_query,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            system_mode = result.get("metadata", {}).get("system_mode", "unknown")
            
            print(f"✅ Query processed successfully!")
            print(f"   🎯 System Mode: {system_mode}")
            print(f"   ⏱️  Processing Time: {result.get('metadata', {}).get('processing_time', 0):.2f}s")
            
            # Check if using mock or actual responses
            response_text = result.get("formatted_response", {}).get("response", "")
            if "[MOCK]" in response_text:
                print("   ⚠️  Using mock responses (API keys not configured)")
            else:
                print("   🚀 Using actual graph system!")
            
            return result
        else:
            print(f"❌ Query failed with status: {response.status_code}")
            print(f"   Error: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Query request failed: {str(e)}")
        return None

def test_streamlit_connection():
    """Test Streamlit UI connection"""
    print("\n🎨 Testing Streamlit UI...")
    
    try:
        response = requests.get("http://localhost:8501/healthz", timeout=5)
        if response.status_code == 200:
            print("✅ Streamlit UI is running!")
            print("   🌐 Open http://localhost:8501 in your browser")
            return True
        else:
            print(f"❌ Streamlit returned status: {response.status_code}")
            return False
    except requests.exceptions.RequestException:
        print("❌ Streamlit UI not accessible")
        print("💡 Make sure to start Streamlit: python run_streamlit.py")
        return False

def main():
    """Run all tests"""
    print("🚀 Dynamic Agent System Setup Test\n")
    
    # Test environment variables
    env_ok = test_environment_variables()
    
    # Test backend health
    health_data = test_backend_health()
    backend_ok = health_data is not None
    
    # Test query processing
    query_result = test_query_processing()
    query_ok = query_result is not None
    
    # Test Streamlit
    streamlit_ok = test_streamlit_connection()
    
    # Summary
    print("\n" + "="*50)
    print("📋 SETUP TEST SUMMARY")
    print("="*50)
    
    print(f"Environment Variables: {'✅ PASS' if env_ok else '❌ FAIL'}")
    print(f"Backend Health: {'✅ PASS' if backend_ok else '❌ FAIL'}")
    print(f"Query Processing: {'✅ PASS' if query_ok else '❌ FAIL'}")
    print(f"Streamlit UI: {'✅ PASS' if streamlit_ok else '❌ FAIL'}")
    
    if health_data:
        print(f"\nSystem Mode: {health_data.get('system_mode')}")
        if health_data.get('system_mode') == 'actual_graph':
            print("🎉 SUCCESS: Real graph system is active!")
        else:
            print("⚠️  INFO: Using mock responses (configure API keys for full functionality)")
    
    if all([env_ok, backend_ok, query_ok]):
        print("\n🎉 Setup test completed successfully!")
        print("🚀 Your Dynamic Agent System is ready to use!")
    else:
        print("\n❌ Setup test failed. Please check the issues above.")
        print("📚 See SETUP.md for detailed configuration instructions.")

if __name__ == "__main__":
    main() 