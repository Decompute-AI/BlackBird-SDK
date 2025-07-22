#!/usr/bin/env python3
"""
Simple test script to verify web search endpoints are accessible
"""

import os
import sys
import json
import requests
from pathlib import Path

def load_env_file():
    """Load environment variables from .env file"""
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        print(f"📄 Loading environment from: {env_file}")
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
        print("✅ Environment variables loaded from .env file")
    else:
        print(f"⚠️ .env file not found at: {env_file}")

def test_backend_connectivity():
    """Test basic backend connectivity"""
    print("🔍 Testing backend connectivity...")
    
    try:
        # Test basic health endpoint
        response = requests.get("http://localhost:5012/health", timeout=5)
        print(f"Health endpoint status: {response.status_code}")
        
        # Test vision chat endpoint
        response = requests.get("http://localhost:5012/vision-chat/test", timeout=5)
        print(f"Vision chat endpoint status: {response.status_code}")
        
        return True
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False

def test_web_search_status():
    """Test web search status endpoint"""
    print("\n🔍 Testing web search status endpoint...")
    
    try:
        response = requests.get("http://localhost:5012/vision-chat/web-search-status", timeout=10)
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Web search status endpoint working!")
            print(f"Response: {json.dumps(data, indent=2)}")
            return True
        else:
            print(f"❌ Status endpoint failed with code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing status endpoint: {e}")
        return False

def test_web_search_test():
    """Test web search test endpoint"""
    print("\n🔍 Testing web search test endpoint...")
    
    try:
        payload = {
            'query': 'test query',
            'max_results': 2
        }
        
        response = requests.post(
            "http://localhost:5012/vision-chat/web-search/test",
            json=payload,
            timeout=30
        )
        
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Web search test endpoint working!")
            print(f"Response: {json.dumps(data, indent=2)}")
            return True
        else:
            print(f"❌ Test endpoint failed with code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing search endpoint: {e}")
        return False

def test_web_search_cache():
    """Test web search cache clear endpoint"""
    print("\n🔍 Testing web search cache clear endpoint...")
    
    try:
        response = requests.post("http://localhost:5012/vision-chat/web-search/clear-cache", timeout=10)
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Web search cache clear endpoint working!")
            print(f"Response: {json.dumps(data, indent=2)}")
            return True
        else:
            print(f"❌ Cache clear endpoint failed with code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing cache clear endpoint: {e}")
        return False

def main():
    """Main function"""
    print("🚀 Simple Web Search Endpoint Test")
    print("=" * 40)
    print()
    
    # Load environment variables
    load_env_file()
    
    # Check API key
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if api_key:
        print(f"✅ PERPLEXITY_API_KEY found (length: {len(api_key)})")
    else:
        print("⚠️ PERPLEXITY_API_KEY not found")
    
    # Test connectivity
    if not test_backend_connectivity():
        print("\n❌ Backend connectivity failed. Make sure the backend is running.")
        return
    
    # Test web search endpoints
    status_ok = test_web_search_status()
    test_ok = test_web_search_test()
    cache_ok = test_web_search_cache()
    
    # Summary
    print("\n" + "=" * 40)
    print("📊 TEST SUMMARY")
    print("=" * 40)
    print(f"Backend Connectivity: {'✅ PASS' if True else '❌ FAIL'}")
    print(f"Web Search Status: {'✅ PASS' if status_ok else '❌ FAIL'}")
    print(f"Web Search Test: {'✅ PASS' if test_ok else '❌ FAIL'}")
    print(f"Web Search Cache: {'✅ PASS' if cache_ok else '❌ FAIL'}")
    
    if all([status_ok, test_ok, cache_ok]):
        print("\n🎉 All tests passed! Web search endpoints are working correctly.")
    else:
        print("\n⚠️ Some tests failed. Check the error messages above.")

if __name__ == "__main__":
    main() 