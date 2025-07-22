#!/usr/bin/env python3
"""
Test runner for Perplexity web search API tests
Runs both comprehensive API tests and focused query generation tests
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

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

def main():
    """Main test runner"""
    print("🚀 Perplexity Web Search API Test Runner")
    print("=" * 50)
    print()
    
    # Load environment variables from .env file
    load_env_file()
    
    # Check if backend is running first
    print("🔍 Checking backend connectivity...")
    try:
        import requests
        # Use the correct backend URL
        response = requests.get("http://localhost:5012/vision-chat/web-search-status", timeout=5)
        if response.status_code == 200:
            print("✅ Backend is running on http://localhost:5012")
        else:
            print("⚠️ Backend responded but with status:", response.status_code)
    except Exception as e:
        print("❌ Cannot connect to backend. Make sure it's running on http://localhost:5012")
        print("   Start the backend with: python decompute.py in sdk/blackbird_sdk/backends/windows/")
        return
    
    print()
    print("Choose test suite to run:")
    print("1. Comprehensive API Tests (tests all endpoints)")
    print("2. Query Generation Tests (tests search query logic)")
    print("3. Run Both Test Suites")
    print("4. Exit")
    print()
    
    try:
        choice = input("Enter your choice (1-4): ").strip()
        
        if choice == "1":
            print("\n" + "="*50)
            print("Running Comprehensive API Tests...")
            print("="*50)
            run_comprehensive_tests()
            
        elif choice == "2":
            print("\n" + "="*50)
            print("Running Query Generation Tests...")
            print("="*50)
            run_query_generation_tests()
            
        elif choice == "3":
            print("\n" + "="*50)
            print("Running Both Test Suites...")
            print("="*50)
            run_comprehensive_tests()
            print("\n" + "="*50)
            run_query_generation_tests()
            
        elif choice == "4":
            print("👋 Goodbye!")
            return
            
        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")
            return
            
    except KeyboardInterrupt:
        print("\n👋 Test run interrupted.")
    except Exception as e:
        print(f"❌ Error running tests: {e}")

def run_comprehensive_tests():
    """Run the comprehensive API tests"""
    try:
        from test_perplexity_api import PerplexityAPITester
        
        tester = PerplexityAPITester()
        results = tester.run_all_tests()
        tester.save_results(results)
        
        return results
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure test_perplexity_api.py is in the same directory.")
    except Exception as e:
        print(f"❌ Error running comprehensive tests: {e}")

def run_query_generation_tests():
    """Run the query generation tests"""
    try:
        from test_query_generation import QueryGenerationTester
        
        tester = QueryGenerationTester()
        results = tester.run_all_tests()
        tester.save_results(results)
        
        return results
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure test_query_generation.py is in the same directory.")
    except Exception as e:
        print(f"❌ Error running query generation tests: {e}")

def check_prerequisites():
    """Check if all prerequisites are met"""
    print("🔍 Checking prerequisites...")
    
    # Load environment variables from .env file
    load_env_file()
    
    # Check if required files exist
    required_files = [
        "test_perplexity_api.py",
        "test_query_generation.py",
        "llm_web_search.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        return False
    
    # Check if Perplexity API key is set
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        print("⚠️ PERPLEXITY_API_KEY not set in environment")
        print("   Check your .env file contains: PERPLEXITY_API_KEY=your_key_here")
    else:
        print("✅ PERPLEXITY_API_KEY is configured")
    
    # Check if requests library is available
    try:
        import requests
        print("✅ requests library is available")
    except ImportError:
        print("❌ requests library not found")
        print("   Install with: pip install requests")
        return False
    
    print("✅ Prerequisites check completed")
    return True

if __name__ == "__main__":
    # Check prerequisites first
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please fix the issues above and try again.")
        sys.exit(1)
    
    # Run the main test runner
    main() 