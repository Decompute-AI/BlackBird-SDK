#!/usr/bin/env python3
"""
Comprehensive test script for Perplexity web search API endpoints
Tests both search query generation and Perplexity API integration
"""

import os
import sys
import json
import time
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

class PerplexityAPITester:
    """Comprehensive tester for Perplexity web search API"""
    
    def __init__(self):
        self.base_url = "http://localhost:5012"  # Correct Flask backend URL
        self.test_results = []
        
        # Load environment variables from .env file
        self._load_env_file()
        self.api_key = os.getenv("PERPLEXITY_API_KEY")
        
        # Test queries for different scenarios
        self.test_queries = [
            # General knowledge queries
            "What is machine learning?",
            "Latest developments in renewable energy",
            "How does blockchain technology work?",
            
            # Medical/health queries
            "COVID-19 vaccine effectiveness",
            "Diabetes treatment options",
            "Mental health benefits of exercise",
            
            # Technical queries
            "Python async programming best practices",
            "Docker containerization tutorial",
            "REST API design principles",
            
            # Current events
            "Latest space exploration missions",
            "Climate change impact 2024",
            "Artificial intelligence regulations",
            
            # Complex queries
            "Compare React vs Vue.js for web development",
            "Best practices for microservices architecture",
            "Machine learning applications in healthcare"
        ]
        
        # Knowledge base context examples
        self.kb_contexts = [
            "User has uploaded medical documents about diabetes treatment.",
            "Technical documentation about Python programming and web development.",
            "Research papers about renewable energy and climate change.",
            "Business documents about blockchain and cryptocurrency.",
            "Educational materials about machine learning and AI."
        ]
    
    def _load_env_file(self):
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
    
    def log_test(self, test_name: str, success: bool, details: str = "", data: Any = None):
        """Log test results"""
        result = {
            'test_name': test_name,
            'success': success,
            'timestamp': datetime.now().isoformat(),
            'details': details,
            'data': data
        }
        self.test_results.append(result)
        
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if details:
            print(f"   Details: {details}")
        if data and not success:
            print(f"   Data: {json.dumps(data, indent=2)[:200]}...")
        print()
    
    def test_api_connectivity(self) -> bool:
        """Test basic API connectivity"""
        try:
            response = requests.get(f"{self.base_url}/vision-chat/web-search-status", timeout=10)
            if response.status_code == 200:
                data = response.json()
                self.log_test("API Connectivity", True, f"Status: {data.get('web_search_available', False)}")
                return data.get('web_search_available', False)
            else:
                self.log_test("API Connectivity", False, f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("API Connectivity", False, f"Connection error: {str(e)}")
            return False
    
    def test_perplexity_api_key(self) -> bool:
        """Test if Perplexity API key is configured"""
        if not self.api_key:
            self.log_test("Perplexity API Key", False, "API key not found in environment")
            return False
        
        if len(self.api_key) < 10:
            self.log_test("Perplexity API Key", False, "API key appears to be invalid (too short)")
            return False
        
        self.log_test("Perplexity API Key", True, f"API key configured (length: {len(self.api_key)})")
        return True
    
    def test_web_search_status_endpoint(self) -> bool:
        """Test the web search status endpoint"""
        try:
            response = requests.get(f"{self.base_url}/vision-chat/web-search-status", timeout=10)
            
            if response.status_code != 200:
                self.log_test("Web Search Status Endpoint", False, f"HTTP {response.status_code}")
                return False
            
            data = response.json()
            required_fields = ['web_search_available', 'perplexity_api_configured', 'web_search_enabled']
            
            for field in required_fields:
                if field not in data:
                    self.log_test("Web Search Status Endpoint", False, f"Missing field: {field}")
                    return False
            
            self.log_test("Web Search Status Endpoint", True, f"Available: {data['web_search_available']}")
            return True
            
        except Exception as e:
            self.log_test("Web Search Status Endpoint", False, f"Error: {str(e)}")
            return False
    
    def test_web_search_test_endpoint(self) -> bool:
        """Test the web search test endpoint"""
        try:
            test_query = "test query"
            payload = {
                'query': test_query,
                'max_results': 2
            }
            
            response = requests.post(
                f"{self.base_url}/vision-chat/web-search/test",
                json=payload,
                timeout=30
            )
            
            if response.status_code != 200:
                self.log_test("Web Search Test Endpoint", False, f"HTTP {response.status_code}")
                return False
            
            data = response.json()
            
            if data.get('status') != 'success':
                self.log_test("Web Search Test Endpoint", False, f"Status: {data.get('status')}")
                return False
            
            results = data.get('results', [])
            if not results:
                self.log_test("Web Search Test Endpoint", False, "No results returned")
                return False
            
            self.log_test("Web Search Test Endpoint", True, f"Found {len(results)} results")
            return True
            
        except Exception as e:
            self.log_test("Web Search Test Endpoint", False, f"Error: {str(e)}")
            return False
    
    def test_web_search_cache_endpoint(self) -> bool:
        """Test the web search cache clear endpoint"""
        try:
            response = requests.post(f"{self.base_url}/vision-chat/web-search/clear-cache", timeout=10)
            
            if response.status_code != 200:
                self.log_test("Web Search Cache Clear Endpoint", False, f"HTTP {response.status_code}")
                return False
            
            data = response.json()
            if data.get('status') != 'success':
                self.log_test("Web Search Cache Clear Endpoint", False, f"Status: {data.get('status')}")
                return False
            
            self.log_test("Web Search Cache Clear Endpoint", True, "Cache cleared successfully")
            return True
            
        except Exception as e:
            self.log_test("Web Search Cache Clear Endpoint", False, f"Error: {str(e)}")
            return False
    
    def test_search_query_generation(self) -> bool:
        """Test search query generation from user queries"""
        try:
            # Test with different types of queries
            test_cases = [
                {
                    'user_query': 'What is machine learning?',
                    'kb_context': 'User has technical documents about AI and programming.',
                    'expected_keywords': ['machine learning', 'AI', 'technology']
                },
                {
                    'user_query': 'Diabetes treatment options',
                    'kb_context': 'Medical documents about diabetes and healthcare.',
                    'expected_keywords': ['diabetes', 'treatment', 'medical', 'health']
                },
                {
                    'user_query': 'Latest renewable energy developments',
                    'kb_context': 'Research papers about climate change and energy.',
                    'expected_keywords': ['renewable energy', 'climate', 'research']
                }
            ]
            
            success_count = 0
            for i, test_case in enumerate(test_cases):
                try:
                    # This would test the actual query generation logic
                    # For now, we'll simulate it
                    generated_query = self._simulate_query_generation(
                        test_case['user_query'], 
                        test_case['kb_context']
                    )
                    
                    # Check if generated query contains expected keywords
                    query_lower = generated_query.lower()
                    expected_keywords = test_case['expected_keywords']
                    keyword_matches = sum(1 for keyword in expected_keywords if keyword.lower() in query_lower)
                    
                    if keyword_matches >= len(expected_keywords) * 0.5:  # At least 50% match
                        success_count += 1
                        self.log_test(f"Query Generation Test {i+1}", True, f"Generated: {generated_query[:50]}...")
                    else:
                        self.log_test(f"Query Generation Test {i+1}", False, f"Generated: {generated_query[:50]}...")
                        
                except Exception as e:
                    self.log_test(f"Query Generation Test {i+1}", False, f"Error: {str(e)}")
            
            overall_success = success_count >= len(test_cases) * 0.7  # At least 70% success rate
            self.log_test("Search Query Generation", overall_success, f"{success_count}/{len(test_cases)} tests passed")
            return overall_success
            
        except Exception as e:
            self.log_test("Search Query Generation", False, f"Error: {str(e)}")
            return False
    
    def _simulate_query_generation(self, user_query: str, kb_context: str) -> str:
        """Simulate query generation (replace with actual implementation)"""
        # This is a simple simulation - replace with actual query generation logic
        enhanced_query = f"{user_query} {kb_context.split()[0]} {kb_context.split()[1]}"
        return enhanced_query
    
    def test_perplexity_api_direct(self) -> bool:
        """Test Perplexity API directly"""
        if not self.api_key:
            self.log_test("Perplexity API Direct", False, "No API key available")
            return False
        
        try:
            # Test with a simple query
            test_query = "What is artificial intelligence?"
            
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'model': 'sonar',
                'messages': [
                    {
                        'role': 'user',
                        'content': test_query
                    }
                ],
                'max_tokens': 100,
                'temperature': 0.1
            }
            
            response = requests.post(
                'https://api.perplexity.ai/chat/completions',
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code != 200:
                self.log_test("Perplexity API Direct", False, f"HTTP {response.status_code}: {response.text}")
                return False
            
            data = response.json()
            if 'choices' not in data or not data['choices']:
                self.log_test("Perplexity API Direct", False, "No choices in response")
                return False
            
            content = data['choices'][0]['message']['content']
            if not content or len(content) < 10:
                self.log_test("Perplexity API Direct", False, "Empty or too short response")
                return False
            
            self.log_test("Perplexity API Direct", True, f"Response length: {len(content)} characters")
            return True
            
        except Exception as e:
            self.log_test("Perplexity API Direct", False, f"Error: {str(e)}")
            return False
    
    def test_web_search_integration(self) -> bool:
        """Test the complete web search integration"""
        try:
            # Test with a real query
            test_query = "latest developments in renewable energy"
            kb_context = "User has research documents about climate change and energy systems."
            
            payload = {
                'query': test_query,
                'kb_context': kb_context,
                'max_results': 3
            }
            
            response = requests.post(
                f"{self.base_url}/vision-chat/web-search/test",
                json=payload,
                timeout=60
            )
            
            if response.status_code != 200:
                self.log_test("Web Search Integration", False, f"HTTP {response.status_code}")
                return False
            
            data = response.json()
            
            if data.get('status') != 'success':
                self.log_test("Web Search Integration", False, f"Status: {data.get('status')}")
                return False
            
            results = data.get('results', [])
            if not results:
                self.log_test("Web Search Integration", False, "No search results")
                return False
            
            # Validate result structure
            for result in results:
                required_fields = ['title', 'content', 'url']
                for field in required_fields:
                    if field not in result:
                        self.log_test("Web Search Integration", False, f"Missing field in result: {field}")
                        return False
            
            self.log_test("Web Search Integration", True, f"Found {len(results)} valid results")
            return True
            
        except Exception as e:
            self.log_test("Web Search Integration", False, f"Error: {str(e)}")
            return False
    
    def test_error_handling(self) -> bool:
        """Test error handling scenarios"""
        test_cases = [
            {
                'name': 'Empty Query',
                'payload': {'query': '', 'max_results': 3},
                'expected_status': 400
            },
            {
                'name': 'Invalid Max Results',
                'payload': {'query': 'test', 'max_results': -1},
                'expected_status': 400
            },
            {
                'name': 'Missing Query',
                'payload': {'max_results': 3},
                'expected_status': 400
            },
            {
                'name': 'Very Long Query',
                'payload': {'query': 'a' * 1000, 'max_results': 3},
                'expected_status': 200  # Should handle long queries gracefully
            }
        ]
        
        success_count = 0
        for test_case in test_cases:
            try:
                response = requests.post(
                    f"{self.base_url}/vision-chat/web-search/test",
                    json=test_case['payload'],
                    timeout=30
                )
                
                if response.status_code == test_case['expected_status']:
                    success_count += 1
                    self.log_test(f"Error Handling - {test_case['name']}", True, f"Status: {response.status_code}")
                else:
                    self.log_test(f"Error Handling - {test_case['name']}", False, f"Expected {test_case['expected_status']}, got {response.status_code}")
                    
            except Exception as e:
                self.log_test(f"Error Handling - {test_case['name']}", False, f"Exception: {str(e)}")
        
        overall_success = success_count >= len(test_cases) * 0.7
        self.log_test("Error Handling", overall_success, f"{success_count}/{len(test_cases)} tests passed")
        return overall_success
    
    def test_performance(self) -> bool:
        """Test performance with multiple concurrent requests"""
        try:
            import threading
            import concurrent.futures
            
            def make_request(query):
                try:
                    payload = {'query': query, 'max_results': 2}
                    response = requests.post(
                        f"{self.base_url}/vision-chat/web-search/test",
                        json=payload,
                        timeout=30
                    )
                    return response.status_code == 200
                except:
                    return False
            
            # Test with 5 concurrent requests
            test_queries = self.test_queries[:5]
            start_time = time.time()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                results = list(executor.map(make_request, test_queries))
            
            end_time = time.time()
            duration = end_time - start_time
            success_count = sum(results)
            
            if success_count >= 3 and duration < 60:  # At least 3 successful, under 60 seconds
                self.log_test("Performance Test", True, f"{success_count}/5 requests successful in {duration:.2f}s")
                return True
            else:
                self.log_test("Performance Test", False, f"{success_count}/5 requests successful in {duration:.2f}s")
                return False
                
        except Exception as e:
            self.log_test("Performance Test", False, f"Error: {str(e)}")
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and return comprehensive results"""
        print("🚀 Starting Perplexity Web Search API Tests")
        print("=" * 60)
        print()
        
        # Reset test results
        self.test_results = []
        
        # Run tests in logical order
        tests = [
            ("API Connectivity", self.test_api_connectivity),
            ("Perplexity API Key", self.test_perplexity_api_key),
            ("Web Search Status Endpoint", self.test_web_search_status_endpoint),
            ("Web Search Test Endpoint", self.test_web_search_test_endpoint),
            ("Web Search Cache Endpoint", self.test_web_search_cache_endpoint),
            ("Search Query Generation", self.test_search_query_generation),
            ("Perplexity API Direct", self.test_perplexity_api_direct),
            ("Web Search Integration", self.test_web_search_integration),
            ("Error Handling", self.test_error_handling),
            ("Performance Test", self.test_performance)
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, test_func in tests:
            try:
                if test_func():
                    passed_tests += 1
            except Exception as e:
                self.log_test(test_name, False, f"Test exception: {str(e)}")
        
        # Generate summary
        success_rate = (passed_tests / total_tests) * 100
        
        summary = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate': success_rate,
            'timestamp': datetime.now().isoformat(),
            'test_results': self.test_results
        }
        
        # Print summary
        print("=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {success_rate:.1f}%")
        print()
        
        if success_rate >= 80:
            print("🎉 Overall Status: EXCELLENT")
        elif success_rate >= 60:
            print("✅ Overall Status: GOOD")
        elif success_rate >= 40:
            print("⚠️ Overall Status: FAIR")
        else:
            print("❌ Overall Status: POOR")
        
        return summary
    
    def save_results(self, results: Dict[str, Any], filename: str = None):
        """Save test results to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"perplexity_api_test_results_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"📄 Test results saved to: {filename}")
        except Exception as e:
            print(f"❌ Error saving results: {e}")

def main():
    """Main function"""
    print("🔍 Perplexity Web Search API Comprehensive Test Suite")
    print("This script tests the complete web search functionality.")
    print()
    
    # Check if backend is running
    tester = PerplexityAPITester()
    
    # Check if we can connect to the backend
    if not tester.test_api_connectivity():
        print("❌ Cannot connect to backend. Make sure the Flask app is running on http://localhost:5012")
        print("   Start the backend with: python decompute.py in sdk/blackbird_sdk/backends/windows/")
        return
    
    # Run all tests
    results = tester.run_all_tests()
    
    # Save results
    tester.save_results(results)
    
    # Print recommendations
    print("\n💡 RECOMMENDATIONS:")
    if results['success_rate'] < 60:
        print("- Check Perplexity API key configuration")
        print("- Verify backend is running and accessible")
        print("- Check network connectivity")
        print("- Review error logs for specific issues")
    elif results['success_rate'] < 80:
        print("- Some tests failed, review specific error messages")
        print("- Check API rate limits and quotas")
        print("- Verify query generation logic")
    else:
        print("- Web search API is working well!")
        print("- Consider running performance tests under load")
        print("- Monitor API usage and costs")

if __name__ == "__main__":
    main() 