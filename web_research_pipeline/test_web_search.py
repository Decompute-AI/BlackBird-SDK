"""
Test script for web search integration
"""

import os
import sys
import json
from datetime import datetime

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_web_search_integration():
    """Test the web search integration"""
    
    print("=== Testing Web Search Integration ===")
    
    # Test 1: Check if web search module can be imported
    try:
        from llm_web_search import (
            get_web_search_manager,
            search_web_for_query,
            search_with_kb_context,
            WebSearchResult
        )
        print("✓ Web search module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import web search module: {e}")
        return False
    
    # Test 2: Check API key configuration
    api_key = os.getenv("PERPLEXITY_API_KEY", "")
    if api_key:
        print("✓ Perplexity API key found")
    else:
        print("⚠ Perplexity API key not found - web search will be disabled")
    
    # Test 3: Test query generation
    try:
        from llm_web_search import QueryGenerator
        query_gen = QueryGenerator()
        
        test_queries = [
            "What is machine learning?",
            "Latest news about AI",
            "How to implement neural networks"
        ]
        
        for query in test_queries:
            enhanced_query = query_gen.generate_search_query(query)
            print(f"✓ Query enhancement: '{query}' -> '{enhanced_query}'")
            
    except Exception as e:
        print(f"✗ Query generation test failed: {e}")
    
    # Test 4: Test web search manager initialization
    try:
        manager = get_web_search_manager()
        print("✓ Web search manager initialized successfully")
    except Exception as e:
        print(f"✗ Web search manager initialization failed: {e}")
        return False
    
    # Test 5: Test web search (if API key is available)
    if api_key:
        try:
            print("\n--- Testing Web Search ---")
            test_query = "artificial intelligence trends 2024"
            
            print(f"Searching for: {test_query}")
            results = search_web_for_query(test_query, max_results=2)
            
            if results:
                print(f"✓ Web search successful - found {len(results)} results")
                for i, result in enumerate(results, 1):
                    print(f"  Result {i}: {result.title}")
                    print(f"    Content: {result.content[:100]}...")
                    print(f"    URL: {result.url}")
                    print(f"    Relevance: {result.relevance_score:.2f}")
            else:
                print("⚠ Web search returned no results")
                
        except Exception as e:
            print(f"✗ Web search test failed: {e}")
    
    # Test 6: Test contextual search
    try:
        print("\n--- Testing Contextual Search ---")
        kb_context = "Machine learning is a subset of artificial intelligence that focuses on algorithms and statistical models."
        user_query = "What are the latest developments in machine learning?"
        
        print(f"User query: {user_query}")
        print(f"KB context: {kb_context}")
        
        search_data = search_with_kb_context(user_query, kb_context, max_results=2)
        
        print(f"✓ Contextual search completed")
        print(f"  Search performed: {search_data.get('search_performed', False)}")
        print(f"  Total sources: {search_data.get('total_sources', 0)}")
        print(f"  Combined context length: {len(search_data.get('combined_context', ''))}")
        
    except Exception as e:
        print(f"✗ Contextual search test failed: {e}")
    
    print("\n=== Web Search Integration Test Complete ===")
    return True

def test_vision_chat_integration():
    """Test vision chat integration with web search"""
    
    print("\n=== Testing Vision Chat Integration ===")
    
    try:
        # Import vision chat components
        from blackbird_sdk.backends.windows.routes.vision_chat import (
            WEB_SEARCH_AVAILABLE,
            get_web_search_manager,
            search_with_kb_context
        )
        
        print(f"✓ Vision chat web search available: {WEB_SEARCH_AVAILABLE}")
        
        if WEB_SEARCH_AVAILABLE:
            # Test the integration
            test_query = "What are the latest AI developments?"
            kb_context = "AI has been advancing rapidly in recent years."
            
            search_data = search_with_kb_context(test_query, kb_context, max_results=2)
            
            print(f"✓ Vision chat integration test successful")
            print(f"  Web search performed: {search_data.get('search_performed', False)}")
            print(f"  Results found: {search_data.get('total_sources', 0)}")
            
        else:
            print("⚠ Web search not available in vision chat")
            
    except ImportError as e:
        print(f"✗ Vision chat integration test failed: {e}")
    except Exception as e:
        print(f"✗ Vision chat integration test failed: {e}")

if __name__ == "__main__":
    print("Starting Web Search Integration Tests...")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    # Run tests
    web_search_success = test_web_search_integration()
    vision_chat_success = test_vision_chat_integration()
    
    print(f"\nTest Summary:")
    print(f"  Web Search Integration: {'✓ PASS' if web_search_success else '✗ FAIL'}")
    print(f"  Vision Chat Integration: {'✓ PASS' if vision_chat_success else '✗ FAIL'}")
    
    if web_search_success and vision_chat_success:
        print("\n🎉 All tests passed! Web search integration is working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the configuration and dependencies.") 