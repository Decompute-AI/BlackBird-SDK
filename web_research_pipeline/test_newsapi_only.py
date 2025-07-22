"""
Simple test script for News API integration

This script tests just the News API provider to ensure it's working correctly.
"""

import os
import sys
from datetime import datetime

# Add the parent directory to the path to import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

def test_newsapi_integration():
    """Test the News API integration specifically"""
    
    print("=== Testing News API Integration ===")
    
    # Check if News API key is configured
    newsapi_key = os.getenv("NEWSAPI_KEY", "")
    if not newsapi_key:
        print("❌ NEWSAPI_KEY not found in environment variables")
        print("Please set NEWSAPI_KEY in your .env file or environment")
        return False
    
    print("✓ News API key found")
    
    # Test 1: Import News API provider
    try:
        from web_research_pipeline.search_providers import NewsAPIProvider, SearchResult
        print("✓ News API provider imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import News API provider: {e}")
        return False
    
    # Test 2: Initialize News API provider
    try:
        news_provider = NewsAPIProvider(api_key=newsapi_key)
        print("✓ News API provider initialized successfully")
        
        # Check if provider is available
        if news_provider.is_available():
            print("✓ News API provider is available")
        else:
            print("✗ News API provider is not available")
            return False
            
    except Exception as e:
        print(f"✗ Failed to initialize News API provider: {e}")
        return False
    
    # Test 3: Get provider info
    try:
        info = news_provider.get_provider_info()
        print(f"✓ Provider info: {info['name']} - {info['description']}")
        print(f"  Features: {', '.join(info['features'])}")
    except Exception as e:
        print(f"✗ Failed to get provider info: {e}")
    
    # Test 4: Test basic search
    try:
        print("\n--- Testing Basic Search ---")
        query = "artificial intelligence"
        results = news_provider.search(query, max_results=3)
        
        if results:
            print(f"✓ Search successful: {len(results)} results")
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result.title[:60]}...")
                print(f"     Source: {result.source}, Score: {result.relevance_score:.2f}")
                if result.metadata.get('source_name'):
                    print(f"     Publisher: {result.metadata['source_name']}")
                if result.metadata.get('published_at'):
                    print(f"     Published: {result.metadata['published_at']}")
                print()
        else:
            print("⚠ Search returned no results")
            
    except Exception as e:
        print(f"✗ Basic search failed: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
    
    # Test 5: Test search with parameters
    try:
        print("--- Testing Search with Parameters ---")
        results = news_provider.search(
            "technology news",
            max_results=2,
            language="en",
            sort_by="publishedAt",
            search_in="title,description"
        )
        
        if results:
            print(f"✓ Parameterized search successful: {len(results)} results")
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result.title[:50]}...")
        else:
            print("⚠ Parameterized search returned no results")
            
    except Exception as e:
        print(f"✗ Parameterized search failed: {e}")
    
    # Test 6: Test top headlines
    try:
        print("--- Testing Top Headlines ---")
        headlines = news_provider.search_top_headlines(
            country="us",
            category="technology",
            max_results=3
        )
        
        if headlines:
            print(f"✓ Top headlines successful: {len(headlines)} results")
            for i, headline in enumerate(headlines, 1):
                print(f"  {i}. {headline.title[:50]}...")
                if headline.metadata.get('source_name'):
                    print(f"     Source: {headline.metadata['source_name']}")
        else:
            print("⚠ Top headlines returned no results")
            
    except Exception as e:
        print(f"✗ Top headlines failed: {e}")
    
    # Test 7: Test enhanced search manager integration
    try:
        print("\n--- Testing Enhanced Search Manager Integration ---")
        from web_research_pipeline.enhanced_search_manager import get_enhanced_search_manager
        
        manager = get_enhanced_search_manager()
        
        # Check if News API is registered
        status = manager.get_provider_status()
        if 'newsapi' in status['configured_providers']:
            print("✓ News API provider registered in enhanced manager")
            
            # Test search through manager
            results = manager.search_web(
                "latest tech news",
                max_results=2,
                provider_name="newsapi"
            )
            
            if results:
                print(f"✓ Manager search successful: {len(results)} results")
                for i, result in enumerate(results, 1):
                    print(f"  {i}. {result.title[:50]}...")
            else:
                print("⚠ Manager search returned no results")
        else:
            print("✗ News API provider not registered in enhanced manager")
            
    except Exception as e:
        print(f"✗ Enhanced manager integration failed: {e}")
    
    print("\n=== News API Integration Tests Complete ===")
    return True

def test_newsapi_error_handling():
    """Test error handling for News API"""
    print("\n=== Testing Error Handling ===")
    
    try:
        from web_research_pipeline.search_providers import NewsAPIProvider
        
        # Test with invalid API key
        invalid_provider = NewsAPIProvider(api_key="invalid_key")
        
        if not invalid_provider.is_available():
            print("✓ Invalid API key properly detected")
        
        # Test search with invalid provider
        results = invalid_provider.search("test query", max_results=1)
        if not results:
            print("✓ Invalid provider properly returns empty results")
        
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")

def main():
    """Main test function"""
    print("📰 News API Integration Test")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check environment
    print("Environment Check:")
    print(f"  - NEWSAPI_KEY: {'✓ Set' if os.getenv('NEWSAPI_KEY') else '✗ Not set'}")
    print(f"  - Current directory: {os.getcwd()}")
    print()
    
    # Run tests
    success = test_newsapi_integration()
    test_newsapi_error_handling()
    
    # Print summary
    print("\n" + "="*50)
    if success:
        print("✅ News API integration tests completed successfully!")
        print("\n💡 Next Steps:")
        print("1. Test with different queries")
        print("2. Experiment with different parameters")
        print("3. Integrate with your existing pipeline")
        print("4. Configure additional News API features")
    else:
        print("❌ News API integration tests failed.")
        print("\n🔧 Troubleshooting:")
        print("1. Verify NEWSAPI_KEY is set correctly")
        print("2. Check internet connectivity")
        print("3. Verify News API service status")
        print("4. Check API rate limits")

if __name__ == "__main__":
    main() 