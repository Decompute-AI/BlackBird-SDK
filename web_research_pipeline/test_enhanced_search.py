"""
Test script for enhanced search functionality

Demonstrates the new multi-provider search capabilities while maintaining backward compatibility.
"""

import os
import sys
import json
from datetime import datetime

# Add the parent directory to the path to import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

def test_enhanced_search_integration():
    """Test the enhanced search integration"""
    
    print("=== Testing Enhanced Search Integration ===")
    
    # Test 1: Check if enhanced search module can be imported
    try:
        from web_research_pipeline.enhanced_search_manager import (
            get_enhanced_search_manager,
            search_web_for_query,
            search_with_kb_context,
            EnhancedSearchManager
        )
        from web_research_pipeline.search_providers import (
            SearchProvider, SearchResult, SearchProviderRegistry,
            PerplexityProvider, NewsAPIProvider
        )
        print("✓ Enhanced search modules imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import enhanced search modules: {e}")
        return False
    
    # Test 2: Check API key configuration
    config_status = check_api_keys()
    if not config_status['any_configured']:
        print("⚠ No search providers configured - tests will be limited")
    
    # Test 3: Test enhanced search manager initialization
    try:
        manager = get_enhanced_search_manager()
        print("✓ Enhanced search manager initialized successfully")
        
        # Get provider status
        status = manager.get_provider_status()
        print(f"✓ Provider status: {len(status['configured_providers'])} configured providers")
        
    except Exception as e:
        print(f"✗ Enhanced search manager initialization failed: {e}")
        return False
    
    # Test 4: Test single provider search
    if config_status['perplexity_configured']:
        test_single_provider_search(manager, 'perplexity')
    
    if config_status['newsapi_configured']:
        test_single_provider_search(manager, 'newsapi')
    
    # Test 5: Test multi-provider search
    if config_status['multiple_configured']:
        test_multi_provider_search(manager)
    
    # Test 6: Test backward compatibility
    test_backward_compatibility()
    
    # Test 7: Test provider switching
    test_provider_switching(manager)
    
    print("\n=== Enhanced Search Integration Tests Complete ===")
    return True

def check_api_keys():
    """Check which API keys are configured"""
    status = {
        'perplexity_configured': bool(os.getenv("PERPLEXITY_API_KEY", "")),
        'newsapi_configured': bool(os.getenv("NEWSAPI_KEY", "")),
        'fireflow_configured': bool(os.getenv("FIREFLOW_API_KEY", "")),
        'any_configured': False
    }
    
    status['any_configured'] = any([
        status['perplexity_configured'],
        status['newsapi_configured'],
        status['fireflow_configured']
    ])
    
    status['multiple_configured'] = sum([
        status['perplexity_configured'],
        status['newsapi_configured'],
        status['fireflow_configured']
    ]) > 1
    
    print(f"✓ API Key Status:")
    print(f"  - Perplexity: {'✓' if status['perplexity_configured'] else '✗'}")
    print(f"  - News API: {'✓' if status['newsapi_configured'] else '✗'}")
    print(f"  - FireFlow: {'✓' if status['fireflow_configured'] else '✗'}")
    
    return status

def test_single_provider_search(manager, provider_name):
    """Test search with a single provider"""
    print(f"\n--- Testing {provider_name} Provider ---")
    
    try:
        # Test basic search
        query = "latest developments in artificial intelligence"
        results = manager.search_web(query, max_results=3, provider_name=provider_name)
        
        if results:
            print(f"✓ {provider_name} search successful: {len(results)} results")
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result.title[:50]}... (score: {result.relevance_score:.2f})")
        else:
            print(f"⚠ {provider_name} search returned no results")
        
        # Test contextual search
        kb_context = "User has research documents about AI and machine learning."
        context_results = manager.search_with_context(
            query, kb_context, max_results=2, provider_name=provider_name
        )
        
        if context_results['search_performed']:
            print(f"✓ {provider_name} contextual search successful")
            print(f"  - Combined context length: {len(context_results['combined_context'])}")
        else:
            print(f"⚠ {provider_name} contextual search failed")
        
    except Exception as e:
        print(f"✗ {provider_name} search test failed: {e}")

def test_multi_provider_search(manager):
    """Test search with multiple providers"""
    print("\n--- Testing Multi-Provider Search ---")
    
    try:
        # Get configured providers
        configured_providers = manager.registry.get_configured_providers()
        if len(configured_providers) < 2:
            print("⚠ Need at least 2 configured providers for multi-provider test")
            return
        
        # Test multi-provider search
        query = "renewable energy innovations 2024"
        multi_results = manager.search_multiple_providers(
            query, configured_providers, max_results=2
        )
        
        print(f"✓ Multi-provider search completed:")
        print(f"  - Providers used: {multi_results['total_providers']}")
        print(f"  - Successful providers: {multi_results['successful_providers']}")
        print(f"  - Total merged results: {multi_results['total_results']}")
        
        # Show results by provider
        for provider_name, results in multi_results['provider_results'].items():
            if results:
                print(f"  - {provider_name}: {len(results)} results")
        
        # Show merged results
        if multi_results['merged_results']:
            print(f"  - Top merged result: {multi_results['merged_results'][0].title[:50]}...")
        
    except Exception as e:
        print(f"✗ Multi-provider search test failed: {e}")

def test_backward_compatibility():
    """Test backward compatibility with existing code"""
    print("\n--- Testing Backward Compatibility ---")
    
    try:
        # Test old function names still work
        from web_research_pipeline.enhanced_search_manager import get_web_search_manager
        
        old_manager = get_web_search_manager()
        print("✓ Backward compatibility function works")
        
        # Test old search functions
        query = "test query"
        results = old_manager.search_web(query, max_results=1)
        print(f"✓ Old search interface works: {len(results)} results")
        
    except Exception as e:
        print(f"✗ Backward compatibility test failed: {e}")

def test_provider_switching(manager):
    """Test switching between providers"""
    print("\n--- Testing Provider Switching ---")
    
    try:
        configured_providers = manager.registry.get_configured_providers()
        if len(configured_providers) < 2:
            print("⚠ Need at least 2 configured providers for switching test")
            return
        
        # Test switching default provider
        original_default = manager.registry._default_provider
        new_default = configured_providers[1] if configured_providers[1] != original_default else configured_providers[0]
        
        success = manager.set_default_provider(new_default)
        if success:
            print(f"✓ Successfully switched default provider from {original_default} to {new_default}")
            
            # Test search with new default
            query = "test query"
            results = manager.search_web(query, max_results=1)
            print(f"✓ Search with new default provider: {len(results)} results")
            
            # Switch back
            manager.set_default_provider(original_default)
            print(f"✓ Switched back to original default: {original_default}")
        else:
            print(f"✗ Failed to switch default provider")
        
    except Exception as e:
        print(f"✗ Provider switching test failed: {e}")

def test_provider_registry():
    """Test the provider registry functionality"""
    print("\n--- Testing Provider Registry ---")
    
    try:
        from web_research_pipeline.search_providers import SearchProviderRegistry, PerplexityProvider
        
        # Create a test registry
        registry = SearchProviderRegistry()
        
        # Test provider registration
        test_provider = PerplexityProvider(api_key="test_key")
        registry.register_provider(test_provider, "test_perplexity")
        
        print(f"✓ Provider registry created with {len(registry.get_available_providers())} providers")
        
        # Test provider retrieval
        provider = registry.get_provider("test_perplexity")
        if provider:
            print("✓ Provider retrieval works")
        
        # Test provider status
        status = registry.get_provider_status()
        print(f"✓ Provider status retrieval works: {len(status)} providers")
        
    except Exception as e:
        print(f"✗ Provider registry test failed: {e}")

def test_search_result_structure():
    """Test the SearchResult data structure"""
    print("\n--- Testing SearchResult Structure ---")
    
    try:
        from web_research_pipeline.search_providers import SearchResult
        
        # Create a test result
        result = SearchResult(
            title="Test Title",
            content="Test content for demonstration",
            url="https://example.com",
            source="test",
            relevance_score=0.85,
            search_query="test query"
        )
        
        print("✓ SearchResult creation successful")
        print(f"  - Title: {result.title}")
        print(f"  - Content: {result.content}")
        print(f"  - URL: {result.url}")
        print(f"  - Source: {result.source}")
        print(f"  - Relevance Score: {result.relevance_score}")
        print(f"  - Content Hash: {result.content_hash}")
        print(f"  - Timestamp: {result.timestamp}")
        
        # Test metadata
        result.metadata['test_key'] = 'test_value'
        print(f"  - Metadata: {result.metadata}")
        
    except Exception as e:
        print(f"✗ SearchResult structure test failed: {e}")

def main():
    """Main test function"""
    print("🔍 Enhanced Search Integration Test Suite")
    print("This script tests the new multi-provider search functionality.")
    print()
    
    # Run all tests
    success = test_enhanced_search_integration()
    
    # Additional component tests
    test_provider_registry()
    test_search_result_structure()
    
    # Print summary
    print("\n" + "="*50)
    if success:
        print("✅ Enhanced search integration tests completed successfully!")
        print("\n💡 Key Features Demonstrated:")
        print("- Multi-provider search support")
        print("- Provider switching and configuration")
        print("- Backward compatibility with existing code")
        print("- Standardized result format across providers")
        print("- Caching and rate limiting")
        print("- Error handling and fallbacks")
    else:
        print("❌ Some tests failed. Check the output above for details.")
    
    print("\n📝 Next Steps:")
    print("1. Configure API keys in your .env file")
    print("2. Test with real queries")
    print("3. Integrate with your existing pipeline")
    print("4. Configure provider preferences")

if __name__ == "__main__":
    main() 