"""
Example usage of the web search integration for Vision Chat
"""

import os
import sys
from datetime import datetime

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def example_basic_web_search():
    """Example of basic web search functionality"""
    
    print("=== Basic Web Search Example ===")
    
    try:
        from llm_web_search import search_web_for_query
        
        # Example queries
        queries = [
            "latest AI developments 2024",
            "machine learning applications in healthcare",
            "Python programming best practices"
        ]
        
        for query in queries:
            print(f"\nSearching for: {query}")
            results = search_web_for_query(query, max_results=2)
            
            if results:
                print(f"Found {len(results)} results:")
                for i, result in enumerate(results, 1):
                    print(f"  {i}. {result.title}")
                    print(f"     Content: {result.content[:100]}...")
                    print(f"     URL: {result.url}")
                    print(f"     Relevance: {result.relevance_score:.2f}")
            else:
                print("  No results found")
                
    except ImportError as e:
        print(f"Error importing web search module: {e}")
    except Exception as e:
        print(f"Error in basic web search: {e}")

def example_contextual_search():
    """Example of contextual search with knowledge base"""
    
    print("\n=== Contextual Search Example ===")
    
    try:
        from llm_web_search import search_with_kb_context
        
        # Example knowledge base context
        kb_context = """
        Machine learning is a subset of artificial intelligence that focuses on algorithms 
        and statistical models. Deep learning is a subset of machine learning that uses 
        neural networks with multiple layers. Recent developments include transformer 
        architectures and large language models.
        """
        
        # Example user queries
        user_queries = [
            "What are the latest breakthroughs in deep learning?",
            "How are neural networks being used in healthcare?",
            "What's new in transformer models?"
        ]
        
        for query in user_queries:
            print(f"\nUser Query: {query}")
            print(f"KB Context: {kb_context.strip()}")
            
            search_data = search_with_kb_context(query, kb_context, max_results=2)
            
            print(f"Search performed: {search_data.get('search_performed', False)}")
            print(f"Total sources: {search_data.get('total_sources', 0)}")
            
            if search_data.get('web_results'):
                print("Web search results:")
                for i, result in enumerate(search_data['web_results'], 1):
                    print(f"  {i}. {result.title}")
                    print(f"     {result.content[:80]}...")
            
            print(f"Combined context length: {len(search_data.get('combined_context', ''))}")
            
    except ImportError as e:
        print(f"Error importing web search module: {e}")
    except Exception as e:
        print(f"Error in contextual search: {e}")

def example_query_generation():
    """Example of query generation and enhancement"""
    
    print("\n=== Query Generation Example ===")
    
    try:
        from llm_web_search import QueryGenerator
        
        query_gen = QueryGenerator()
        
        # Example user queries
        user_queries = [
            "What is machine learning?",
            "Can you tell me about the latest AI news?",
            "How does neural network work?",
            "I want to know about Python programming",
            "Explain quantum computing"
        ]
        
        for query in user_queries:
            # Generate enhanced queries for different types
            enhanced_general = query_gen.generate_search_query(query, 'general')
            enhanced_technical = query_gen.generate_search_query(query, 'technical')
            enhanced_news = query_gen.generate_search_query(query, 'news')
            
            print(f"\nOriginal: {query}")
            print(f"General:  {enhanced_general}")
            print(f"Technical: {enhanced_technical}")
            print(f"News:     {enhanced_news}")
            
    except ImportError as e:
        print(f"Error importing query generator: {e}")
    except Exception as e:
        print(f"Error in query generation: {e}")

def example_vision_chat_integration():
    """Example of how the integration works in vision chat"""
    
    print("\n=== Vision Chat Integration Example ===")
    
    try:
        # Simulate the vision chat flow
        user_query = "What are the latest developments in artificial intelligence?"
        
        # Step 1: Knowledge base retrieval (simulated)
        kb_context = """
        [Source 1: PDF_PAGE] (Page 15) [Relevance: 0.85]
        ------------------------------
        Artificial intelligence has evolved significantly over the past decade...
        
        [Source 2: IMAGE] (Image: ai_diagram.png) [Relevance: 0.72]
        ------------------------------
        Neural network architecture diagram showing deep learning layers...
        """
        
        print(f"User Query: {user_query}")
        print(f"Knowledge Base Context: {kb_context.strip()}")
        
        # Step 2: Web search integration
        from llm_web_search import search_with_kb_context
        
        search_data = search_with_kb_context(user_query, kb_context, max_results=3)
        
        if search_data.get('search_performed'):
            print(f"\nWeb Search Results ({search_data.get('total_sources', 0)} sources):")
            
            for i, result in enumerate(search_data.get('web_results', []), 1):
                print(f"  Web Source {i}:")
                print(f"    Title: {result.title}")
                print(f"    Content: {result.content[:150]}...")
                print(f"    URL: {result.url}")
                print(f"    Relevance: {result.relevance_score:.2f}")
        
        # Step 3: Combined context for LLM
        combined_context = search_data.get('combined_context', kb_context)
        print(f"\nCombined Context Length: {len(combined_context)} characters")
        print(f"Context Preview: {combined_context[:200]}...")
        
        # Step 4: LLM would receive this combined context
        print(f"\nLLM would receive the combined context with:")
        print(f"- Knowledge base sources: {kb_context.count('[Source')} sources")
        print(f"- Web search sources: {len(search_data.get('web_results', []))} sources")
        print(f"- Total context length: {len(combined_context)} characters")
        
    except ImportError as e:
        print(f"Error importing web search module: {e}")
    except Exception as e:
        print(f"Error in vision chat integration example: {e}")

def example_error_handling():
    """Example of error handling scenarios"""
    
    print("\n=== Error Handling Example ===")
    
    try:
        from llm_web_search import search_web_for_query
        
        # Test with invalid query
        print("Testing with empty query:")
        results = search_web_for_query("", max_results=2)
        print(f"Results: {len(results)} (should be 0)")
        
        # Test with very long query
        long_query = "a" * 1000
        print(f"\nTesting with very long query ({len(long_query)} characters):")
        results = search_web_for_query(long_query, max_results=2)
        print(f"Results: {len(results)}")
        
        # Test with special characters
        special_query = "What is AI? @#$%^&*()"
        print(f"\nTesting with special characters: {special_query}")
        results = search_web_for_query(special_query, max_results=2)
        print(f"Results: {len(results)}")
        
    except Exception as e:
        print(f"Error in error handling example: {e}")

def main():
    """Run all examples"""
    
    print("Web Search Integration Examples")
    print("=" * 50)
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    # Check if API key is available
    api_key = os.getenv("PERPLEXITY_API_KEY", "")
    if api_key:
        print("✓ Perplexity API key found")
    else:
        print("⚠ Perplexity API key not found - some examples may not work")
    
    # Run examples
    example_basic_web_search()
    example_contextual_search()
    example_query_generation()
    example_vision_chat_integration()
    example_error_handling()
    
    print("\n" + "=" * 50)
    print("Examples completed!")

if __name__ == "__main__":
    main() 