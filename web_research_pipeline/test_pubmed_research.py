#!/usr/bin/env python3
"""
Test script for PubMed Research System
"""

import os
import sys
import json
from datetime import datetime

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_pubmed_research_system():
    """Test the PubMed research system"""
    
    print("=== Testing PubMed Research System ===")
    
    # Test 1: Check if module can be imported
    try:
        from pubmed_research_system import (
            PubMedAPI,
            ResearchQueryGenerator,
            ResearchKnowledgeBase,
            PubMedResearchSystem,
            run_research_analysis,
            get_research_context
        )
        print("✓ PubMed research module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import PubMed research module: {e}")
        return False
    
    # Test 2: Check PubMed email configuration
    pubmed_email = os.getenv("PUBMED_EMAIL", "")
    if pubmed_email:
        print(f"✓ PubMed email configured: {pubmed_email}")
    else:
        print("⚠ PubMed email not configured - using default")
    
    # Test 3: Test PubMed API
    try:
        print("\n--- Testing PubMed API ---")
        pubmed_api = PubMedAPI()
        
        # Test search with a simple query
        test_query = "machine learning healthcare"
        print(f"Searching PubMed for: {test_query}")
        
        papers = pubmed_api.search_papers(test_query, max_results=2, days_back=365)
        
        if papers:
            print(f"✓ PubMed search successful - found {len(papers)} papers")
            for i, paper in enumerate(papers, 1):
                print(f"  Paper {i}: {paper.title}")
                print(f"    Journal: {paper.journal}")
                print(f"    Authors: {', '.join(paper.authors[:2])}")
                print(f"    Relevance: {paper.relevance_score:.2f}")
        else:
            print("⚠ PubMed search returned no results")
            
    except Exception as e:
        print(f"✗ PubMed API test failed: {e}")
    
    # Test 4: Test query generation
    try:
        print("\n--- Testing Query Generation ---")
        query_gen = ResearchQueryGenerator()
        
        # Sample knowledge base content
        sample_content = """
        The patient was diagnosed with diabetes mellitus type 2. 
        Treatment included metformin and lifestyle modifications.
        Machine learning algorithms were used to analyze patient data.
        Clinical trials showed improved outcomes with the new treatment protocol.
        """
        
        queries = query_gen.generate_research_queries(sample_content, max_queries=3)
        
        if queries:
            print(f"✓ Query generation successful - generated {len(queries)} queries:")
            for i, query in enumerate(queries, 1):
                print(f"  Query {i}: {query}")
        else:
            print("⚠ No queries generated")
            
    except Exception as e:
        print(f"✗ Query generation test failed: {e}")
    
    # Test 5: Test research knowledge base
    try:
        print("\n--- Testing Research Knowledge Base ---")
        
        # Create a temporary research KB
        test_kb_path = "test_research_kb"
        research_kb = ResearchKnowledgeBase(test_kb_path)
        
        print(f"✓ Research KB created at: {research_kb.base_path}")
        print(f"  Total papers: {research_kb.metadata['total_papers']}")
        
        # Test adding a sample paper
        from pubmed_research_system import PubMedPaper
        
        sample_paper = PubMedPaper(
            pmid="12345678",
            title="Test Research Paper",
            abstract="This is a test abstract for a research paper.",
            authors=["Test Author 1", "Test Author 2"],
            journal="Test Journal",
            publication_date="2024",
            doi="10.1234/test.2024.001",
            keywords=["test", "research"],
            relevance_score=0.8
        )
        
        added = research_kb.add_paper(sample_paper)
        if added:
            print("✓ Sample paper added successfully")
            print(f"  Total papers: {research_kb.metadata['total_papers']}")
        else:
            print("⚠ Sample paper not added (may already exist)")
        
        # Test querying papers
        papers = research_kb.get_papers_by_query("test", max_results=2)
        print(f"  Papers matching 'test': {len(papers)}")
        
        # Clean up test KB
        import shutil
        if os.path.exists(test_kb_path):
            shutil.rmtree(test_kb_path)
            print("✓ Test research KB cleaned up")
            
    except Exception as e:
        print(f"✗ Research KB test failed: {e}")
    
    # Test 6: Test research system integration
    try:
        print("\n--- Testing Research System Integration ---")
        
        # Create research system
        research_system = PubMedResearchSystem()
        print("✓ Research system initialized")
        
        # Test research context generation
        user_query = "diabetes treatment"
        context = research_system.get_research_context_for_query(user_query, max_papers=2)
        
        if context:
            print("✓ Research context generation successful")
            print(f"  Context length: {len(context)} characters")
        else:
            print("⚠ No research context generated (no matching papers)")
            
    except Exception as e:
        print(f"✗ Research system integration test failed: {e}")
    
    # Test 7: Test manual research analysis (if vision KB exists)
    try:
        print("\n--- Testing Manual Research Analysis ---")
        
        # Check if vision KB exists
        possible_kb_paths = [
            os.path.join(os.getcwd(), 'uploads', 'knowledge_bases'),
            os.path.join(os.getcwd(), 'data', 'knowledge_bases'),
            os.path.join(os.getcwd(), 'knowledge_bases')
        ]
        
        vision_kb_found = False
        for path in possible_kb_paths:
            if os.path.exists(path):
                print(f"✓ Found vision KB at: {path}")
                vision_kb_found = True
                
                # Test research analysis
                print("Running research analysis...")
                result = run_research_analysis(path)
                
                print(f"✓ Research analysis completed")
                print(f"  Status: {result.get('status', 'unknown')}")
                print(f"  Papers added: {result.get('total_papers_added', 0)}")
                print(f"  Queries searched: {result.get('queries_searched', 0)}")
                
                break
        
        if not vision_kb_found:
            print("⚠ No vision KB found - skipping research analysis test")
            
    except Exception as e:
        print(f"✗ Manual research analysis test failed: {e}")
    
    print("\n=== PubMed Research System Test Complete ===")
    return True

def test_research_context_integration():
    """Test how research context would integrate with vision chat"""
    
    print("\n=== Testing Research Context Integration ===")
    
    try:
        from pubmed_research_system import get_research_context
        
        # Simulate a user query
        user_query = "What are the latest treatments for diabetes?"
        
        print(f"User Query: {user_query}")
        
        # Get research context
        research_context = get_research_context(user_query, max_papers=3)
        
        if research_context:
            print("✓ Research context retrieved successfully")
            print(f"Context preview: {research_context[:200]}...")
            
            # Simulate how this would be integrated with vision chat
            print("\n--- Integration Simulation ---")
            print("This research context would be combined with:")
            print("1. Vision chat knowledge base content")
            print("2. Web search results (if enabled)")
            print("3. User query")
            print("4. Passed to LLM for comprehensive response")
            
        else:
            print("⚠ No research context available")
            
    except Exception as e:
        print(f"✗ Research context integration test failed: {e}")

def main():
    """Run all tests"""
    
    print("Starting PubMed Research System Tests...")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    # Run tests
    research_system_success = test_pubmed_research_system()
    integration_success = test_research_context_integration()
    
    print(f"\nTest Summary:")
    print(f"  PubMed Research System: {'✓ PASS' if research_system_success else '✗ FAIL'}")
    print(f"  Research Integration: {'✓ PASS' if integration_success else '✗ FAIL'}")
    
    if research_system_success and integration_success:
        print("\n🎉 All tests passed! PubMed research system is working correctly.")
        print("\nNext steps:")
        print("1. Set PUBMED_EMAIL in your .env file for better API access")
        print("2. Run research analysis manually: python -c 'from pubmed_research_system import run_research_analysis; run_research_analysis()'")
        print("3. Schedule daily research: python -c 'from pubmed_research_system import schedule_research_analysis; schedule_research_analysis()'")
    else:
        print("\n❌ Some tests failed. Please check the configuration and dependencies.")

if __name__ == "__main__":
    main() 