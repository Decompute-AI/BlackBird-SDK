#!/usr/bin/env python3
"""
Simple script to run PubMed research analysis
"""

import os
import sys
from datetime import datetime

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    """Run PubMed research analysis"""
    
    print("🔬 PubMed Research Analysis")
    print("=" * 50)
    print(f"Started at: {datetime.now().isoformat()}")
    
    try:
        from pubmed_research_system import run_research_analysis
        
        # Check if vision KB exists
        possible_kb_paths = [
            os.path.join(os.getcwd(), 'uploads', 'knowledge_bases'),
            os.path.join(os.getcwd(), 'data', 'knowledge_bases'),
            os.path.join(os.getcwd(), 'knowledge_bases')
        ]
        
        vision_kb_path = None
        for path in possible_kb_paths:
            if os.path.exists(path):
                vision_kb_path = path
                print(f"✓ Found vision KB at: {path}")
                break
        
        if not vision_kb_path:
            print("⚠ No vision KB found. Creating sample content for testing...")
            # Create a sample vision KB for testing
            sample_kb_path = "sample_vision_kb"
            os.makedirs(sample_kb_path, exist_ok=True)
            
            # Create sample metadata
            sample_metadata = {
                "session_id": "test_session",
                "agent_id": "test_agent",
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "documents": {},
                "images": {},
                "processed_content": {
                    "sample_doc": {
                        "type": "document",
                        "file": "sample_content.json",
                        "processed_at": datetime.now().isoformat()
                    }
                }
            }
            
            # Create sample content
            sample_content = {
                "doc_id": "sample_doc",
                "type": "document",
                "content": """
                The patient was diagnosed with diabetes mellitus type 2. 
                Treatment included metformin and lifestyle modifications.
                Machine learning algorithms were used to analyze patient data.
                Clinical trials showed improved outcomes with the new treatment protocol.
                Recent research indicates potential benefits of GLP-1 receptor agonists.
                """,
                "summary": "Sample medical content for research analysis",
                "processed_at": datetime.now().isoformat()
            }
            
            # Save files
            with open(os.path.join(sample_kb_path, "metadata.json"), 'w') as f:
                import json
                json.dump(sample_metadata, f, indent=2)
            
            with open(os.path.join(sample_kb_path, "sample_content.json"), 'w') as f:
                json.dump(sample_content, f, indent=2)
            
            vision_kb_path = sample_kb_path
            print(f"✓ Created sample vision KB at: {vision_kb_path}")
        
        # Run research analysis
        print("\n🔍 Starting research analysis...")
        result = run_research_analysis(vision_kb_path)
        
        # Display results
        print("\n📊 Research Analysis Results:")
        print(f"  Status: {result.get('status', 'unknown')}")
        print(f"  Total papers added: {result.get('total_papers_added', 0)}")
        print(f"  Queries searched: {result.get('queries_searched', 0)}")
        print(f"  Timestamp: {result.get('timestamp', 'unknown')}")
        
        if result.get('search_results'):
            print("\n📋 Search Results:")
            for i, search_result in enumerate(result['search_results'], 1):
                print(f"  Query {i}: {search_result.get('query', 'unknown')}")
                print(f"    Papers found: {search_result.get('papers_found', 0)}")
                print(f"    Papers added: {search_result.get('papers_added', 0)}")
        
        if result.get('status') == 'success':
            print("\n✅ Research analysis completed successfully!")
            
            # Show research KB location
            from pubmed_research_system import get_research_system
            system = get_research_system(vision_kb_path)
            research_kb_path = system.research_kb.base_path
            print(f"📁 Research papers stored in: {research_kb_path}")
            
        elif result.get('status') == 'no_content':
            print("\n⚠ No content found in vision KB for analysis")
        elif result.get('status') == 'no_queries':
            print("\n⚠ No research queries could be generated")
        else:
            print(f"\n❌ Research analysis failed: {result.get('error', 'Unknown error')}")
        
        # Clean up sample KB if created
        if vision_kb_path == "sample_vision_kb":
            import shutil
            shutil.rmtree(vision_kb_path)
            print("✓ Cleaned up sample vision KB")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you have installed the required dependencies:")
        print("pip install requests schedule")
    except Exception as e:
        print(f"❌ Error running research analysis: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\nCompleted at: {datetime.now().isoformat()}")

if __name__ == "__main__":
    main() 