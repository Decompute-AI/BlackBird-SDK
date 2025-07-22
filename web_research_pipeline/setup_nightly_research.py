#!/usr/bin/env python3
"""
User-friendly setup for nightly PubMed research analysis
"""

import os
import sys
from pathlib import Path

def main():
    """Setup nightly research analysis"""
    
    print("🌙 Nightly PubMed Research Setup")
    print("=" * 40)
    print()
    print("This will setup automatic research analysis to run every night at 2 AM.")
    print("The analysis will work even when your app is closed.")
    print()
    
    # Check if user wants to proceed
    response = input("Do you want to setup nightly research analysis? (y/N): ").lower()
    if response != 'y':
        print("Setup cancelled.")
        return
    
    print()
    print("🔧 Setting up system scheduler...")
    
    try:
        # Import and run system scheduler setup
        from system_scheduler import setup_system_scheduler
        
        success = setup_system_scheduler()
        
        if success:
            print()
            print("🎉 Setup completed successfully!")
            print()
            print("What happens next:")
            print("✅ Research analysis will run automatically every night at 2 AM")
            print("✅ Works even when your app is closed")
            print("✅ Analyzes your vision chat knowledge base")
            print("✅ Searches PubMed for relevant research papers")
            print("✅ Stores papers in a separate research knowledge base")
            print()
            print("To manage the scheduler:")
            print("  python system_scheduler.py status   # Check if it's working")
            print("  python system_scheduler.py remove   # Remove the scheduler")
            print()
            print("To run research analysis manually:")
            print("  python run_research.py")
            
        else:
            print()
            print("❌ Setup failed!")
            print()
            print("Alternative options:")
            print("1. Run research analysis manually: python run_research.py")
            print("2. Use the app-level scheduler (only works when app is running)")
            print("3. Check system requirements and try again")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running this from the web_research_pipeline directory.")
    except Exception as e:
        print(f"❌ Setup error: {e}")
        print("Please check your system permissions and try again.")

if __name__ == "__main__":
    main() 