#!/usr/bin/env python3
"""
System-level scheduling for PubMed research analysis
Works independently of the main application
"""

import os
import sys
import platform
import subprocess
from pathlib import Path
from datetime import datetime

def setup_windows_task_scheduler():
    """Setup Windows Task Scheduler for nightly research analysis"""
    try:
        # Get the path to the research script
        current_dir = Path(__file__).parent
        research_script = current_dir / "run_research.py"
        
        # Create the command
        python_exe = sys.executable
        command = f'"{python_exe}" "{research_script}"'
        
        # Task name and description
        task_name = "PubMedResearchAnalysis"
        task_description = "Runs PubMed research analysis nightly at 2 AM"
        
        # Create the schtasks command
        schtasks_cmd = [
            "schtasks", "/create", "/tn", task_name,
            "/tr", command,
            "/sc", "daily",
            "/st", "02:00",
            "/f"  # Force creation (overwrite if exists)
        ]
        
        # Execute the command
        result = subprocess.run(schtasks_cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Windows Task Scheduler setup successful!")
            print(f"   Task Name: {task_name}")
            print(f"   Schedule: Daily at 2:00 AM")
            print(f"   Command: {command}")
            return True
        else:
            print(f"❌ Failed to setup Windows Task Scheduler:")
            print(f"   Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error setting up Windows Task Scheduler: {e}")
        return False

def setup_linux_cron():
    """Setup Linux cron job for nightly research analysis"""
    try:
        # Get the path to the research script
        current_dir = Path(__file__).parent
        research_script = current_dir / "run_research.py"
        
        # Create the cron entry (run at 2 AM daily)
        cron_entry = f"0 2 * * * {sys.executable} {research_script}"
        
        # Add to crontab
        result = subprocess.run(
            ["crontab", "-l"], 
            capture_output=True, 
            text=True
        )
        
        current_crontab = result.stdout if result.returncode == 0 else ""
        
        # Check if entry already exists
        if cron_entry not in current_crontab:
            # Add new entry
            new_crontab = current_crontab + "\n" + cron_entry + "\n"
            
            # Write to temporary file
            temp_cron_file = "/tmp/temp_crontab"
            with open(temp_cron_file, "w") as f:
                f.write(new_crontab)
            
            # Install new crontab
            result = subprocess.run(["crontab", temp_cron_file])
            
            if result.returncode == 0:
                print(f"✅ Linux cron job setup successful!")
                print(f"   Schedule: Daily at 2:00 AM")
                print(f"   Command: {sys.executable} {research_script}")
                return True
            else:
                print(f"❌ Failed to setup Linux cron job")
                return False
        else:
            print(f"✅ Linux cron job already exists!")
            return True
            
    except Exception as e:
        print(f"❌ Error setting up Linux cron job: {e}")
        return False

def setup_macos_launchd():
    """Setup macOS launchd for nightly research analysis"""
    try:
        # Get the path to the research script
        current_dir = Path(__file__).parent
        research_script = current_dir / "run_research.py"
        
        # Create plist content
        plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.decompute.pubmedresearch</string>
    <key>ProgramArguments</key>
    <array>
        <string>{sys.executable}</string>
        <string>{research_script}</string>
    </array>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>2</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>/tmp/pubmed_research.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/pubmed_research_error.log</string>
</dict>
</plist>"""
        
        # Write plist file
        plist_path = os.path.expanduser("~/Library/LaunchAgents/com.decompute.pubmedresearch.plist")
        os.makedirs(os.path.dirname(plist_path), exist_ok=True)
        
        with open(plist_path, "w") as f:
            f.write(plist_content)
        
        # Load the launchd job
        result = subprocess.run(["launchctl", "load", plist_path])
        
        if result.returncode == 0:
            print(f"✅ macOS launchd setup successful!")
            print(f"   Schedule: Daily at 2:00 AM")
            print(f"   Plist: {plist_path}")
            return True
        else:
            print(f"❌ Failed to setup macOS launchd")
            return False
            
    except Exception as e:
        print(f"❌ Error setting up macOS launchd: {e}")
        return False

def setup_system_scheduler():
    """Setup system-level scheduler based on platform"""
    system = platform.system().lower()
    
    print(f"🔧 Setting up system scheduler for {system}...")
    
    if system == "windows":
        return setup_windows_task_scheduler()
    elif system == "linux":
        return setup_linux_cron()
    elif system == "darwin":  # macOS
        return setup_macos_launchd()
    else:
        print(f"❌ Unsupported operating system: {system}")
        return False

def remove_system_scheduler():
    """Remove system-level scheduler"""
    system = platform.system().lower()
    
    print(f"🗑️ Removing system scheduler for {system}...")
    
    try:
        if system == "windows":
            # Remove Windows task
            task_name = "PubMedResearchAnalysis"
            subprocess.run(["schtasks", "/delete", "/tn", task_name, "/f"])
            print(f"✅ Removed Windows task: {task_name}")
            
        elif system == "linux":
            # Remove cron entry
            result = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
            if result.returncode == 0:
                current_crontab = result.stdout
                # Remove our entry
                lines = current_crontab.split('\n')
                filtered_lines = [line for line in lines if "run_research.py" not in line]
                new_crontab = '\n'.join(filtered_lines)
                
                # Write back to crontab
                temp_cron_file = "/tmp/temp_crontab"
                with open(temp_cron_file, "w") as f:
                    f.write(new_crontab)
                subprocess.run(["crontab", temp_cron_file])
                print("✅ Removed Linux cron job")
                
        elif system == "darwin":
            # Remove launchd job
            plist_path = os.path.expanduser("~/Library/LaunchAgents/com.decompute.pubmedresearch.plist")
            if os.path.exists(plist_path):
                subprocess.run(["launchctl", "unload", plist_path])
                os.remove(plist_path)
                print("✅ Removed macOS launchd job")
                
        return True
        
    except Exception as e:
        print(f"❌ Error removing system scheduler: {e}")
        return False

def check_scheduler_status():
    """Check if system scheduler is properly configured"""
    system = platform.system().lower()
    
    print(f"🔍 Checking scheduler status for {system}...")
    
    try:
        if system == "windows":
            task_name = "PubMedResearchAnalysis"
            result = subprocess.run(["schtasks", "/query", "/tn", task_name], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ Windows Task Scheduler is configured")
                print(f"   Task: {task_name}")
                return True
            else:
                print(f"❌ Windows Task Scheduler not configured")
                return False
                
        elif system == "linux":
            result = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
            if result.returncode == 0 and "run_research.py" in result.stdout:
                print(f"✅ Linux cron job is configured")
                return True
            else:
                print(f"❌ Linux cron job not configured")
                return False
                
        elif system == "darwin":
            plist_path = os.path.expanduser("~/Library/LaunchAgents/com.decompute.pubmedresearch.plist")
            if os.path.exists(plist_path):
                print(f"✅ macOS launchd is configured")
                return True
            else:
                print(f"❌ macOS launchd not configured")
                return False
                
    except Exception as e:
        print(f"❌ Error checking scheduler status: {e}")
        return False

def main():
    """Main function to setup system scheduler"""
    print("🚀 PubMed Research System Scheduler Setup")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "setup":
            success = setup_system_scheduler()
            if success:
                print("\n🎉 System scheduler setup completed!")
                print("The research analysis will now run automatically every night at 2 AM.")
            else:
                print("\n❌ System scheduler setup failed!")
                
        elif command == "remove":
            success = remove_system_scheduler()
            if success:
                print("\n✅ System scheduler removed successfully!")
            else:
                print("\n❌ Failed to remove system scheduler!")
                
        elif command == "status":
            check_scheduler_status()
            
        else:
            print(f"❌ Unknown command: {command}")
            print("Available commands: setup, remove, status")
    else:
        print("Usage:")
        print("  python system_scheduler.py setup    # Setup system scheduler")
        print("  python system_scheduler.py remove   # Remove system scheduler")
        print("  python system_scheduler.py status   # Check scheduler status")

if __name__ == "__main__":
    main() 