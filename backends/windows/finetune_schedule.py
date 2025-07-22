import requests
import json
import os
import time
import argparse
from datetime import datetime, timedelta

class AgentFineTuner:
    def __init__(self, finetune_api_url="http://127.0.0.1:5012", files_api_url="http://127.0.0.1:5012"):
        self.finetune_api_url = finetune_api_url
        self.files_api_url = files_api_url
        
    def get_agent_files(self, agent_name):
        """Fetch all files for a specific agent"""
        url = f"{self.files_api_url}/api/files/{agent_name}"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json().get("savedFiles", [])
        except requests.exceptions.RequestException as e:
            print(f"Error fetching agent files: {e}")
            return []
    
    def get_model_info(self, file_location):
        """Extract model info from config.json in the file location"""
        config_path = os.path.join(file_location, "config.json")
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                return config
            else:
                print(f"Config file not found at {config_path}")
                return None
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error reading config file: {e}")
            return None
    
    def update_config_with_finetuning_info(self, file_location, epochs):
        """Update config.json with fine-tuning info"""
        config_path = os.path.join(file_location, "config.json")
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # Add/update fine-tuning information
                config["epochs"] = epochs  # This is now the cumulative total
                config["last_finetuned"] = datetime.now().isoformat()
                
                # Add training history for transparency
                if "training_history" not in config:
                    config["training_history"] = []
                
                config["training_history"].append({
                    "date": datetime.now().isoformat(),
                    "epochs": epochs
                })
                
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=4)
                    
                print(f"Updated config file for {file_location} with epochs={epochs}")
                return True
            else:
                print(f"Config file not found at {config_path}")
                return False
        except Exception as e:
            print(f"Error updating config file: {e}")
            return False
    
    def start_finetuning(self, file_path, model_name, epochs=3, learning_rate=1e-5):
        """Start fine-tuning process via API"""
        url = f"{self.finetune_api_url}/api/finetune"
        
        payload = {
            "file_path": file_path,
            "model_name": model_name,
            "epochs": epochs,
            "learning_rate": learning_rate
        }
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            job_data = response.json()
            return job_data.get("job_id")
        except requests.exceptions.RequestException as e:
            print(f"Error starting fine-tuning: {e}")
            if hasattr(e, 'response') and e.response:
                print(f"Response: {e.response.text}")
            return None
    
    def check_finetuning_status(self, job_id):
        """Check status of a fine-tuning job"""
        url = f"{self.finetune_api_url}/api/finetune/status/{job_id}"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error checking fine-tuning status: {e}")
            return None
    
    def wait_for_completion(self, job_id, poll_interval=30, timeout=7200):
        """Wait for fine-tuning job to complete with timeout"""
        start_time = time.time()
        
        while (time.time() - start_time) < timeout:
            status_data = self.check_finetuning_status(job_id)
            
            if not status_data:
                print("Failed to get status update, retrying...")
                time.sleep(poll_interval)
                continue
            
            status = status_data.get("status")
            print(f"Job {job_id} status: {status}")
            
            if status == "completed":
                return True
            elif status == "failed":
                print(f"Job failed: {status_data.get('error')}")
                return False
            
            # Show last progress update if available
            progress = status_data.get("progress", [])
            if progress:
                print(f"Progress: {progress[-1]}")
            
            time.sleep(poll_interval)
        
        print(f"Timeout waiting for job {job_id} to complete")
        return False
    
    def finetune_agent(self, agent_name, epochs_increment=25, learning_rate=1e-5, after_date=None):
        """Fine-tune all files for a specific agent"""
        files = self.get_agent_files(agent_name)
        
        if not files:
            print(f"No files found for agent {agent_name}")
            return []
        
        results = []
        
        # Sort files by date (newest first)
        files = sorted(files, key=lambda x: x.get("date", ""), reverse=True)
        
        print(f"Found {len(files)} files for agent {agent_name}")
        if after_date:
            print(f"Will only process files created after {after_date}")
        
        for file_info in files:
            file_location = file_info.get("location")
            file_date = file_info.get("date")
            
            # Skip files before the given date if specified
            if after_date and file_date < after_date:
                print(f"Skipping {file_location} (created {file_date}, which is before {after_date})")
                continue
            
            # Get model info from config
            config = self.get_model_info(file_location)
            if not config:
                print(f"Skipping {file_location} - no config found")
                continue
            
            model_name = config.get("model")
            if not model_name:
                print(f"Skipping {file_location} - no model specified in config")
                continue
            
            # Check existing epochs value - skip if already 100 or greater
            current_epochs = config.get("epochs", 0)
            if current_epochs >= 100:
                print(f"Skipping {file_location} - already trained for {current_epochs} epochs (≥ 100)")
                results.append({
                    "location": file_location,
                    "success": True,
                    "skipped": True,
                    "reason": "Already reached maximum epochs (100)"
                })
                continue
            
            # Calculate how many epochs to add this run
            # Make sure we don't exceed 100 total epochs
            epochs_to_add = min(epochs_increment, 100 - current_epochs)
            
            print(f"Starting fine-tuning for {file_location} with model {model_name}")
            print(f"Current epochs: {current_epochs}, adding {epochs_to_add} epochs")
            
            # Start fine-tuning
            job_id = self.start_finetuning(file_location, model_name, epochs_to_add, learning_rate)
            if not job_id:
                results.append({
                    "location": file_location,
                    "success": False,
                    "error": "Failed to start fine-tuning job"
                })
                continue
            
            # Wait for completion
            success = self.wait_for_completion(job_id)
            
            # Update config if successful
            if success:
                # Update with cumulative epoch count
                new_total_epochs = current_epochs + epochs_to_add
                self.update_config_with_finetuning_info(file_location, new_total_epochs)
            
            results.append({
                "location": file_location,
                "success": success,
                "job_id": job_id,
                "previous_epochs": current_epochs,
                "epochs_added": epochs_to_add,
                "total_epochs": current_epochs + epochs_to_add if success else current_epochs
            })
        
        return results

# def main():
#     # Calculate default date (2 days ago from now)
#     default_after_date = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
    
#     parser = argparse.ArgumentParser(description="Fine-tune agent models")
#     parser.add_argument("agent_name", help="Name of the agent (general, research, legal)")
#     parser.add_argument("--epochs-increment", type=int, default=25, help="Number of epochs to add per training run (default: 25, max total: 100)")
#     parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate for fine-tuning")
#     parser.add_argument("--after-date", default=default_after_date, 
#                         help=f"Only process files created after this date (YYYY-MM-DD). Default: {default_after_date} (2 days ago)")
    
#     args = parser.parse_args()
    
#     print(f"Processing files created after: {args.after_date}")
    
#     tuner = AgentFineTuner()
#     results = tuner.finetune_agent(
#         args.agent_name,
#         epochs_increment=args.epochs_increment, 
#         learning_rate=args.learning_rate,
#         after_date=args.after_date
#     )
    
#     print("\nFine-tuning results:")
#     for result in results:
#         status = "SUCCESS" if result.get("success") else "FAILED"
#         print(f"{status}: {result.get('location')}")

# if __name__ == "__main__":
#     main()
