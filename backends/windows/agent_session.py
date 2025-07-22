import os
import json
from datetime import datetime
from flask import session, current_app
import time

class AgentSessionManager:
    """
    Manages agent sessions across different requests to enable cumulative
    fine-tuning with saved_files mode when using the same agent.
    """
    SESSION_KEY = 'agent_sessions'
    SESSION_FILE_DIR = os.path.expanduser('~/Documents/Decompute-Files/sessions')
    
    @staticmethod
    def _ensure_session_dir():
        """Ensure session directory exists"""
        os.makedirs(AgentSessionManager.SESSION_FILE_DIR, exist_ok=True)
    
    @staticmethod
    def _get_session_file_path(agent_type):
        """Get path to session file for an agent"""
        AgentSessionManager._ensure_session_dir()
        return os.path.join(AgentSessionManager.SESSION_FILE_DIR, f"{agent_type}_session.json")
    
    @staticmethod
    def get_current_agent_session(agent_type):
        """
        Get the current session for a specific agent.
        Uses filesystem-based persistence instead of Flask session.
        """
        session_file = AgentSessionManager._get_session_file_path(agent_type)
        
        if os.path.exists(session_file):
            try:
                with open(session_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                # If file exists but is corrupt, initialize new session
                pass
        
        # Initialize new agent session
        new_session = {
            'last_weights_path': None,
            'files_processed': [],
            'session_started': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'active': True
        }
        
        # Save new session
        with open(session_file, 'w') as f:
            json.dump(new_session, f, indent=2)
            
        return new_session
    
    @staticmethod
    def update_agent_session(agent_type, weights_path, file_path):
        """Update the session data for an agent after processing a file"""
        session_file = AgentSessionManager._get_session_file_path(agent_type)
        
        # Get current session (or create if doesn't exist)
        agent_session = AgentSessionManager.get_current_agent_session(agent_type)
        
        # Update session data
        agent_session['last_weights_path'] = weights_path
        agent_session['files_processed'].append({
            'path': file_path,
            'timestamp': datetime.now().isoformat()
        })
        agent_session['last_updated'] = datetime.now().isoformat()
        agent_session['active'] = True
        
        # Save updated session
        with open(session_file, 'w') as f:
            json.dump(agent_session, f, indent=2)
            
        return agent_session
    
    @staticmethod
    def reset_agent_session(agent_type):
        """Reset the session for a specific agent"""
        session_file = AgentSessionManager._get_session_file_path(agent_type)
        
        # Check if the file exists
        if os.path.exists(session_file):
            # Get current session first to keep history
            try:
                with open(session_file, 'r') as f:
                    old_session = json.load(f)
                    
                # Create archive version of old session
                archive_path = os.path.join(
                    AgentSessionManager.SESSION_FILE_DIR, 
                    f"{agent_type}_session_archive_{int(time.time())}.json"
                )
                with open(archive_path, 'w') as f:
                    json.dump(old_session, f, indent=2)
            except:
                pass  # Skip archiving if there's an error
        
        # Create new empty session
        new_session = {
            'last_weights_path': None,
            'files_processed': [],
            'session_started': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'active': True,
            'previous_session_ended': datetime.now().isoformat() if os.path.exists(session_file) else None
        }
        
        # Save new session
        with open(session_file, 'w') as f:
            json.dump(new_session, f, indent=2)
            
        return new_session
    
    @staticmethod
    def get_all_agent_sessions():
        """Get all active agent sessions"""
        AgentSessionManager._ensure_session_dir()
        sessions = {}
        
        for filename in os.listdir(AgentSessionManager.SESSION_FILE_DIR):
            if filename.endswith("_session.json"):
                agent_type = filename.replace("_session.json", "")
                try:
                    with open(os.path.join(AgentSessionManager.SESSION_FILE_DIR, filename), 'r') as f:
                        sessions[agent_type] = json.load(f)
                except:
                    continue
                    
        return sessions