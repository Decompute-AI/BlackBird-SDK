"""Agent management service for the Decompute SDK."""

from blackbird_sdk.utils.constants import CHAT_INITIALIZE, LOAD_EXISTING_MODEL, AGENT_TYPES
from blackbird_sdk.utils.errors import ValidationError, APIError

class AgentManager:
    """Manages AI agents and models."""
    
    def __init__(self, http_client):
        """Initialize the agent manager."""
        self.http_client = http_client
        
    def initialize_agent(self, agent_type, model_name=None, options=None):
        """Initialize an AI agent."""
        if agent_type not in AGENT_TYPES:
            raise ValidationError(f"Invalid agent type: {agent_type}. Must be one of {AGENT_TYPES}")
        
        data = {
            'agent': agent_type
        }
        
        if model_name:
            data['model_name'] = model_name
            
        if options:
            data.update(options)
        
        print(f"Initializing agent with data: {data}")  # Debug logging
        
        try:
            response = self.http_client.post(CHAT_INITIALIZE, data=data)
            print(f"Agent initialization response: {response}")  # Debug logging
            return response
        except Exception as e:
            print(f"Error initializing agent: {str(e)}")
            raise
    
    def load_existing_model(self, filename, model_name, agent, finance_toggle=False):
        """Load a previously saved model."""
        data = {
            'filename': filename,
            'modelname': model_name,
            'agent': agent
        }
        
        if finance_toggle:
            data['finance_toggle'] = True
            
        return self.http_client.post(LOAD_EXISTING_MODEL, data=data)
    
    def switch_agent(self, new_agent_type):
        """Switch to a different agent type."""
        return self.initialize_agent(new_agent_type)
    
    def get_available_agents(self):
        """Get list of all available agent types."""
        return AGENT_TYPES
    
    def get_agent_capabilities(self, agent_type):
        """Get capabilities of a specific agent."""
        if agent_type not in AGENT_TYPES:
            raise ValidationError(f"Invalid agent type: {agent_type}. Must be one of {AGENT_TYPES}")
            
        capabilities = {
            'general': {
                'description': 'General-purpose conversational AI',
                'supported_files': ['.pdf', '.docx', '.txt'],
                'features': ['chat', 'rag', 'fine-tuning']
            },
            'tech': {
                'description': 'Technical assistant for coding and development',
                'supported_files': ['.pdf', '.py', '.js', '.txt'],
                'features': ['chat', 'rag', 'code-analysis', 'fine-tuning']
            },
            'legal': {
                'description': 'Legal document analysis and assistance',
                'supported_files': ['.pdf', '.docx', '.txt'],
                'features': ['chat', 'rag', 'document-analysis', 'fine-tuning']
            },
            'finance': {
                'description': 'Financial analysis and reporting',
                'supported_files': ['.pdf', '.xlsx', '.xls', '.csv', '.txt'],
                'features': ['chat', 'rag', 'table-analysis', 'fine-tuning']
            },
            'meetings': {
                'description': 'Meeting transcription and analysis',
                'supported_files': ['.wav', '.mp3', '.m4a', '.txt'],
                'features': ['chat', 'rag', 'audio-processing', 'fine-tuning']
            },
            'research': {
                'description': 'Research assistant for academic content',
                'supported_files': ['.pdf', '.docx', '.txt'],
                'features': ['chat', 'rag', 'citation', 'fine-tuning']
            },
            'image-generator': {
                'description': 'AI image generation',
                'supported_files': ['.jpg', '.png', '.jpeg', '.txt'],
                'features': ['image-generation']
            }
        }
        
        return capabilities.get(agent_type, {})
