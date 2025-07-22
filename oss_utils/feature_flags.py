"""
Feature flags system for modular SDK functionality.
"""

import os
import json
from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass, asdict

class FeatureCategory(Enum):
    """Categories of SDK features."""
    CORE = "core"
    AGENTS = "agents" 
    ADVANCED = "advanced"
    TRAINING = "training"
    INTEGRATIONS = "integrations"
    EXPERIMENTAL = "experimental"

@dataclass
class FeatureDefinition:
    """Definition of a feature flag."""
    name: str
    category: FeatureCategory
    description: str
    default_enabled: bool = False
    dependencies: List[str] = None
    minimum_version: str = "0.1.0"
    deprecation_warning: Optional[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

class FeatureRegistry:
    """Registry of all available features."""
    
    # Core SDK features (always available)
    CORE_FEATURES = {
        "core_sdk": FeatureDefinition(
            name="core_sdk",
            category=FeatureCategory.CORE,
            description="Core SDK functionality with basic agent management",
            default_enabled=True
        ),
        "error_handling": FeatureDefinition(
            name="error_handling",
            category=FeatureCategory.CORE,
            description="Enhanced error handling and logging",
            default_enabled=True,
            dependencies=["core_sdk"]
        ),
        "streaming_responses": FeatureDefinition(
            name="streaming_responses", 
            category=FeatureCategory.CORE,
            description="Server-sent events response parsing",
            default_enabled=True,
            dependencies=["core_sdk"]
        ),
        # --- inside FeatureRegistry.CORE_FEATURES ---

"internal_console_logs": FeatureDefinition(
    name="internal_console_logs",
    category=FeatureCategory.CORE,
    description="Print SDK internal logs to the terminal",
    default_enabled=False,           # 🔇  OFF by default
    dependencies=["core_sdk"]
),

"verbose_internal_logs": FeatureDefinition(
    name="verbose_internal_logs",
    category=FeatureCategory.CORE,
    description="Use DEBUG level for SDK internal file logs",
    default_enabled=False,
    dependencies=["core_sdk"]
)

    }
    
    # Agent-related features
    AGENT_FEATURES = {
        "basic_agents": FeatureDefinition(
            name="basic_agents",
            category=FeatureCategory.AGENTS,
            description="Basic agent types (general, tech, legal, finance)",
            default_enabled=True,
            dependencies=["core_sdk"]
        ),
        "specialized_agents": FeatureDefinition(
            name="specialized_agents",
            category=FeatureCategory.AGENTS,
            description="Specialized agents (meetings, research, image-generator)",
            default_enabled=True,
            dependencies=["basic_agents"]
        ),
        "mas_router": FeatureDefinition(
            name="mas_router",
            category=FeatureCategory.AGENTS,
            description="Multi-Agent System router for intelligent prompt delegation",
            default_enabled=False,
            dependencies=["basic_agents"]
        )
    }
    
    # Advanced features
    ADVANCED_FEATURES = {
        "file_upload": FeatureDefinition(
            name="file_upload",
            category=FeatureCategory.ADVANCED,
            description="File upload and document processing",
            default_enabled= True,
            dependencies=["basic_agents"]
        ),
        "knowledge_graphs": FeatureDefinition(
            name="knowledge_graphs",
            category=FeatureCategory.ADVANCED,
            description="Web-based knowledge graph integration", 
            default_enabled=False,
            dependencies=["basic_agents"]
        ),
        "function_calling": FeatureDefinition(
            name="function_calling",
            category=FeatureCategory.ADVANCED,
            description="Third-party function and API calling capabilities",
            default_enabled=True,
            dependencies=["basic_agents"]
        )
    }
    
    # Training and fine-tuning features
    TRAINING_FEATURES = {
        "basic_finetuning": FeatureDefinition(
            name="basic_finetuning",
            category=FeatureCategory.TRAINING,
            description="Basic model fine-tuning with LaserTune",
            default_enabled=True,
            dependencies=["basic_agents", "file_upload"]
        ),
        "aurora_training": FeatureDefinition(
            name="aurora_training",
            category=FeatureCategory.TRAINING,
            description="Aurora LLM training algorithm integration",
            default_enabled=False,
            dependencies=["basic_finetuning"]
        ),
        "atlastune_finetuning": FeatureDefinition(
            name="atlastune_finetuning",
            category=FeatureCategory.TRAINING,
            description="AtlasTune fine-tuning algorithm integration",
            default_enabled=True,
            dependencies=["basic_finetuning"]
        )
    }
    
    # Integration features
    INTEGRATION_FEATURES = {
        "browser_integration": FeatureDefinition(
            name="browser_integration",
            category=FeatureCategory.INTEGRATIONS,
            description="Browser extension and web integration capabilities",
            default_enabled=False,
            dependencies=["core_sdk"],
            deprecation_warning="Future feature - not yet implemented"
        )
    }
    
    @classmethod
    def get_all_features(cls) -> Dict[str, FeatureDefinition]:
        """Get all registered features."""
        all_features = {}
        all_features.update(cls.CORE_FEATURES)
        all_features.update(cls.AGENT_FEATURES)
        all_features.update(cls.ADVANCED_FEATURES)
        all_features.update(cls.TRAINING_FEATURES)
        all_features.update(cls.INTEGRATION_FEATURES)
        return all_features
    
    @classmethod
    def get_features_by_category(cls, category: FeatureCategory) -> Dict[str, FeatureDefinition]:
        """Get features by category."""
        all_features = cls.get_all_features()
        return {name: feature for name, feature in all_features.items() 
                if feature.category == category}

class FeatureFlags:
    """Feature flags manager for the SDK."""
    
    def __init__(self, config: Dict[str, Any] = None, config_file: str = None):
        """Initialize feature flags with configuration."""
        self.registry = FeatureRegistry()
        self.enabled_features = set()
        self.feature_config = {}
        
        # Load configuration
        if config_file:
            self._load_from_file(config_file)
        elif config:
            self._load_from_dict(config)
        else:
            self._load_defaults()
        
        # Validate dependencies
        self._validate_dependencies()
    
    def _load_from_file(self, config_file: str):
        """Load feature configuration from file."""
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            self._load_from_dict(config)
        except FileNotFoundError:
            print(f"Feature config file {config_file} not found, using defaults")
            self._load_defaults()
        except json.JSONDecodeError as e:
            print(f"Invalid JSON in feature config file: {e}")
            self._load_defaults()
    
    def _load_from_dict(self, config: Dict[str, Any]):
        """Load feature configuration from dictionary."""
        self.feature_config = config.get('features', {})
        
        # Determine enabled features
        all_features = self.registry.get_all_features()
        for feature_name, feature_def in all_features.items():
            if feature_name in self.feature_config:
                if self.feature_config[feature_name].get('enabled', feature_def.default_enabled):
                    self.enabled_features.add(feature_name)
            elif feature_def.default_enabled:
                self.enabled_features.add(feature_name)
    
    def _load_defaults(self):
        """Load default feature configuration."""
        all_features = self.registry.get_all_features()
        for feature_name, feature_def in all_features.items():
            if feature_def.default_enabled:
                self.enabled_features.add(feature_name)
    
    def _validate_dependencies(self):
        """Validate that all dependencies for enabled features are also enabled."""
        all_features = self.registry.get_all_features()
        invalid_features = set()
        
        for feature_name in self.enabled_features:
            if feature_name in all_features:
                feature_def = all_features[feature_name]
                for dependency in feature_def.dependencies:
                    if dependency not in self.enabled_features:
                        print(f"Warning: Feature '{feature_name}' requires '{dependency}' but it's not enabled")
                        invalid_features.add(feature_name)
        
        # Remove features with missing dependencies
        self.enabled_features -= invalid_features
    
    def is_enabled(self, feature_name: str) -> bool:
        """Check if a feature is enabled."""
        return feature_name in self.enabled_features
    
    def enable_feature(self, feature_name: str) -> bool:
        """Enable a feature and its dependencies."""
        all_features = self.registry.get_all_features()
        
        if feature_name not in all_features:
            return False
        
        feature_def = all_features[feature_name]
        
        # Enable dependencies first
        for dependency in feature_def.dependencies:
            if not self.enable_feature(dependency):
                return False
        
        self.enabled_features.add(feature_name)
        return True
    
    def disable_feature(self, feature_name: str) -> bool:
        """Disable a feature and features that depend on it."""
        if feature_name not in self.enabled_features:
            return True
        
        all_features = self.registry.get_all_features()
        
        # Find features that depend on this one
        dependent_features = []
        for name, feature_def in all_features.items():
            if feature_name in feature_def.dependencies and name in self.enabled_features:
                dependent_features.append(name)
        
        # Disable dependent features first
        for dependent in dependent_features:
            self.disable_feature(dependent)
        
        self.enabled_features.discard(feature_name)
        return True
    
    def get_enabled_features(self) -> List[str]:
        """Get list of enabled features."""
        return list(self.enabled_features)
    
    def get_feature_info(self, feature_name: str) -> Optional[FeatureDefinition]:
        """Get information about a feature."""
        all_features = self.registry.get_all_features()
        return all_features.get(feature_name)
    
    def get_features_by_category(self, category: FeatureCategory) -> Dict[str, bool]:
        """Get features in a category with their enabled status."""
        category_features = self.registry.get_features_by_category(category)
        return {name: self.is_enabled(name) for name in category_features.keys()}
    
    def export_config(self) -> Dict[str, Any]:
        """Export current configuration."""
        all_features = self.registry.get_all_features()
        config = {
            "features": {
                name: {
                    "enabled": self.is_enabled(name),
                    "description": feature_def.description,
                    "category": feature_def.category.value
                }
                for name, feature_def in all_features.items()
            }
        }
        return config
    
    def save_config(self, config_file: str):
        """Save current configuration to file."""
        config = self.export_config()
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

# Global feature flags instance
_feature_flags = None

def get_feature_flags() -> FeatureFlags:
    """Get the global feature flags instance."""
    global _feature_flags
    if _feature_flags is None:
        _feature_flags = FeatureFlags()
    return _feature_flags

def configure_features(config: Dict[str, Any] = None, config_file: str = None) -> FeatureFlags:
    """Configure global feature flags."""
    global _feature_flags
    _feature_flags = FeatureFlags(config, config_file)
    return _feature_flags

def is_feature_enabled(feature_name: str) -> bool:
    """Check if a feature is enabled (convenience function)."""
    return get_feature_flags().is_enabled(feature_name)

def require_feature(feature_name: str):
    """Decorator to require a feature for a function."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not is_feature_enabled(feature_name):
                from .errors import ValidationError
                raise ValidationError(
                    f"Feature '{feature_name}' is not enabled",
                    field_name="feature_flag",
                    field_value=feature_name
                )
            return func(*args, **kwargs)
        return wrapper
    return decorator
