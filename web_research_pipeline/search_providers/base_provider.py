"""
Base Search Provider Interface

Defines the common interface and registry for all search providers in the web research pipeline.
"""

import os
import logging
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
from enum import Enum

logger = logging.getLogger(__name__)

class SearchProviderType(Enum):
    """Enumeration of available search provider types"""
    PERPLEXITY = "perplexity"
    NEWSAPI = "newsapi"
    FIREFLOW = "fireflow"

@dataclass
class SearchResult:
    """Standardized search result structure for all providers"""
    title: str
    content: str
    url: str
    source: str
    relevance_score: float = 0.0
    search_query: str = ""
    timestamp: str = ""
    content_hash: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if not self.content_hash:
            content = f"{self.title}|{self.content}|{self.url}"
            self.content_hash = hashlib.md5(content.encode()).hexdigest()
        if self.metadata is None:
            self.metadata = {}

class SearchProvider(ABC):
    """Abstract base class for all search providers"""
    
    def __init__(self, api_key: str = None, **kwargs):
        """
        Initialize the search provider
        
        Args:
            api_key: API key for the provider
            **kwargs: Additional provider-specific configuration
        """
        self.api_key = api_key
        self.provider_name = self.__class__.__name__
        self.is_configured = bool(self.api_key)
        self.last_request_time = 0
        self.request_count = 0
        self.error_count = 0
        
        if not self.is_configured:
            logger.warning(f"{self.provider_name} API key not found. Provider will be disabled.")
    
    @abstractmethod
    def search(self, query: str, max_results: int = 5, **kwargs) -> List[SearchResult]:
        """
        Perform search using this provider
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            **kwargs: Additional search parameters
            
        Returns:
            List of SearchResult objects
        """
        pass
    
    @abstractmethod
    def get_provider_info(self) -> Dict[str, Any]:
        """
        Get information about this provider
        
        Returns:
            Dictionary containing provider information
        """
        pass
    
    def is_available(self) -> bool:
        """Check if the provider is available and configured"""
        return self.is_configured
    
    def get_rate_limit_info(self) -> Dict[str, Any]:
        """Get rate limiting information for this provider"""
        return {
            'provider': self.provider_name,
            'is_configured': self.is_configured,
            'last_request_time': self.last_request_time,
            'request_count': self.request_count,
            'error_count': self.error_count
        }
    
    def _rate_limit_check(self, min_interval: float = 0.1) -> bool:
        """
        Check if enough time has passed since the last request
        
        Args:
            min_interval: Minimum time between requests in seconds
            
        Returns:
            True if rate limit allows, False otherwise
        """
        current_time = time.time()
        if current_time - self.last_request_time < min_interval:
            logger.warning(f"Rate limit hit for {self.provider_name}")
            return False
        return True
    
    def _update_request_stats(self, success: bool = True):
        """Update request statistics"""
        self.last_request_time = time.time()
        self.request_count += 1
        if not success:
            self.error_count += 1
    
    def _calculate_relevance_score(self, query: str, content: str) -> float:
        """
        Calculate relevance score between query and content
        
        Args:
            query: Search query
            content: Content to score
            
        Returns:
            Relevance score between 0.0 and 1.0
        """
        try:
            query_words = set(query.lower().split())
            content_words = set(content.lower().split())
            
            if not query_words:
                return 0.0
            
            # Calculate word overlap
            overlap = len(query_words.intersection(content_words))
            relevance = overlap / len(query_words)
            
            # Boost score for longer, more detailed content
            if len(content) > 100:
                relevance += 0.1
            
            return min(relevance, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating relevance score: {e}")
            return 0.5

class SearchProviderRegistry:
    """Registry for managing search providers"""
    
    def __init__(self):
        self._providers: Dict[str, SearchProvider] = {}
        self._default_provider: Optional[str] = None
    
    def register_provider(self, provider: SearchProvider, name: str = None) -> None:
        """
        Register a search provider
        
        Args:
            provider: SearchProvider instance
            name: Optional custom name for the provider
        """
        provider_name = name or provider.provider_name
        self._providers[provider_name] = provider
        logger.info(f"Registered search provider: {provider_name}")
        
        # Set as default if it's the first provider
        if self._default_provider is None:
            self._default_provider = provider_name
    
    def get_provider(self, name: str) -> Optional[SearchProvider]:
        """
        Get a provider by name
        
        Args:
            name: Provider name
            
        Returns:
            SearchProvider instance or None if not found
        """
        return self._providers.get(name)
    
    def get_available_providers(self) -> List[str]:
        """
        Get list of available provider names
        
        Returns:
            List of provider names
        """
        return list(self._providers.keys())
    
    def get_configured_providers(self) -> List[str]:
        """
        Get list of configured provider names
        
        Returns:
            List of configured provider names
        """
        return [name for name, provider in self._providers.items() if provider.is_available()]
    
    def set_default_provider(self, name: str) -> bool:
        """
        Set the default provider
        
        Args:
            name: Provider name
            
        Returns:
            True if successful, False if provider not found
        """
        if name in self._providers:
            self._default_provider = name
            logger.info(f"Set default search provider: {name}")
            return True
        else:
            logger.error(f"Provider not found: {name}")
            return False
    
    def get_default_provider(self) -> Optional[SearchProvider]:
        """
        Get the default provider
        
        Returns:
            Default SearchProvider instance or None
        """
        if self._default_provider and self._default_provider in self._providers:
            return self._providers[self._default_provider]
        return None
    
    def search(self, query: str, provider_name: str = None, max_results: int = 5, **kwargs) -> List[SearchResult]:
        """
        Perform search using specified or default provider
        
        Args:
            query: Search query
            provider_name: Provider name (uses default if None)
            max_results: Maximum number of results
            **kwargs: Additional search parameters
            
        Returns:
            List of SearchResult objects
        """
        provider = None
        
        if provider_name:
            provider = self.get_provider(provider_name)
            if not provider:
                logger.error(f"Provider not found: {provider_name}")
                return []
        else:
            provider = self.get_default_provider()
            if not provider:
                logger.error("No default provider configured")
                return []
        
        if not provider.is_available():
            logger.error(f"Provider not available: {provider.provider_name}")
            return []
        
        try:
            return provider.search(query, max_results, **kwargs)
        except Exception as e:
            logger.error(f"Error searching with {provider.provider_name}: {e}")
            return []
    
    def search_multiple(self, query: str, provider_names: List[str], max_results: int = 5, **kwargs) -> Dict[str, List[SearchResult]]:
        """
        Perform search using multiple providers
        
        Args:
            query: Search query
            provider_names: List of provider names to use
            max_results: Maximum number of results per provider
            **kwargs: Additional search parameters
            
        Returns:
            Dictionary mapping provider names to their results
        """
        results = {}
        
        for provider_name in provider_names:
            provider = self.get_provider(provider_name)
            if provider and provider.is_available():
                try:
                    provider_results = provider.search(query, max_results, **kwargs)
                    results[provider_name] = provider_results
                except Exception as e:
                    logger.error(f"Error searching with {provider_name}: {e}")
                    results[provider_name] = []
            else:
                logger.warning(f"Provider not available: {provider_name}")
                results[provider_name] = []
        
        return results
    
    def get_provider_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status of all providers
        
        Returns:
            Dictionary mapping provider names to their status information
        """
        status = {}
        
        for name, provider in self._providers.items():
            status[name] = {
                'is_configured': provider.is_configured,
                'is_available': provider.is_available(),
                'rate_limit_info': provider.get_rate_limit_info(),
                'provider_info': provider.get_provider_info()
            }
        
        return status

# Global registry instance
_search_provider_registry = SearchProviderRegistry()

def get_search_provider_registry() -> SearchProviderRegistry:
    """Get the global search provider registry"""
    return _search_provider_registry

def register_search_provider(provider: SearchProvider, name: str = None) -> None:
    """Register a search provider in the global registry"""
    _search_provider_registry.register_provider(provider, name)

def get_search_provider(name: str) -> Optional[SearchProvider]:
    """Get a search provider from the global registry"""
    return _search_provider_registry.get_provider(name)

def search_with_provider(query: str, provider_name: str = None, max_results: int = 5, **kwargs) -> List[SearchResult]:
    """Perform search using the global registry"""
    return _search_provider_registry.search(query, provider_name, max_results, **kwargs) 