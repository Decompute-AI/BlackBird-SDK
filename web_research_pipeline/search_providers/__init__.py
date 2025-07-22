"""
Search Providers Package

This package contains modular search provider implementations for the web research pipeline.
Each provider implements a common interface and can be easily swapped or combined.
"""

from .base_provider import SearchProvider, SearchProviderRegistry, SearchResult
from .perplexity_provider import PerplexityProvider
from .newsapi_provider import NewsAPIProvider

__all__ = [
    'SearchProvider',
    'SearchProviderRegistry',
    'SearchResult',
    'PerplexityProvider',
    'NewsAPIProvider'
] 