"""
LLM-based Web Search Integration for Vision Chat
Provides intelligent web search using Perplexity API with dynamic query generation
"""

import os
import json
import requests
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
from urllib.parse import quote_plus
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class WebSearchResult:
    """Structured web search result"""
    title: str
    content: str
    url: str
    source: str
    relevance_score: float = 0.0
    search_query: str = ""
    timestamp: str = ""
    content_hash: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if not self.content_hash:
            content = f"{self.title}|{self.content}|{self.url}"
            self.content_hash = hashlib.md5(content.encode()).hexdigest()

class QueryGenerator:
    """Generates optimized search queries from user input"""
    
    def __init__(self):
        self.query_enhancement_prompts = {
            'general': "Generate a concise, specific search query for web search based on this user question. Focus on key terms and concepts. Return only the search query, nothing else: ",
            'technical': "Generate a technical search query for finding detailed information, research papers, or technical documentation. Include specific technical terms: ",
            'news': "Generate a search query for finding recent news, updates, or current events related to this topic: ",
            'academic': "Generate an academic search query for finding research papers, studies, or scholarly articles: "
        }
    
    def generate_search_query(self, user_query: str, query_type: str = 'general', context: str = "") -> str:
        """
        Generate an optimized search query from user input
        
        Args:
            user_query: The original user query
            query_type: Type of search ('general', 'technical', 'news', 'academic')
            context: Additional context from knowledge base
            
        Returns:
            Optimized search query string
        """
        try:
            # Simple rule-based query enhancement
            enhanced_query = self._enhance_query_rules(user_query, query_type, context)
            
            # If we have access to an LLM for query generation, use it
            if hasattr(self, '_llm_enhance_query'):
                try:
                    llm_enhanced = self._llm_enhance_query(user_query, query_type, context)
                    if llm_enhanced and len(llm_enhanced.strip()) > 0:
                        enhanced_query = llm_enhanced
                except Exception as e:
                    logger.warning(f"LLM query enhancement failed, using rule-based: {e}")
            
            return enhanced_query.strip()
            
        except Exception as e:
            logger.error(f"Error generating search query: {e}")
            return user_query
    
    def _enhance_query_rules(self, user_query: str, query_type: str, context: str = "") -> str:
        """Rule-based query enhancement"""
        query = user_query.strip()
        
        # Remove common conversational phrases
        conversational_phrases = [
            "can you tell me about", "what is", "how does", "explain", 
            "i want to know about", "tell me about", "what are", "how to"
        ]
        
        for phrase in conversational_phrases:
            if query.lower().startswith(phrase.lower()):
                query = query[len(phrase):].strip()
                break
        
        # Add context-specific enhancements
        if context:
            # Extract key terms from context
            context_terms = self._extract_key_terms(context)
            if context_terms:
                query = f"{query} {' '.join(context_terms[:3])}"
        
        # Add type-specific enhancements
        if query_type == 'technical':
            query = f"{query} technical documentation research"
        elif query_type == 'news':
            query = f"{query} latest news 2024"
        elif query_type == 'academic':
            query = f"{query} research paper study"
        
        # Add quotes for exact phrases if query contains specific terms
        if len(query.split()) > 2 and any(word in query.lower() for word in ['latest', 'new', 'recent', 'current']):
            # Don't add quotes for time-sensitive queries
            pass
        elif len(query.split()) <= 3:
            # Add quotes for short, specific queries
            query = f'"{query}"'
        
        return query
    
    def _extract_key_terms(self, text: str, max_terms: int = 3) -> List[str]:
        """Extract key terms from context text"""
        # Simple keyword extraction
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
        
        words = text.lower().split()
        word_freq = {}
        
        for word in words:
            word = word.strip('.,!?;:()[]{}"\'-')
            if word and len(word) > 2 and word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Return most frequent words
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:max_terms]]

class PerplexityAPI:
    """Perplexity API integration for intelligent web search"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY", "")
        self.base_url = "https://api.perplexity.ai/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        if not self.api_key:
            logger.warning("Perplexity API key not found. Web search will be disabled.")
    
    def search(self, query: str, max_results: int = 5, search_type: str = "web") -> List[WebSearchResult]:
        """
        Perform web search using Perplexity API
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            search_type: Type of search ("web", "news", "academic")
            
        Returns:
            List of WebSearchResult objects
        """
        if not self.api_key:
            logger.error("Perplexity API key not available")
            return []
        
        try:
            logger.info(f"Starting Perplexity search for query: '{query}' (max_results: {max_results}, type: {search_type})")
            
            # Determine focus based on search type
            focus = self._get_search_focus(search_type)
            
            # Construct the prompt
            prompt = f"""Please search the web for information about: {query}

Focus on: {focus}

Please provide {max_results} relevant results. For each result, use this exact format:

- Title: [clear, descriptive title]
- Content: [brief summary or key information, 2-3 sentences]
- URL: [source URL if available]

Example format:
- Title: Introduction to Machine Learning
- Content: Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It uses algorithms to identify patterns in data and make predictions or decisions.
- URL: https://example.com/machine-learning

Make sure the information is accurate, recent, and directly addresses the query. If you can't find a URL for a result, leave the URL field empty."""

            payload = {
                "model": "sonar",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 2000,
                "temperature": 0.3,
                "top_p": 0.9,
                "stream": False
            }
            
            logger.info(f"Sending request to Perplexity API...")
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            logger.info(f"Perplexity API response status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                
                logger.info(f"Received response content (length: {len(content)})")
                
                # Parse the response to extract structured results
                results = self._parse_perplexity_response(content, query)
                logger.info(f"Parsed {len(results)} results from response")
                return results[:max_results]
            else:
                logger.error(f"Perplexity API error: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"Error in Perplexity search: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []
    
    def _get_search_focus(self, search_type: str) -> str:
        """Get search focus based on type"""
        focus_map = {
            "web": "general web information, recent updates, and comprehensive coverage",
            "news": "latest news, current events, and recent developments",
            "academic": "research papers, scholarly articles, and academic sources",
            "technical": "technical documentation, specifications, and detailed technical information"
        }
        return focus_map.get(search_type, focus_map["web"])
    
    def _parse_perplexity_response(self, content: str, original_query: str) -> List[WebSearchResult]:
        """Parse Perplexity API response into structured results"""
        results = []
        
        try:
            logger.info(f"Parsing Perplexity response (length: {len(content)})")
            logger.debug(f"Raw response: {content[:500]}...")
            
            # First, try to parse structured format with dashes
            sections = content.split('\n\n')
            
            for section in sections:
                if not section.strip():
                    continue
                
                # Try to extract structured information
                title = ""
                content_text = ""
                url = ""
                
                lines = section.split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith('- Title:'):
                        title = line.replace('- Title:', '').strip()
                    elif line.startswith('- Content:'):
                        content_text = line.replace('- Content:', '').strip()
                    elif line.startswith('- URL:'):
                        url = line.replace('- URL:', '').strip()
                
                # If we found structured data, create result
                if title and content_text:
                    # Calculate relevance score
                    relevance_score = self._calculate_relevance_score(original_query, title + " " + content_text)
                    
                    result = WebSearchResult(
                        title=title,
                        content=content_text,
                        url=url,
                        source="perplexity",
                        relevance_score=relevance_score,
                        search_query=original_query
                    )
                    results.append(result)
                    logger.info(f"Parsed structured result: {title[:50]}...")
            
            # If no structured results found, try alternative parsing
            if not results:
                logger.info("No structured results found, trying alternative parsing...")
                
                # Try to split by common separators
                separators = ['\n\n', '\n---\n', '\n***\n', '\n###\n']
                for separator in separators:
                    if separator in content:
                        sections = content.split(separator)
                        break
                else:
                    # If no clear separators, treat as single section
                    sections = [content]
                
                for i, section in enumerate(sections):
                    if not section.strip():
                        continue
                    
                    # Try to extract title from first line
                    lines = section.strip().split('\n')
                    title = lines[0].strip() if lines else f"Result {i+1}"
                    
                    # Get content from remaining lines
                    content_lines = lines[1:] if len(lines) > 1 else lines
                    content_text = '\n'.join(content_lines).strip()
                    
                    # Look for URLs in the content
                    url = ""
                    url_pattern = r'https?://[^\s]+'
                    urls = re.findall(url_pattern, content_text)
                    if urls:
                        url = urls[0]
                        # Remove URL from content for cleaner display
                        content_text = re.sub(url_pattern, '', content_text).strip()
                    
                    if content_text:
                        # Calculate relevance score
                        relevance_score = self._calculate_relevance_score(original_query, title + " " + content_text)
                        
                        result = WebSearchResult(
                            title=title,
                            content=content_text,
                            url=url,
                            source="perplexity",
                            relevance_score=relevance_score,
                            search_query=original_query
                        )
                        results.append(result)
                        logger.info(f"Parsed alternative result: {title[:50]}...")
            
            # If still no results, create a single result from the content
            if not results and content.strip():
                logger.info("Creating fallback result from raw content")
                result = WebSearchResult(
                    title=f"Search results for: {original_query}",
                    content=content[:500] + "..." if len(content) > 500 else content,
                    url="",
                    source="perplexity",
                    relevance_score=0.8,
                    search_query=original_query
                )
                results.append(result)
            
            logger.info(f"Parsed {len(results)} results from Perplexity response")
            return results
                
        except Exception as e:
            logger.error(f"Error parsing Perplexity response: {e}")
            # Return a fallback result
            return [WebSearchResult(
                title=f"Search results for: {original_query}",
                content=f"Error parsing response: {str(e)}. Raw content: {content[:200]}...",
                url="",
                source="perplexity",
                relevance_score=0.5,
                search_query=original_query
            )]
    
    def _calculate_relevance_score(self, query: str, content: str) -> float:
        """Calculate relevance score between query and content"""
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

class LLMWebSearchManager:
    """Main manager for LLM-based web search integration"""
    
    def __init__(self, api_key: str = None):
        self.query_generator = QueryGenerator()
        self.perplexity_api = PerplexityAPI(api_key)
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour cache
        
    def search_web(self, user_query: str, context: str = "", max_results: int = 5) -> List[WebSearchResult]:
        """
        Perform intelligent web search based on user query
        
        Args:
            user_query: The original user query
            context: Additional context from knowledge base
            max_results: Maximum number of results to return
            
        Returns:
            List of WebSearchResult objects
        """
        try:
            # Generate optimized search query
            search_query = self.query_generator.generate_search_query(
                user_query, 
                query_type='general', 
                context=context
            )
            
            # Check cache first
            cache_key = hashlib.md5(f"{search_query}:{context}".encode()).hexdigest()
            if cache_key in self.cache:
                cached_result = self.cache[cache_key]
                if time.time() - cached_result['timestamp'] < self.cache_ttl:
                    logger.info("Returning cached web search results")
                    return cached_result['results']
            
            # Perform web search
            logger.info(f"Performing web search for query: {search_query}")
            results = self.perplexity_api.search(search_query, max_results)
            
            # Cache results
            self.cache[cache_key] = {
                'results': results,
                'timestamp': time.time()
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in web search: {e}")
            return []
    
    def search_with_context(self, user_query: str, kb_context: str = "", max_results: int = 5) -> Dict[str, Any]:
        """
        Perform web search with knowledge base context integration
        
        Args:
            user_query: The original user query
            kb_context: Knowledge base context
            max_results: Maximum number of results to return
            
        Returns:
            Dictionary containing web results and enhanced context
        """
        try:
            # Perform web search
            web_results = self.search_web(user_query, kb_context, max_results)
            
            # Combine web results with KB context
            combined_context = self._combine_contexts(kb_context, web_results)
            
            return {
                'web_results': web_results,
                'combined_context': combined_context,
                'search_performed': len(web_results) > 0,
                'total_sources': len(web_results)
            }
            
        except Exception as e:
            logger.error(f"Error in contextual web search: {e}")
            return {
                'web_results': [],
                'combined_context': kb_context,
                'search_performed': False,
                'total_sources': 0
            }
    
    def _combine_contexts(self, kb_context: str, web_results: List[WebSearchResult]) -> str:
        """Combine knowledge base context with web search results"""
        combined = ""
        
        # Add knowledge base context if available
        if kb_context.strip():
            combined += "KNOWLEDGE BASE CONTEXT:\n"
            combined += "=" * 50 + "\n"
            combined += kb_context + "\n\n"
        
        # Add web search results
        if web_results:
            combined += "WEB SEARCH RESULTS:\n"
            combined += "=" * 50 + "\n"
            
            for i, result in enumerate(web_results, 1):
                combined += f"[Web Source {i}]\n"
                combined += f"Title: {result.title}\n"
                combined += f"Content: {result.content}\n"
                if result.url:
                    combined += f"URL: {result.url}\n"
                combined += f"Relevance Score: {result.relevance_score:.2f}\n"
                combined += "-" * 30 + "\n\n"
        
        return combined.strip()
    
    def clear_cache(self):
        """Clear the search cache"""
        self.cache.clear()
        logger.info("Web search cache cleared")

# Global instance for easy access
_web_search_manager = None

def get_web_search_manager(api_key: str = None) -> LLMWebSearchManager:
    """Get or create global web search manager instance"""
    global _web_search_manager
    if _web_search_manager is None:
        _web_search_manager = LLMWebSearchManager(api_key)
    return _web_search_manager

def search_web_for_query(user_query: str, context: str = "", max_results: int = 5) -> List[WebSearchResult]:
    """Convenience function for web search"""
    manager = get_web_search_manager()
    return manager.search_web(user_query, context, max_results)

def search_with_kb_context(user_query: str, kb_context: str = "", max_results: int = 5) -> Dict[str, Any]:
    """Convenience function for contextual web search"""
    manager = get_web_search_manager()
    return manager.search_with_context(user_query, kb_context, max_results) 