"""
Enhanced Multi-API Medical Research Content Discovery System
Optimized for medical research with Brave, PubMed, Semantic Scholar, and NewsAPI
"""

import requests
import time
import json
import hashlib
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import concurrent.futures
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET
import feedparser
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Unified search result structure"""
    title: str
    abstract: str
    url: str
    authors: List[str]
    publication_date: str
    source: str
    source_type: str  # 'academic', 'news', 'preprint', 'social'
    doi: Optional[str] = None
    journal: Optional[str] = None
    keywords: List[str] = None
    content_hash: str = None
    relevance_score: float = 0.0
    medical_relevance: float = 0.0

    def __post_init__(self):
        if self.content_hash is None:
            content = f"{self.title}|{self.abstract}|{self.url}"
            self.content_hash = hashlib.md5(content.encode()).hexdigest()
        if self.keywords is None:
            self.keywords = []

class MedicalSearchConfig:
    """Enhanced configuration for medical research searches"""
    
    def __init__(self):
        # API Keys (set these in environment or config file)
        self.newsapi_key = ""  # Get from newsapi.org
        self.brave_api_key = ""  # Get from brave.com/search/api
        self.semantic_scholar_key = ""  # Optional but recommended
        
        # Medical-specific search terms
        self.medical_keywords = [
            "clinical trial", "randomized controlled trial", "systematic review",
            "meta-analysis", "peer reviewed", "biomarker", "treatment efficacy",
            "patient outcomes", "therapeutic intervention", "medical device",
            "drug discovery", "pharmaceutical", "healthcare innovation",
            "medical breakthrough", "FDA approval", "clinical study",
            "diagnosis", "prognosis", "pathology", "epidemiology",
            "immunotherapy", "precision medicine", "personalized medicine"
        ]
        
        # Trusted medical domains
        self.medical_domains = [
            "pubmed.ncbi.nlm.nih.gov", "nature.com", "nejm.org", "thelancet.com",
            "bmj.com", "jamanetwork.com", "cell.com", "science.org",
            "frontiersin.org", "plos.org", "biorxiv.org", "medrxiv.org",
            "cochranelibrary.com", "who.int", "cdc.gov", "fda.gov"
        ]
        
        # Search parameters
        self.max_results_per_source = 50
        self.days_back = 30
        self.min_relevance_threshold = 0.6

class BraveSearchAPI:
    """Brave Search API implementation"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.search.brave.com/res/v1/web/search"
        self.headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": api_key
        }
    
    def search(self, query: str, count: int = 20, medical_focus: bool = True) -> List[SearchResult]:
        """Enhanced search with medical domain biasing"""
        results = []
        
        if medical_focus:
            # Add medical domain bias to query
            domain_filter = " OR ".join([f"site:{domain}" for domain in MedicalSearchConfig().medical_domains[:5]])
            enhanced_query = f"({query}) AND ({domain_filter})"
        else:
            enhanced_query = query
        
        params = {
            "q": enhanced_query,
            "count": count,
            "result_filter": "web",
            "safesearch": "moderate",
            "freshness": "pm",  # Past month
            "text_decorations": False
        }
        
        try:
            response = requests.get(self.base_url, headers=self.headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            for item in data.get("web", {}).get("results", []):
                result = SearchResult(
                    title=item.get("title", ""),
                    abstract=item.get("description", ""),
                    url=item.get("url", ""),
                    authors=[item.get("profile", {}).get("name", "")] if item.get("profile") else [],
                    publication_date=item.get("age", ""),
                    source="brave",
                    source_type="web",
                    relevance_score=self._calculate_medical_relevance(item.get("title", "") + " " + item.get("description", ""))
                )
                results.append(result)
                
        except Exception as e:
            logger.error(f"Brave Search API error: {e}")
        
        return results
    
    def _calculate_medical_relevance(self, text: str) -> float:
        """Calculate medical relevance score"""
        medical_terms = MedicalSearchConfig().medical_keywords
        text_lower = text.lower()
        matches = sum(1 for term in medical_terms if term in text_lower)
        return min(matches / len(medical_terms), 1.0)

class PubMedAPI:
    """PubMed/NCBI E-utilities API implementation"""
    
    def __init__(self, email: str):
        self.email = email
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.tool = "medical_research_pipeline"
    
    def search(self, query: str, max_results: int = 50) -> List[SearchResult]:
        """Search PubMed for medical research papers"""
        results = []
        
        try:
            # First, search for PMIDs
            search_url = f"{self.base_url}/esearch.fcgi"
            search_params = {
                "db": "pubmed",
                "term": query,
                "retmax": max_results,
                "retmode": "json",
                "email": self.email,
                "tool": self.tool,
                "sort": "relevance"
            }
            
            search_response = requests.get(search_url, params=search_params, timeout=10)
            search_data = search_response.json()
            
            pmids = search_data.get("esearchresult", {}).get("idlist", [])
            
            if pmids:
                # Fetch detailed information
                fetch_url = f"{self.base_url}/efetch.fcgi"
                fetch_params = {
                    "db": "pubmed",
                    "id": ",".join(pmids),
                    "retmode": "xml",
                    "email": self.email,
                    "tool": self.tool
                }
                
                fetch_response = requests.get(fetch_url, params=fetch_params, timeout=15)
                
                # Parse XML response
                root = ET.fromstring(fetch_response.text)
                
                for article in root.findall(".//PubmedArticle"):
                    result = self._parse_pubmed_article(article)
                    if result:
                        results.append(result)
                        
        except Exception as e:
            logger.error(f"PubMed API error: {e}")
        
        return results
    
    def _parse_pubmed_article(self, article_xml) -> Optional[SearchResult]:
        """Parse PubMed XML article"""
        try:
            medline = article_xml.find(".//MedlineCitation")
            article_elem = medline.find(".//Article")
            
            # Extract title
            title_elem = article_elem.find(".//ArticleTitle")
            title = title_elem.text if title_elem is not None else ""
            
            # Extract abstract
            abstract_elem = article_elem.find(".//Abstract/AbstractText")
            abstract = abstract_elem.text if abstract_elem is not None else ""
            
            # Extract authors
            authors = []
            for author in article_elem.findall(".//AuthorList/Author"):
                last_name = author.find("LastName")
                first_name = author.find("ForeName")
                if last_name is not None and first_name is not None:
                    authors.append(f"{first_name.text} {last_name.text}")
            
            # Extract publication date
            pub_date = article_elem.find(".//PubDate")
            year = pub_date.find("Year").text if pub_date.find("Year") is not None else ""
            
            # Extract journal
            journal_elem = article_elem.find(".//Journal/Title")
            journal = journal_elem.text if journal_elem is not None else ""
            
            # Extract PMID for URL
            pmid_elem = medline.find(".//PMID")
            pmid = pmid_elem.text if pmid_elem is not None else ""
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""
            
            return SearchResult(
                title=title,
                abstract=abstract,
                url=url,
                authors=authors,
                publication_date=year,
                source="pubmed",
                source_type="academic",
                journal=journal,
                relevance_score=1.0,  # PubMed results are inherently medically relevant
                medical_relevance=1.0
            )
            
        except Exception as e:
            logger.error(f"Error parsing PubMed article: {e}")
            return None

class SemanticScholarAPI:
    """Semantic Scholar API implementation"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        self.headers = {"x-api-key": api_key} if api_key else {}
    
    def search(self, query: str, limit: int = 50) -> List[SearchResult]:
        """Search Semantic Scholar for academic papers"""
        results = []
        
        try:
            search_url = f"{self.base_url}/paper/search"
            params = {
                "query": query,
                "limit": limit,
                "fields": "title,abstract,authors,year,journal,url,externalIds"
            }
            
            response = requests.get(search_url, headers=self.headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            for paper in data.get("data", []):
                authors = [author.get("name", "") for author in paper.get("authors", [])]
                
                result = SearchResult(
                    title=paper.get("title", ""),
                    abstract=paper.get("abstract", ""),
                    url=paper.get("url", ""),
                    authors=authors,
                    publication_date=str(paper.get("year", "")),
                    source="semantic_scholar",
                    source_type="academic",
                    journal=paper.get("journal", {}).get("name", "") if paper.get("journal") else "",
                    doi=paper.get("externalIds", {}).get("DOI", ""),
                    relevance_score=self._calculate_medical_relevance(paper.get("title", "") + " " + paper.get("abstract", ""))
                )
                results.append(result)
                
        except Exception as e:
            logger.error(f"Semantic Scholar API error: {e}")
        
        return results
    
    def _calculate_medical_relevance(self, text: str) -> float:
        """Calculate medical relevance score"""
        medical_terms = MedicalSearchConfig().medical_keywords
        text_lower = text.lower()
        matches = sum(1 for term in medical_terms if term in text_lower)
        return min(matches / len(medical_terms), 1.0)

class BioRxivAPI:
    """bioRxiv and medRxiv API implementation"""
    
    def __init__(self):
        self.base_url = "https://api.biorxiv.org"
    
    def search_recent(self, days_back: int = 7, server: str = "medrxiv") -> List[SearchResult]:
        """Search recent preprints from bioRxiv/medRxiv"""
        results = []
        
        try:
            # Get recent papers
            url = f"{self.base_url}/details/{server}/{days_back}d"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            for paper in data.get("collection", []):
                result = SearchResult(
                    title=paper.get("title", ""),
                    abstract=paper.get("abstract", ""),
                    url=f"https://www.{server}.org/content/{paper.get('doi', '')}",
                    authors=paper.get("authors", "").split(";") if paper.get("authors") else [],
                    publication_date=paper.get("date", ""),
                    source=server,
                    source_type="preprint",
                    doi=paper.get("doi", ""),
                    relevance_score=self._calculate_medical_relevance(paper.get("title", "") + " " + paper.get("abstract", ""))
                )
                results.append(result)
                
        except Exception as e:
            logger.error(f"bioRxiv API error: {e}")
        
        return results
    
    def _calculate_medical_relevance(self, text: str) -> float:
        """Calculate medical relevance score"""
        medical_terms = MedicalSearchConfig().medical_keywords
        text_lower = text.lower()
        matches = sum(1 for term in medical_terms if term in text_lower)
        return min(matches / len(medical_terms), 1.0)

class MultiAPISearchManager:
    """Main class to coordinate all search APIs"""
    
    def __init__(self, config: MedicalSearchConfig):
        self.config = config
        self.apis = {}
        
        # Initialize APIs
        if config.brave_api_key:
            self.apis['brave'] = BraveSearchAPI(config.brave_api_key)
        
        if config.newsapi_key:
            self.apis['news'] = NewsAPI(config.newsapi_key)
        
        # PubMed requires email but no API key
        self.apis['pubmed'] = PubMedAPI("your-email@domain.com")  # Replace with actual email
        
        # Semantic Scholar can work without API key but better with it
        self.apis['semantic'] = SemanticScholarAPI(config.semantic_scholar_key)
        
        # bioRxiv/medRxiv are free
        self.apis['biorxiv'] = BioRxivAPI()
    
    def comprehensive_search(self, query: str, max_results_per_source: int = 20) -> List[SearchResult]:
        """Perform comprehensive search across all available APIs"""
        all_results = []
        
        # Execute searches in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_api = {}
            
            # Submit search tasks
            if 'brave' in self.apis:
                future_to_api[executor.submit(self.apis['brave'].search, query, max_results_per_source)] = 'brave'
            
            if 'pubmed' in self.apis:
                future_to_api[executor.submit(self.apis['pubmed'].search, query, max_results_per_source)] = 'pubmed'
            
            if 'semantic' in self.apis:
                future_to_api[executor.submit(self.apis['semantic'].search, query, max_results_per_source)] = 'semantic'
            
            if 'biorxiv' in self.apis:
                future_to_api[executor.submit(self.apis['biorxiv'].search_recent, 30, "medrxiv")] = 'medrxiv'
                future_to_api[executor.submit(self.apis['biorxiv'].search_recent, 30, "biorxiv")] = 'biorxiv'
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_api):
                api_name = future_to_api[future]
                try:
                    results = future.result()
                    logger.info(f"Retrieved {len(results)} results from {api_name}")
                    all_results.extend(results)
                except Exception as e:
                    logger.error(f"Error retrieving results from {api_name}: {e}")
        
        # Deduplicate and filter
        deduplicated_results = self._deduplicate_results(all_results)
        filtered_results = self._filter_by_relevance(deduplicated_results)
        
        logger.info(f"Total results after deduplication and filtering: {len(filtered_results)}")
        return filtered_results
    
    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate results based on content hash"""
        seen_hashes = set()
        unique_results = []
        
        for result in results:
            if result.content_hash not in seen_hashes:
                seen_hashes.add(result.content_hash)
                unique_results.append(result)
        
        return unique_results
    
    def _filter_by_relevance(self, results: List[SearchResult]) -> List[SearchResult]:
        """Filter results by medical relevance threshold"""
        filtered = [r for r in results if r.relevance_score >= self.config.min_relevance_threshold]
        return sorted(filtered, key=lambda x: x.relevance_score, reverse=True)

class NewsAPI:
    """NewsAPI implementation for health news"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2"
    
    def search(self, query: str, max_results: int = 50) -> List[SearchResult]:
        """Search for health-related news"""
        results = []
        
        try:
            url = f"{self.base_url}/everything"
            params = {
                "q": f"{query} AND (health OR medical OR healthcare)",
                "language": "en",
                "sortBy": "relevancy",
                "pageSize": min(max_results, 100),
                "apiKey": self.api_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            for article in data.get("articles", []):
                result = SearchResult(
                    title=article.get("title", ""),
                    abstract=article.get("description", ""),
                    url=article.get("url", ""),
                    authors=[article.get("author", "")] if article.get("author") else [],
                    publication_date=article.get("publishedAt", ""),
                    source="newsapi",
                    source_type="news",
                    relevance_score=self._calculate_medical_relevance(article.get("title", "") + " " + article.get("description", ""))
                )
                results.append(result)
                
        except Exception as e:
            logger.error(f"NewsAPI error: {e}")
        
        return results
    
    def _calculate_medical_relevance(self, text: str) -> float:
        """Calculate medical relevance score"""
        medical_terms = MedicalSearchConfig().medical_keywords
        text_lower = text.lower()
        matches = sum(1 for term in medical_terms if term in text_lower)
        return min(matches / len(medical_terms), 1.0)

# Example usage
if __name__ == "__main__":
    config = MedicalSearchConfig()
    config.brave_api_key = "YOUR_BRAVE_API_KEY"  # Set your API keys
    config.newsapi_key = "YOUR_NEWS_API_KEY"
    
    search_manager = MultiAPISearchManager(config)
    
    # Example search
    results = search_manager.comprehensive_search("immunotherapy cancer treatment 2024")
    
    print(f"Found {len(results)} relevant medical research results")
    for result in results[:5]:  # Show top 5
        print(f"Title: {result.title}")
        print(f"Source: {result.source} ({result.source_type})")
        print(f"Relevance: {result.relevance_score:.2f}")
        print(f"URL: {result.url}")
        print("-" * 80)
