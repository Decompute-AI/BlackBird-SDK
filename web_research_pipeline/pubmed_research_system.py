"""
PubMed Research System for Vision Chat
Independent system that analyzes knowledge base content, generates research queries,
searches PubMed for peer-reviewed papers, and stores results in a separate research KB
"""

import os
import json
import requests
import time
import logging
import hashlib
import threading
import schedule
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus, urlparse
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PubMedPaper:
    """Structured PubMed research paper"""
    pmid: str
    title: str
    abstract: str
    authors: List[str]
    journal: str
    publication_date: str
    doi: str
    keywords: List[str]
    relevance_score: float = 0.0
    search_query: str = ""
    timestamp: str = ""
    content_hash: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if not self.content_hash:
            content = f"{self.pmid}|{self.title}|{self.abstract}"
            self.content_hash = hashlib.md5(content.encode()).hexdigest()

class PubMedAPI:
    """PubMed API integration for research paper search"""
    
    def __init__(self, email: str = None):
        self.email = email or os.getenv("PUBMED_EMAIL", "researcher@university.edu")
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.search_url = f"{self.base_url}/esearch.fcgi"
        self.fetch_url = f"{self.base_url}/efetch.fcgi"
        self.summary_url = f"{self.base_url}/esummary.fcgi"
        
        # Top peer-reviewed journals (can be expanded)
        self.top_journals = {
            'nature', 'science', 'cell', 'lancet', 'nejm', 'jama', 'bmj', 
            'annals of internal medicine', 'plos medicine', 'plos biology',
            'proceedings of the national academy of sciences', 'pnas',
            'journal of the american medical association', 'the new england journal of medicine',
            'british medical journal', 'the lancet', 'cell press', 'nature publishing group'
        }
    
    def search_papers(self, query: str, max_results: int = 5, days_back: int = 365) -> List[PubMedPaper]:
        """
        Search PubMed for peer-reviewed papers
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            days_back: Search papers published within this many days
            
        Returns:
            List of PubMedPaper objects
        """
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # Format dates for PubMed
            start_date_str = start_date.strftime("%Y/%m/%d")
            end_date_str = end_date.strftime("%Y/%m/%d")
            
            # Build search parameters
            params = {
                'db': 'pubmed',
                'term': query,
                'retmax': max_results * 3,  # Get more results for filtering
                'retmode': 'json',
                'email': self.email,
                'tool': 'decompute_research_system',
                'datetype': 'pdat',
                'mindate': start_date_str,
                'maxdate': end_date_str,
                'sort': 'relevance'  # Sort by relevance
            }
            
            # Search for papers
            response = requests.get(self.search_url, params=params, timeout=30)
            
            if response.status_code != 200:
                logger.error(f"PubMed search failed: {response.status_code}")
                return []
            
            data = response.json()
            id_list = data.get('esearchresult', {}).get('idlist', [])
            
            if not id_list:
                logger.info(f"No papers found for query: {query}")
                return []
            
            # Fetch detailed information for each paper
            papers = []
            for pmid in id_list[:max_results * 2]:  # Limit for detailed fetching
                try:
                    paper = self._fetch_paper_details(pmid)
                    if paper and self._is_peer_reviewed(paper):
                        papers.append(paper)
                        if len(papers) >= max_results:
                            break
                except Exception as e:
                    logger.warning(f"Error fetching paper {pmid}: {e}")
                    continue
            
            # Sort by relevance and return top results
            papers.sort(key=lambda x: x.relevance_score, reverse=True)
            return papers[:max_results]
            
        except Exception as e:
            logger.error(f"Error in PubMed search: {e}")
            return []
    
    def _fetch_paper_details(self, pmid: str) -> Optional[PubMedPaper]:
        """Fetch detailed information for a specific paper"""
        try:
            # Get summary information
            summary_params = {
                'db': 'pubmed',
                'id': pmid,
                'retmode': 'json',
                'email': self.email,
                'tool': 'decompute_research_system'
            }
            
            response = requests.get(self.summary_url, params=summary_params, timeout=30)
            
            if response.status_code != 200:
                return None
            
            data = response.json()
            result = data.get('result', {}).get(pmid, {})
            
            if not result:
                return None
            
            # Extract paper information
            title = result.get('title', '')
            abstract = result.get('abstract', '')
            journal = result.get('fulljournalname', '')
            pub_date = result.get('pubdate', '')
            doi = result.get('elocationid', '')
            
            # Extract authors
            authors = []
            author_list = result.get('authors', [])
            for author in author_list:
                if isinstance(author, dict):
                    name = author.get('name', '')
                    if name:
                        authors.append(name)
            
            # Extract keywords
            keywords = []
            keyword_list = result.get('keywords', [])
            for keyword in keyword_list:
                if isinstance(keyword, str):
                    keywords.append(keyword)
            
            # Calculate relevance score (simple implementation)
            relevance_score = self._calculate_relevance_score(title, abstract)
            
            return PubMedPaper(
                pmid=pmid,
                title=title,
                abstract=abstract,
                authors=authors,
                journal=journal,
                publication_date=pub_date,
                doi=doi,
                keywords=keywords,
                relevance_score=relevance_score
            )
            
        except Exception as e:
            logger.error(f"Error fetching paper details for {pmid}: {e}")
            return None
    
    def _is_peer_reviewed(self, paper: PubMedPaper) -> bool:
        """Check if paper is from a peer-reviewed journal"""
        journal_lower = paper.journal.lower()
        
        # Check against known top journals
        for top_journal in self.top_journals:
            if top_journal in journal_lower:
                return True
        
        # Additional checks for peer-reviewed indicators
        peer_reviewed_indicators = [
            'peer reviewed', 'peer-reviewed', 'refereed', 'academic',
            'university', 'medical school', 'research institute'
        ]
        
        for indicator in peer_reviewed_indicators:
            if indicator in journal_lower:
                return True
        
        return False
    
    def _calculate_relevance_score(self, title: str, abstract: str) -> float:
        """Calculate relevance score for a paper"""
        try:
            # Simple relevance scoring based on content length and quality
            title_words = len(title.split())
            abstract_words = len(abstract.split())
            
            # Base score from content length
            score = min(0.5, (title_words + abstract_words) / 1000)
            
            # Bonus for having both title and abstract
            if title and abstract:
                score += 0.2
            
            # Bonus for longer abstracts (more detailed papers)
            if abstract_words > 200:
                score += 0.2
            
            # Bonus for recent papers
            score += 0.1
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating relevance score: {e}")
            return 0.5

class ResearchQueryGenerator:
    """Generates research queries from knowledge base content"""
    
    def __init__(self):
        self.query_templates = {
            'medical': "{topic} clinical trial randomized controlled trial",
            'technical': "{topic} research study systematic review",
            'general': "{topic} peer reviewed research paper",
            'latest': "{topic} latest research 2024"
        }
    
    def generate_research_queries(self, kb_content: str, max_queries: int = 3) -> List[str]:
        """
        Generate research queries from knowledge base content
        
        Args:
            kb_content: Content from the vision chat knowledge base
            max_queries: Maximum number of queries to generate
            
        Returns:
            List of research query strings
        """
        try:
            # Extract key topics from KB content
            topics = self._extract_research_topics(kb_content)
            
            # Generate queries for each topic
            queries = []
            for topic in topics[:max_queries]:
                # Generate different types of queries for each topic
                for query_type, template in self.query_templates.items():
                    query = template.format(topic=topic)
                    queries.append(query)
                    
                    if len(queries) >= max_queries:
                        break
                
                if len(queries) >= max_queries:
                    break
            
            return queries[:max_queries]
            
        except Exception as e:
            logger.error(f"Error generating research queries: {e}")
            return []
    
    def _extract_research_topics(self, content: str) -> List[str]:
        """Extract research topics from content"""
        try:
            # Simple topic extraction based on medical/technical keywords
            medical_keywords = [
                'treatment', 'therapy', 'diagnosis', 'clinical', 'patient',
                'disease', 'condition', 'symptom', 'medication', 'drug',
                'surgery', 'procedure', 'test', 'scan', 'imaging',
                'cancer', 'diabetes', 'heart', 'brain', 'lung', 'liver',
                'infection', 'virus', 'bacteria', 'immune', 'genetic'
            ]
            
            technical_keywords = [
                'algorithm', 'model', 'system', 'technology', 'method',
                'analysis', 'data', 'software', 'hardware', 'network',
                'machine learning', 'artificial intelligence', 'neural network',
                'database', 'protocol', 'framework', 'architecture'
            ]
            
            # Find topics containing these keywords
            topics = []
            sentences = content.split('.')
            
            for sentence in sentences:
                sentence_lower = sentence.lower()
                
                # Check for medical topics
                for keyword in medical_keywords:
                    if keyword in sentence_lower and len(sentence.strip()) > 20:
                        # Extract a phrase around the keyword
                        topic = self._extract_topic_phrase(sentence, keyword)
                        if topic and topic not in topics:
                            topics.append(topic)
                
                # Check for technical topics
                for keyword in technical_keywords:
                    if keyword in sentence_lower and len(sentence.strip()) > 20:
                        topic = self._extract_topic_phrase(sentence, keyword)
                        if topic and topic not in topics:
                            topics.append(topic)
            
            return topics[:5]  # Return top 5 topics
            
        except Exception as e:
            logger.error(f"Error extracting research topics: {e}")
            return []
    
    def _extract_topic_phrase(self, sentence: str, keyword: str) -> str:
        """Extract a topic phrase around a keyword"""
        try:
            # Find the keyword position
            keyword_pos = sentence.lower().find(keyword)
            if keyword_pos == -1:
                return ""
            
            # Extract a phrase around the keyword (up to 5 words on each side)
            words = sentence.split()
            keyword_word_pos = -1
            
            # Find which word contains the keyword
            for i, word in enumerate(words):
                if keyword in word.lower():
                    keyword_word_pos = i
                    break
            
            if keyword_word_pos == -1:
                return ""
            
            # Extract phrase
            start = max(0, keyword_word_pos - 3)
            end = min(len(words), keyword_word_pos + 4)
            phrase_words = words[start:end]
            
            return ' '.join(phrase_words).strip()
            
        except Exception as e:
            logger.error(f"Error extracting topic phrase: {e}")
            return ""

class ResearchKnowledgeBase:
    """Separate knowledge base for research papers"""
    
    def __init__(self, base_path: str = None):
        if base_path is None:
            base_path = os.path.join(os.getcwd(), 'research_knowledge_base')
        
        self.base_path = Path(base_path)
        self.papers_path = self.base_path / 'papers'
        self.metadata_path = self.base_path / 'metadata.json'
        self.index_path = self.base_path / 'index'
        
        # Create directory structure
        self._create_directory_structure()
        
        # Load or create metadata
        self.metadata = self._load_metadata()
    
    def _create_directory_structure(self):
        """Create the research KB directory structure"""
        directories = [self.base_path, self.papers_path, self.index_path]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load or create metadata file"""
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading metadata: {e}")
        
        # Create new metadata
        metadata = {
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'total_papers': 0,
            'papers': {},
            'search_history': [],
            'last_research_run': None
        }
        
        self._save_metadata(metadata)
        return metadata
    
    def _save_metadata(self, metadata: Dict[str, Any]):
        """Save metadata to file"""
        try:
            metadata['last_updated'] = datetime.now().isoformat()
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
    
    def add_paper(self, paper: PubMedPaper) -> bool:
        """Add a research paper to the knowledge base"""
        try:
            # Check if paper already exists
            if paper.pmid in self.metadata['papers']:
                logger.info(f"Paper {paper.pmid} already exists in KB")
                return False
            
            # Save paper data
            paper_file = self.papers_path / f"{paper.pmid}.json"
            with open(paper_file, 'w') as f:
                json.dump(asdict(paper), f, indent=2)
            
            # Update metadata
            self.metadata['papers'][paper.pmid] = {
                'title': paper.title,
                'journal': paper.journal,
                'publication_date': paper.publication_date,
                'relevance_score': paper.relevance_score,
                'search_query': paper.search_query,
                'added_at': datetime.now().isoformat(),
                'file_path': str(paper_file)
            }
            
            self.metadata['total_papers'] = len(self.metadata['papers'])
            self._save_metadata(self.metadata)
            
            logger.info(f"Added paper {paper.pmid} to research KB")
            return True
            
        except Exception as e:
            logger.error(f"Error adding paper to KB: {e}")
            return False
    
    def get_papers_by_query(self, query: str, max_results: int = 5) -> List[PubMedPaper]:
        """Get papers from KB that match a query"""
        try:
            matching_papers = []
            query_lower = query.lower()
            
            for pmid, paper_info in self.metadata['papers'].items():
                title = paper_info.get('title', '').lower()
                search_query = paper_info.get('search_query', '').lower()
                
                # Check if query matches title or search query
                if (query_lower in title or 
                    query_lower in search_query or
                    any(word in title for word in query_lower.split())):
                    
                    # Load paper data
                    paper_file = Path(paper_info.get('file_path', ''))
                    if paper_file.exists():
                        with open(paper_file, 'r') as f:
                            paper_data = json.load(f)
                            paper = PubMedPaper(**paper_data)
                            matching_papers.append(paper)
                    
                    if len(matching_papers) >= max_results:
                        break
            
            # Sort by relevance score
            matching_papers.sort(key=lambda x: x.relevance_score, reverse=True)
            return matching_papers
            
        except Exception as e:
            logger.error(f"Error getting papers by query: {e}")
            return []
    
    def get_all_papers(self) -> List[PubMedPaper]:
        """Get all papers from the KB"""
        try:
            papers = []
            for pmid, paper_info in self.metadata['papers'].items():
                paper_file = Path(paper_info.get('file_path', ''))
                if paper_file.exists():
                    with open(paper_file, 'r') as f:
                        paper_data = json.load(f)
                        paper = PubMedPaper(**paper_data)
                        papers.append(paper)
            
            return papers
            
        except Exception as e:
            logger.error(f"Error getting all papers: {e}")
            return []

class PubMedResearchSystem:
    """Main system for PubMed research integration"""
    
    def __init__(self, vision_kb_path: str = None, research_kb_path: str = None):
        self.pubmed_api = PubMedAPI()
        self.query_generator = ResearchQueryGenerator()
        self.research_kb = ResearchKnowledgeBase(research_kb_path)
        self.vision_kb_path = vision_kb_path
        
        # Configuration
        self.max_papers_per_query = 5
        self.max_queries_per_run = 3
        self.days_back = 365  # Search papers from last year
        
        logger.info("PubMed Research System initialized")
    
    def analyze_vision_kb_and_research(self) -> Dict[str, Any]:
        """
        Analyze vision chat KB and perform research
        
        Returns:
            Dictionary with research results
        """
        try:
            logger.info("Starting vision KB analysis and research")
            
            # Step 1: Extract content from vision KB
            vision_content = self._extract_vision_kb_content()
            if not vision_content:
                logger.warning("No content found in vision KB")
                return {'status': 'no_content', 'papers_added': 0}
            
            # Step 2: Generate research queries
            queries = self.query_generator.generate_research_queries(
                vision_content, 
                max_queries=self.max_queries_per_run
            )
            
            if not queries:
                logger.warning("No research queries generated")
                return {'status': 'no_queries', 'papers_added': 0}
            
            logger.info(f"Generated {len(queries)} research queries")
            
            # Step 3: Search PubMed for each query
            total_papers_added = 0
            search_results = []
            
            for query in queries:
                logger.info(f"Searching PubMed for: {query}")
                
                papers = self.pubmed_api.search_papers(
                    query, 
                    max_results=self.max_papers_per_query,
                    days_back=self.days_back
                )
                
                # Add papers to research KB
                papers_added = 0
                for paper in papers:
                    paper.search_query = query
                    if self.research_kb.add_paper(paper):
                        papers_added += 1
                
                total_papers_added += papers_added
                search_results.append({
                    'query': query,
                    'papers_found': len(papers),
                    'papers_added': papers_added
                })
                
                # Rate limiting - be respectful to PubMed
                time.sleep(1)
            
            # Update last research run
            self.research_kb.metadata['last_research_run'] = datetime.now().isoformat()
            self.research_kb._save_metadata(self.research_kb.metadata)
            
            result = {
                'status': 'success',
                'total_papers_added': total_papers_added,
                'queries_searched': len(queries),
                'search_results': search_results,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Research completed. Added {total_papers_added} papers to research KB")
            return result
            
        except Exception as e:
            logger.error(f"Error in research analysis: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _extract_vision_kb_content(self) -> str:
        """Extract content from vision chat knowledge base"""
        try:
            if not self.vision_kb_path:
                # Try to find vision KB automatically
                possible_paths = [
                    os.path.join(os.getcwd(), 'uploads', 'knowledge_bases'),
                    os.path.join(os.getcwd(), 'data', 'knowledge_bases'),
                    os.path.join(os.getcwd(), 'knowledge_bases')
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        self.vision_kb_path = path
                        break
            
            if not self.vision_kb_path or not os.path.exists(self.vision_kb_path):
                logger.warning("Vision KB path not found")
                return ""
            
            # Extract content from all knowledge bases
            content = []
            
            for session_dir in os.listdir(self.vision_kb_path):
                session_path = os.path.join(self.vision_kb_path, session_dir)
                if os.path.isdir(session_path):
                    # Look for metadata files
                    metadata_file = os.path.join(session_path, 'metadata.json')
                    if os.path.exists(metadata_file):
                        try:
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)
                            
                            # Extract content from processed files
                            for content_id, content_info in metadata.get('processed_content', {}).items():
                                processed_file = content_info.get('file', '')
                                if os.path.exists(processed_file):
                                    with open(processed_file, 'r') as f:
                                        content_data = json.load(f)
                                        
                                        if content_data.get('type') == 'pdf':
                                            # Extract text from PDF pages
                                            for page_data in content_data.get('pages', {}).values():
                                                content.append(page_data.get('text', ''))
                                                content.append(page_data.get('enhanced_content', ''))
                                        else:
                                            # Extract general content
                                            content.append(content_data.get('content', ''))
                                            content.append(content_data.get('description', ''))
                            
                        except Exception as e:
                            logger.warning(f"Error reading metadata from {session_dir}: {e}")
                            continue
            
            return ' '.join(content)
            
        except Exception as e:
            logger.error(f"Error extracting vision KB content: {e}")
            return ""
    
    def get_research_context_for_query(self, user_query: str, max_papers: int = 3) -> str:
        """
        Get relevant research context for a user query
        
        Args:
            user_query: The user's query
            max_papers: Maximum number of papers to include
            
        Returns:
            Formatted research context string
        """
        try:
            # Get relevant papers from research KB
            papers = self.research_kb.get_papers_by_query(user_query, max_papers)
            
            if not papers:
                return ""
            
            # Format research context
            context = "\n\nRESEARCH PAPERS CONTEXT:\n"
            context += "=" * 50 + "\n"
            
            for i, paper in enumerate(papers, 1):
                context += f"[Research Paper {i}]\n"
                context += f"Title: {paper.title}\n"
                context += f"Journal: {paper.journal}\n"
                context += f"Authors: {', '.join(paper.authors[:3])}\n"
                context += f"Abstract: {paper.abstract[:300]}...\n"
                context += f"Relevance Score: {paper.relevance_score:.2f}\n"
                context += f"PMID: {paper.pmid}\n"
                if paper.doi:
                    context += f"DOI: {paper.doi}\n"
                context += "-" * 30 + "\n\n"
            
            return context
            
        except Exception as e:
            logger.error(f"Error getting research context: {e}")
            return ""

# Global instance for easy access
_research_system = None

def get_research_system(vision_kb_path: str = None) -> PubMedResearchSystem:
    """Get or create global research system instance"""
    global _research_system
    if _research_system is None:
        _research_system = PubMedResearchSystem(vision_kb_path)
    return _research_system

def run_research_analysis(vision_kb_path: str = None) -> Dict[str, Any]:
    """Run research analysis on vision KB"""
    system = get_research_system(vision_kb_path)
    return system.analyze_vision_kb_and_research()

def get_research_context(user_query: str, max_papers: int = 3) -> str:
    """Get research context for a user query"""
    system = get_research_system()
    return system.get_research_context_for_query(user_query, max_papers)

def schedule_research_analysis(vision_kb_path: str = None, time: str = "02:00"):
    """
    Schedule daily research analysis using application-level scheduling
    
    This method uses Python's schedule library and works while the application is running.
    For system-level scheduling that works even when the app is closed, use system_scheduler.py:
    
    python system_scheduler.py setup    # Setup system scheduler
    python system_scheduler.py status   # Check status
    python system_scheduler.py remove   # Remove scheduler
    
    Args:
        vision_kb_path: Path to vision KB
        time: Time to run analysis (24-hour format, default 2 AM)
    """
    def run_analysis():
        logger.info("Running scheduled research analysis")
        result = run_research_analysis(vision_kb_path)
        logger.info(f"Scheduled analysis completed: {result}")
    
    schedule.every().day.at(time).do(run_analysis)
    logger.info(f"Application-level research analysis scheduled for daily at {time}")
    
    # Start the scheduler in a separate thread
    def run_scheduler():
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
    
    logger.info("Application-level scheduler started. Note: This only works while the app is running.")
    logger.info("For system-level scheduling (works when app is closed), use: python system_scheduler.py setup") 