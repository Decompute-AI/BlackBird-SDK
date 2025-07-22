#!/usr/bin/env python3
"""
Focused test script for search query generation logic
Tests how user queries are enhanced with knowledge base context
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

class QueryGenerationTester:
    """Tester for search query generation logic"""
    
    def __init__(self):
        self.test_cases = [
            # Medical/Health queries
            {
                'user_query': 'What are the symptoms of diabetes?',
                'kb_context': 'User has uploaded medical documents about diabetes treatment and management.',
                'expected_keywords': ['diabetes', 'symptoms', 'medical', 'treatment'],
                'category': 'medical'
            },
            {
                'user_query': 'COVID-19 vaccine side effects',
                'kb_context': 'Medical research papers about COVID-19 vaccines and clinical trials.',
                'expected_keywords': ['covid-19', 'vaccine', 'side effects', 'clinical trials'],
                'category': 'medical'
            },
            
            # Technical queries
            {
                'user_query': 'How to implement authentication in React?',
                'kb_context': 'Technical documentation about React development and web security.',
                'expected_keywords': ['react', 'authentication', 'web security', 'development'],
                'category': 'technical'
            },
            {
                'user_query': 'Docker container best practices',
                'kb_context': 'DevOps documentation about containerization and deployment.',
                'expected_keywords': ['docker', 'container', 'best practices', 'deployment'],
                'category': 'technical'
            },
            
            # Research queries
            {
                'user_query': 'Latest developments in renewable energy',
                'kb_context': 'Research papers about climate change and renewable energy technologies.',
                'expected_keywords': ['renewable energy', 'climate change', 'research', 'technologies'],
                'category': 'research'
            },
            {
                'user_query': 'Machine learning applications in healthcare',
                'kb_context': 'Academic papers about AI and machine learning in medical diagnosis.',
                'expected_keywords': ['machine learning', 'healthcare', 'AI', 'medical diagnosis'],
                'category': 'research'
            },
            
            # Business queries
            {
                'user_query': 'Blockchain use cases in finance',
                'kb_context': 'Business documents about cryptocurrency and financial technology.',
                'expected_keywords': ['blockchain', 'finance', 'cryptocurrency', 'fintech'],
                'category': 'business'
            },
            {
                'user_query': 'Supply chain optimization strategies',
                'kb_context': 'Business reports about logistics and supply chain management.',
                'expected_keywords': ['supply chain', 'optimization', 'logistics', 'management'],
                'category': 'business'
            },
            
            # Complex queries
            {
                'user_query': 'Compare React vs Vue.js for enterprise applications',
                'kb_context': 'Technical documentation about frontend frameworks and enterprise development.',
                'expected_keywords': ['react', 'vue.js', 'enterprise', 'frontend frameworks'],
                'category': 'complex'
            },
            {
                'user_query': 'Best practices for microservices architecture in cloud environments',
                'kb_context': 'Architecture documentation about distributed systems and cloud computing.',
                'expected_keywords': ['microservices', 'architecture', 'cloud', 'distributed systems'],
                'category': 'complex'
            }
        ]
    
    def test_basic_query_generation(self) -> Dict[str, Any]:
        """Test basic query generation without KB context"""
        print("🔍 Testing Basic Query Generation")
        print("-" * 40)
        
        results = []
        for i, test_case in enumerate(self.test_cases):
            user_query = test_case['user_query']
            
            # Test basic query (no enhancement)
            basic_query = self._generate_basic_query(user_query)
            
            # Check if basic query contains key terms from user query
            user_words = set(user_query.lower().split())
            basic_words = set(basic_query.lower().split())
            overlap = len(user_words.intersection(basic_words))
            
            success = overlap >= len(user_words) * 0.7  # At least 70% overlap
            
            result = {
                'test_id': i + 1,
                'user_query': user_query,
                'generated_query': basic_query,
                'word_overlap': overlap,
                'total_user_words': len(user_words),
                'success': success,
                'category': test_case['category']
            }
            results.append(result)
            
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{status} Test {i+1}: {user_query[:50]}...")
            print(f"   Generated: {basic_query[:60]}...")
            print(f"   Overlap: {overlap}/{len(user_words)} words")
            print()
        
        return results
    
    def test_enhanced_query_generation(self) -> Dict[str, Any]:
        """Test query generation with KB context enhancement"""
        print("🔍 Testing Enhanced Query Generation (with KB Context)")
        print("-" * 60)
        
        results = []
        for i, test_case in enumerate(self.test_cases):
            user_query = test_case['user_query']
            kb_context = test_case['kb_context']
            expected_keywords = test_case['expected_keywords']
            
            # Test enhanced query with KB context
            enhanced_query = self._generate_enhanced_query(user_query, kb_context)
            
            # Check if enhanced query contains expected keywords
            query_lower = enhanced_query.lower()
            keyword_matches = sum(1 for keyword in expected_keywords if keyword.lower() in query_lower)
            keyword_score = keyword_matches / len(expected_keywords)
            
            # Check if enhanced query is longer than basic query (should be enhanced)
            basic_query = self._generate_basic_query(user_query)
            enhancement_ratio = len(enhanced_query.split()) / len(basic_query.split())
            
            success = keyword_score >= 0.6 and enhancement_ratio >= 1.2  # At least 60% keywords and 20% longer
            
            result = {
                'test_id': i + 1,
                'user_query': user_query,
                'kb_context': kb_context,
                'enhanced_query': enhanced_query,
                'expected_keywords': expected_keywords,
                'keyword_matches': keyword_matches,
                'keyword_score': keyword_score,
                'enhancement_ratio': enhancement_ratio,
                'success': success,
                'category': test_case['category']
            }
            results.append(result)
            
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{status} Test {i+1}: {user_query[:40]}...")
            print(f"   Enhanced: {enhanced_query[:70]}...")
            print(f"   Keywords: {keyword_matches}/{len(expected_keywords)} matched")
            print(f"   Enhancement: {enhancement_ratio:.2f}x longer")
            print()
        
        return results
    
    def test_query_relevance_scoring(self) -> Dict[str, Any]:
        """Test relevance scoring of generated queries"""
        print("🔍 Testing Query Relevance Scoring")
        print("-" * 40)
        
        results = []
        for i, test_case in enumerate(self.test_cases):
            user_query = test_case['user_query']
            kb_context = test_case['kb_context']
            expected_keywords = test_case['expected_keywords']
            
            # Generate enhanced query
            enhanced_query = self._generate_enhanced_query(user_query, kb_context)
            
            # Calculate relevance score
            relevance_score = self._calculate_relevance_score(enhanced_query, expected_keywords)
            
            # Determine if score is reasonable
            success = 0.3 <= relevance_score <= 1.0  # Reasonable range
            
            result = {
                'test_id': i + 1,
                'user_query': user_query,
                'enhanced_query': enhanced_query,
                'expected_keywords': expected_keywords,
                'relevance_score': relevance_score,
                'success': success,
                'category': test_case['category']
            }
            results.append(result)
            
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{status} Test {i+1}: {user_query[:40]}...")
            print(f"   Relevance Score: {relevance_score:.3f}")
            print(f"   Expected Keywords: {', '.join(expected_keywords)}")
            print()
        
        return results
    
    def test_query_diversity(self) -> Dict[str, Any]:
        """Test that different queries generate different results"""
        print("🔍 Testing Query Diversity")
        print("-" * 30)
        
        # Test with similar queries to ensure diversity
        similar_queries = [
            "What is machine learning?",
            "How does machine learning work?",
            "Machine learning applications",
            "Machine learning algorithms"
        ]
        
        kb_context = "Technical documentation about AI and machine learning."
        
        generated_queries = []
        for query in similar_queries:
            enhanced_query = self._generate_enhanced_query(query, kb_context)
            generated_queries.append(enhanced_query)
        
        # Calculate diversity (how different the queries are)
        diversity_scores = []
        for i in range(len(generated_queries)):
            for j in range(i + 1, len(generated_queries)):
                similarity = self._calculate_similarity(generated_queries[i], generated_queries[j])
                diversity_scores.append(1 - similarity)
        
        avg_diversity = sum(diversity_scores) / len(diversity_scores) if diversity_scores else 0
        success = avg_diversity >= 0.3  # At least 30% different on average
        
        result = {
            'similar_queries': similar_queries,
            'generated_queries': generated_queries,
            'diversity_scores': diversity_scores,
            'average_diversity': avg_diversity,
            'success': success
        }
        
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} Query Diversity Test")
        print(f"   Average Diversity: {avg_diversity:.3f}")
        print(f"   Number of Comparisons: {len(diversity_scores)}")
        print()
        
        for i, (original, generated) in enumerate(zip(similar_queries, generated_queries)):
            print(f"   Query {i+1}: {original}")
            print(f"   Generated: {generated}")
            print()
        
        return result
    
    def _generate_basic_query(self, user_query: str) -> str:
        """Generate basic query without enhancement"""
        # This is a simple implementation - replace with actual logic
        return user_query.strip()
    
    def _generate_enhanced_query(self, user_query: str, kb_context: str) -> str:
        """Generate enhanced query with KB context"""
        # This is a simple implementation - replace with actual logic
        # Extract key terms from KB context
        kb_words = kb_context.lower().split()
        # Take first few meaningful words from KB context
        kb_keywords = [word for word in kb_words[:5] if len(word) > 3]
        
        # Combine user query with KB keywords
        enhanced_parts = [user_query]
        if kb_keywords:
            enhanced_parts.extend(kb_keywords[:3])  # Add up to 3 KB keywords
        
        return " ".join(enhanced_parts)
    
    def _calculate_relevance_score(self, query: str, expected_keywords: List[str]) -> float:
        """Calculate relevance score for a query"""
        query_lower = query.lower()
        matches = sum(1 for keyword in expected_keywords if keyword.lower() in query_lower)
        return matches / len(expected_keywords) if expected_keywords else 0
    
    def _calculate_similarity(self, query1: str, query2: str) -> float:
        """Calculate similarity between two queries"""
        words1 = set(query1.lower().split())
        words2 = set(query2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all query generation tests"""
        print("🚀 Starting Query Generation Test Suite")
        print("=" * 60)
        print()
        
        # Run all tests
        basic_results = self.test_basic_query_generation()
        enhanced_results = self.test_enhanced_query_generation()
        relevance_results = self.test_query_relevance_scoring()
        diversity_result = self.test_query_diversity()
        
        # Calculate overall statistics
        basic_success = sum(1 for r in basic_results if r['success'])
        enhanced_success = sum(1 for r in enhanced_results if r['success'])
        relevance_success = sum(1 for r in relevance_results if r['success'])
        
        total_tests = len(basic_results) + len(enhanced_results) + len(relevance_results) + 1
        total_success = basic_success + enhanced_success + relevance_success + (1 if diversity_result['success'] else 0)
        
        success_rate = (total_success / total_tests) * 100
        
        # Generate summary
        summary = {
            'total_tests': total_tests,
            'passed_tests': total_success,
            'failed_tests': total_tests - total_success,
            'success_rate': success_rate,
            'test_breakdown': {
                'basic_generation': {
                    'total': len(basic_results),
                    'passed': basic_success,
                    'success_rate': (basic_success / len(basic_results)) * 100 if basic_results else 0
                },
                'enhanced_generation': {
                    'total': len(enhanced_results),
                    'passed': enhanced_success,
                    'success_rate': (enhanced_success / len(enhanced_results)) * 100 if enhanced_results else 0
                },
                'relevance_scoring': {
                    'total': len(relevance_results),
                    'passed': relevance_success,
                    'success_rate': (relevance_success / len(relevance_results)) * 100 if relevance_results else 0
                },
                'query_diversity': {
                    'total': 1,
                    'passed': 1 if diversity_result['success'] else 0,
                    'success_rate': 100 if diversity_result['success'] else 0
                }
            },
            'detailed_results': {
                'basic_results': basic_results,
                'enhanced_results': enhanced_results,
                'relevance_results': relevance_results,
                'diversity_result': diversity_result
            }
        }
        
        # Print summary
        print("=" * 60)
        print("📊 QUERY GENERATION TEST SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {total_success}")
        print(f"Failed: {total_tests - total_success}")
        print(f"Success Rate: {success_rate:.1f}%")
        print()
        
        print("📈 Test Breakdown:")
        print(f"  Basic Generation: {basic_success}/{len(basic_results)} ({summary['test_breakdown']['basic_generation']['success_rate']:.1f}%)")
        print(f"  Enhanced Generation: {enhanced_success}/{len(enhanced_results)} ({summary['test_breakdown']['enhanced_generation']['success_rate']:.1f}%)")
        print(f"  Relevance Scoring: {relevance_success}/{len(relevance_results)} ({summary['test_breakdown']['relevance_scoring']['success_rate']:.1f}%)")
        print(f"  Query Diversity: {'PASS' if diversity_result['success'] else 'FAIL'}")
        print()
        
        if success_rate >= 80:
            print("🎉 Overall Status: EXCELLENT")
        elif success_rate >= 60:
            print("✅ Overall Status: GOOD")
        elif success_rate >= 40:
            print("⚠️ Overall Status: FAIR")
        else:
            print("❌ Overall Status: POOR")
        
        return summary
    
    def save_results(self, results: Dict[str, Any], filename: str = None):
        """Save test results to file"""
        if filename is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"query_generation_test_results_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"📄 Test results saved to: {filename}")
        except Exception as e:
            print(f"❌ Error saving results: {e}")

def main():
    """Main function"""
    print("🔍 Query Generation Test Suite")
    print("This script tests search query generation logic.")
    print()
    
    # Run tests
    tester = QueryGenerationTester()
    results = tester.run_all_tests()
    
    # Save results
    tester.save_results(results)
    
    # Print recommendations
    print("\n💡 RECOMMENDATIONS:")
    if results['success_rate'] < 60:
        print("- Review query generation algorithm")
        print("- Check keyword extraction logic")
        print("- Improve KB context integration")
        print("- Test with more diverse query types")
    elif results['success_rate'] < 80:
        print("- Some tests failed, review specific areas")
        print("- Consider improving relevance scoring")
        print("- Test with edge cases")
    else:
        print("- Query generation is working well!")
        print("- Consider adding more test cases")
        print("- Monitor performance in production")

if __name__ == "__main__":
    main() 