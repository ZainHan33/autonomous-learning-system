import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Tuple
import torch.nn.functional as F
from dataclasses import dataclass
from datetime import datetime
import asyncio
from search_manager import BraveSearchManager, SearchResult
from pattern_recognition import PatternRecognizer

@dataclass
class LearningResult:
    topic: str
    confidence: float
    source: str
    content: str
    timestamp: datetime
    patterns: List[str]

class LearningManager:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.search_manager = BraveSearchManager()
        self.pattern_recognizer = PatternRecognizer()
        self.knowledge_base = {}
        self.learning_confidence_threshold = 0.7
        
    async def learn_topic(self, topic: str, depth: int = 3) -> List[LearningResult]:
        """
        Autonomously learn about a topic through iterative search and pattern recognition
        """
        results = []
        search_queries = [topic]  # Start with the main topic
        processed_urls = set()
        
        for depth_level in range(depth):
            for query in search_queries:
                # Search for information
                search_results = await self.search_manager.search(
                    query=query, 
                    count=5
                )
                
                # Process and validate each result
                for result in search_results:
                    if result.url in processed_urls:
                        continue
                        
                    processed_urls.add(result.url)
                    
                    # Analyze content
                    confidence = self._evaluate_content_relevance(
                        query=query,
                        content=result.description
                    )
                    
                    if confidence > self.learning_confidence_threshold:
                        # Find patterns in the content
                        patterns = self.pattern_recognizer.find_patterns([
                            result.title,
                            result.description
                        ])
                        
                        # Store the learning result
                        learning_result = LearningResult(
                            topic=query,
                            confidence=confidence,
                            source=result.url,
                            content=result.description,
                            timestamp=datetime.now(),
                            patterns=list(patterns.values())[0] if patterns else []
                        )
                        
                        results.append(learning_result)
                        
                        # Extract new topics to explore
                        new_queries = self._extract_related_topics(
                            content=result.description,
                            original_query=query
                        )
                        search_queries.extend(new_queries)
            
            # Remove duplicates and limit queries
            search_queries = list(set(search_queries))[:5]
        
        # Update knowledge base
        self._update_knowledge_base(results)
        
        return results
    
    def _evaluate_content_relevance(self, query: str, content: str) -> float:
        """
        Evaluate how relevant content is to the query
        """
        try:
            similarity = self.pattern_recognizer.calculate_similarity(query, content)
            return float(similarity)
        except Exception as e:
            print(f"Error evaluating content relevance: {str(e)}")
            return 0.0
    
    def _extract_related_topics(self, content: str, original_query: str) -> List[str]:
        """
        Extract related topics from content for further exploration
        """
        try:
            # Use pattern recognition to find related concepts
            patterns = self.pattern_recognizer.find_patterns([content])
            if not patterns:
                return []
                
            # Extract topics from patterns
            topics = []
            for pattern_group in patterns.values():
                for text in pattern_group:
                    if text != original_query:
                        topics.append(text)
            
            return topics[:3]  # Limit to top 3 related topics
            
        except Exception as e:
            print(f"Error extracting related topics: {str(e)}")
            return []
    
    def _update_knowledge_base(self, results: List[LearningResult]):
        """
        Update the knowledge base with new learning results
        """
        for result in results:
            if result.topic not in self.knowledge_base:
                self.knowledge_base[result.topic] = []
            
            # Add new information
            self.knowledge_base[result.topic].append({
                'content': result.content,
                'confidence': result.confidence,
                'source': result.source,
                'timestamp': result.timestamp,
                'patterns': result.patterns
            })
            
            # Sort by confidence
            self.knowledge_base[result.topic].sort(
                key=lambda x: x['confidence'],
                reverse=True
            )
            
            # Keep only top 10 most confident results per topic
            self.knowledge_base[result.topic] = \
                self.knowledge_base[result.topic][:10]
    
    def get_topic_knowledge(self, topic: str) -> List[Dict]:
        """
        Retrieve current knowledge about a specific topic
        """
        return self.knowledge_base.get(topic, [])
    
    def get_related_topics(self, topic: str) -> List[str]:
        """
        Get list of related topics that have been learned
        """
        related = []
        if topic in self.knowledge_base:
            # Look through patterns in topic's knowledge
            for entry in self.knowledge_base[topic]:
                for pattern in entry['patterns']:
                    if pattern != topic:
                        related.append(pattern)
        return list(set(related))  # Remove duplicates
