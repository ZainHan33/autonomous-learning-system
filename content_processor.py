import re
from typing import List, Dict, Set, Tuple
import spacy
from html import unescape
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class ProcessedContent:
    main_concepts: List[str]
    key_points: List[str]
    related_topics: List[str]
    definitions: Dict[str, str]
    relations: List[Tuple[str, str, str]]  # (entity1, relation, entity2)

class ContentProcessor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.stopwords = set(['the', 'a', 'an', 'and', 'or', 'but', 'for', 'with', 'in', 'on', 'at'])
        self.topic_blocklist = {'the laws', 'a', 'an', 'the', 'services', 'technology'}
        
    def clean_text(self, text: str) -> str:
        """Deep clean text content"""
        # Unescape HTML entities
        text = unescape(text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Fix common formatting issues
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces
        text = re.sub(r'[\r\n]+', '\n', text)  # Multiple newlines
        
        # Remove URLs
        text = re.sub(r'http[s]?://\S+', '', text)
        
        return text.strip()
    
    def _is_valid_concept(self, text: str) -> bool:
        """Check if concept is meaningful"""
        if text.lower() in self.topic_blocklist:
            return False
        if len(text.split()) < 2 and not any(char.isdigit() for char in text):
            return False
        if all(word in self.stopwords for word in text.lower().split()):
            return False
        return True
    
    def extract_concepts(self, text: str) -> List[str]:
        """Extract main concepts from text"""
        doc = self.nlp(text)
        concepts = []
        seen = set()
        
        # Extract noun phrases and named entities
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) >= 2:  # At least two words
                clean_chunk = chunk.text.lower().strip()
                if clean_chunk not in seen and self._is_valid_concept(clean_chunk):
                    concepts.append(clean_chunk)
                    seen.add(clean_chunk)
                
        for ent in doc.ents:
            if ent.label_ in ['PRODUCT', 'ORG', 'TECHNOLOGY', 'GPE', 'DATE', 'MONEY']:
                clean_ent = ent.text.lower().strip()
                if clean_ent not in seen and self._is_valid_concept(clean_ent):
                    concepts.append(clean_ent)
                    seen.add(clean_ent)
        
        return concepts
    
    def extract_key_points(self, text: str) -> List[str]:
        """Extract key points from text"""
        doc = self.nlp(text)
        points = []
        seen = set()
        
        for sent in doc.sents:
            # Look for sentences with key information markers
            text = sent.text.strip()
            if any(marker in text.lower() for marker in [
                'is', 'are', 'means', 'refers to', 'defined as',
                'consists of', 'involves', 'uses', 'enables'
            ]):
                if text not in seen:
                    points.append(text)
                    seen.add(text)
                
        return points
    
    def find_definitions(self, text: str) -> Dict[str, str]:
        """Find term definitions in text"""
        definitions = {}
        doc = self.nlp(text)
        
        for sent in doc.sents:
            # Look for definition patterns
            text = sent.text.strip()
            matches = re.finditer(
                r'([A-Za-z\s]+(?:computing|technology|system|processor|algorithm|method))'
                r'\s+(?:is|refers to|means|defines)\s+([^.]+)',
                text,
                re.IGNORECASE
            )
            
            for match in matches:
                term = match.group(1).strip().lower()
                definition = match.group(2).strip()
                if self._is_valid_concept(term):
                    definitions[term] = definition
                
        return definitions
    
    def extract_relations(self, text: str) -> List[Tuple[str, str, str]]:
        """Extract relationships between concepts"""
        doc = self.nlp(text)
        relations = []
        
        for sent in doc.sents:
            for token in sent:
                if token.dep_ == "ROOT":
                    # Find subject
                    subj = None
                    for child in token.children:
                        if child.dep_ in ["nsubj", "nsubjpass"]:
                            subj = child.text
                            
                    # Find object
                    obj = None
                    for child in token.children:
                        if child.dep_ in ["dobj", "pobj"]:
                            obj = child.text
                            
                    if subj and obj and token.text:
                        relations.append((subj.lower(), token.text.lower(), obj.lower()))
                        
        return relations
    
    def process_content(self, text: str) -> ProcessedContent:
        """Process content and extract structured information"""
        cleaned_text = self.clean_text(text)
        
        # Extract all components
        concepts = self.extract_concepts(cleaned_text)
        points = self.extract_key_points(cleaned_text)
        definitions = self.find_definitions(cleaned_text)
        relations = self.extract_relations(cleaned_text)
        
        # Filter related topics to remove noise
        related_topics = [
            concept for concept in concepts 
            if concept not in self.topic_blocklist and len(concept.split()) > 1
        ]
        
        return ProcessedContent(
            main_concepts=concepts,
            key_points=points,
            related_topics=related_topics,
            definitions=definitions,
            relations=relations
        )