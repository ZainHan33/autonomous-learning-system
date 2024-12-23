import requests
import asyncio
from typing import List, Dict
from datetime import datetime
import re
from dataclasses import dataclass
from config import BRAVE_API_KEY, BRAVE_SEARCH_URL

@dataclass
class SearchResult:
    title: str
    description: str
    url: str
    timestamp: datetime

class BraveSearchManager:
    def __init__(self):
        self.headers = {
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.5',
            'X-Subscription-Token': BRAVE_API_KEY
        }
        self.last_request_time = datetime.min
        self.min_request_interval = 1.0  # Minimum seconds between requests
        
    def _clean_html(self, text: str) -> str:
        """Remove HTML tags and decode entities"""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Fix common HTML entities
        text = text.replace('&quot;', '"').replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
        return text.strip()
        
    async def search(self, query: str, count: int = 5) -> List[SearchResult]:
        """
        Perform a web search using Brave Search API with rate limiting
        """
        try:
            # Implement rate limiting
            now = datetime.now()
            time_since_last = (now - self.last_request_time).total_seconds()
            if time_since_last < self.min_request_interval:
                await asyncio.sleep(self.min_request_interval - time_since_last)
            
            params = {
                'q': query,
                'count': count,
                'freshness': 'month',  # Get recent results
                'textDecorations': 'false',  # Disable highlighting
                'safesearch': 'moderate'
            }
            
            self.last_request_time = datetime.now()
            response = requests.get(
                BRAVE_SEARCH_URL,
                headers=self.headers,
                params=params
            )
            
            if response.status_code == 429:
                print("Rate limit hit, waiting before retry...")
                await asyncio.sleep(5)  # Wait 5 seconds
                return await self.search(query, count)  # Retry
                
            if response.status_code != 200:
                print(f"Search error: {response.status_code}")
                return []
                
            data = response.json()
            results = []
            
            for item in data.get('web', {}).get('results', []):
                results.append(SearchResult(
                    title=self._clean_html(item.get('title', '')),
                    description=self._clean_html(item.get('description', '')),
                    url=item.get('url', ''),
                    timestamp=datetime.now()
                ))
                
            return results
            
        except Exception as e:
            print(f"Search error: {str(e)}")
            return []
