from typing import Dict, List, Optional
import pandas as pd
import requests
from datetime import datetime, timedelta
import logging
from pathlib import Path
import yaml
import json
from textblob import TextBlob
from bs4 import BeautifulSoup
import re
from urllib.parse import quote_plus
import time

class NewsAgent:
    def __init__(self, config_path: Optional[str] = None, llm_config: Optional[Dict] = None):
        """
        Initialize the News Agent.
        
        Args:
            config_path (str, optional): Path to configuration file
            llm_config (dict, optional): Configuration for LLM
        """
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
        self.config = self.load_config(config_path)
        self.llm_config = llm_config or {}
        
    def setup_logging(self):
        """Configure logging."""
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f'news_agent_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
    
    def load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from YAML file."""
        default_config = {
            'news_sources': [
                'google_news',
                'reuters',
                'bloomberg'
            ],
            'search_window_days': 30,
            'max_results_per_source': 10,
            'headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
                if loaded_config and 'news' in loaded_config:
                    default_config.update(loaded_config['news'])
        return default_config

    def search_google_news(self, query: str) -> List[Dict]:
        """
        Search Google News for relevant articles.
        
        Args:
            query (str): Search query
            
        Returns:
            List[Dict]: Relevant articles
        """
        try:
            encoded_query = quote_plus(query)
            url = f"https://news.google.com/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
            
            response = requests.get(url, headers=self.config['headers'])
            soup = BeautifulSoup(response.text, 'html.parser')
            
            articles = []
            for article in soup.find_all('article', limit=self.config['max_results_per_source']):
                try:
                    title_elem = article.find('h3')
                    link_elem = article.find('a')
                    time_elem = article.find('time')
                    
                    if title_elem and link_elem:
                        title = title_elem.text.strip()
                        link = 'https://news.google.com' + link_elem['href'][1:] if link_elem['href'].startswith('.') else link_elem['href']
                        published_time = time_elem['datetime'] if time_elem else None
                        
                        # Get article content
                        article_content = self.get_article_content(link)
                        
                        articles.append({
                            'title': title,
                            'url': link,
                            'published_time': published_time,
                            'content': article_content,
                            'sentiment': self.analyze_sentiment(title + ' ' + (article_content or ''))
                        })
                except Exception as e:
                    self.logger.warning(f"Error processing article: {str(e)}")
                    continue
                
                # Be nice to the servers
                time.sleep(1)
            
            return articles
        except Exception as e:
            self.logger.error(f"Error searching Google News: {str(e)}")
            return []

    def get_article_content(self, url: str) -> Optional[str]:
        """
        Get the content of an article.
        
        Args:
            url (str): Article URL
            
        Returns:
            Optional[str]: Article content
        """
        try:
            response = requests.get(url, headers=self.config['headers'], timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for element in soup.find_all(['script', 'style', 'nav', 'footer', 'header']):
                element.decompose()
            
            # Get main content
            content = []
            for p in soup.find_all('p'):
                text = p.get_text().strip()
                if text and len(text) > 50:  # Only keep substantial paragraphs
                    content.append(text)
            
            return ' '.join(content) if content else None
        except Exception as e:
            self.logger.warning(f"Error getting article content: {str(e)}")
            return None

    def analyze_sentiment(self, text: str) -> Dict:
        """
        Analyze sentiment of text.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict: Sentiment analysis results
        """
        try:
            blob = TextBlob(text)
            return {
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity,
                'sentiment': 'positive' if blob.sentiment.polarity > 0 else 'negative' if blob.sentiment.polarity < 0 else 'neutral'
            }
        except Exception as e:
            self.logger.warning(f"Error analyzing sentiment: {str(e)}")
            return {'polarity': 0, 'subjectivity': 0, 'sentiment': 'neutral'}

    def run(self, forecast_context: str = "") -> Dict:
        """
        Run the news collection and analysis.
        
        Args:
            forecast_context (str): Context from forecasting results
            
        Returns:
            Dict: News analysis results
        """
        try:
            # Extract key terms from forecast context
            search_terms = self.extract_search_terms(forecast_context)
            
            # Collect news for each term
            all_news = []
            for term in search_terms:
                news = self.search_google_news(term)
                all_news.extend(news)
            
            # Remove duplicates
            seen_urls = set()
            unique_news = []
            for article in all_news:
                if article['url'] not in seen_urls:
                    seen_urls.add(article['url'])
                    unique_news.append(article)
            
            # Sort by relevance and recency
            unique_news.sort(key=lambda x: (
                x['sentiment']['polarity'] * 0.3 +  # Sentiment weight
                (1 if x['content'] else 0) * 0.7,   # Content availability weight
                x['published_time'] if x['published_time'] else '1970-01-01'  # Recency
            ), reverse=True)
            
            # Prepare results
            results = {
                'articles': unique_news[:self.config['max_results_per_source']],
                'search_terms': search_terms,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Save results
            self.save_results(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in news collection: {str(e)}")
            return {'articles': [], 'search_terms': [], 'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

    def extract_search_terms(self, context: str) -> List[str]:
        """
        Extract relevant search terms from forecast context.
        
        Args:
            context (str): Forecast context
            
        Returns:
            List[str]: Search terms
        """
        # Basic term extraction - can be enhanced with NLP
        terms = []
        
        # Add default terms
        terms.extend(['market trends', 'industry news', 'economic outlook'])
        
        # Extract terms from context
        if context:
            # Split into words and filter
            words = re.findall(r'\b\w+\b', context.lower())
            significant_words = [w for w in words if len(w) > 3 and w not in {'this', 'that', 'with', 'from', 'have', 'will'}]
            
            # Add significant terms
            if significant_words:
                terms.extend(significant_words)
        
        return list(set(terms))  # Remove duplicates

    def save_results(self, results: Dict):
        """Save collected news and data."""
        output_dir = Path('output/news')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f'news_results_{timestamp}.json'
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4, default=str)
        
        self.logger.info(f"News results saved to {output_file}") 