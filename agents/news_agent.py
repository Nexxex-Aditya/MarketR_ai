import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
import re
from collections import defaultdict
from utils.llm_utils import call_llm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewsAgent:
    """Enhanced news analysis agent"""
    
    def __init__(self, query: str = "general market trends", max_articles: int = 10, 
                 api_key: Optional[str] = None, llm_config: Optional[Dict[str, Any]] = None):
        self.query = query
        self.max_articles = max_articles
        self.articles = []
        self.summary = None
        self.llm_insights = None
        # In a real application, consider using a news API like NewsAPI, GNews API, etc.
        # For this example, we'll demonstrate a placeholder for API integration.
        self.api_key = api_key # API key for a news service like NewsAPI
        self.NEWS_API_BASE_URL = "https://newsapi.org/v2/everything" # Example for NewsAPI
        self.llm_config = llm_config # Store LLM configuration
        
        self.news_sources = [
            'reuters.com',
            'bloomberg.com',
            'wsj.com',
            'ft.com',
            'cnbc.com'
        ]
        self.keywords = set()
        self.sentiment_scores = defaultdict(list)
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from text"""
        # Remove special characters and convert to lowercase
        text = re.sub(r'[^\w\s]', '', text.lower())
        
        # Split into words and remove common words
        words = text.split()
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        keywords = [word for word in words if word not in stop_words and len(word) > 3]
        
        return list(set(keywords))
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of text"""
        blob = TextBlob(text)
        
        # Get polarity (-1 to 1) and subjectivity (0 to 1)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Categorize sentiment
        if polarity > 0.1:
            sentiment = 'positive'
        elif polarity < -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'polarity': polarity,
            'subjectivity': subjectivity,
            'sentiment': sentiment
        }
    
    def fetch_news_from_api(self, query: str) -> List[Dict[str, Any]]:
        """Fetches news articles using a hypothetical news API."""
        if not self.api_key:
            logger.warning("News API key not provided. Skipping API fetch and using fallback (if any).")
            return []

        articles = []
        try:
            params = {
                'q': query,
                'apiKey': self.api_key,
                'language': 'en',
                'pageSize': self.max_articles,
                'sortBy': 'relevancy'
            }
            response = requests.get(self.NEWS_API_BASE_URL, params=params, timeout=10)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            data = response.json()
            
            for article in data.get('articles', []):
                title = article.get('title', '')
                description = article.get('description', '')
                content = article.get('content', '')
                source_name = article.get('source', {}).get('name', 'Unknown')
                published_at = article.get('publishedAt', '')
                url = article.get('url', '')

                full_text = f"{title} {description} {content}"
                keywords = self.extract_keywords(full_text)
                sentiment = self.analyze_sentiment(full_text)

                self.keywords.update(keywords)
                self.sentiment_scores[sentiment['sentiment']].append(sentiment['polarity'])
                
                articles.append({
                    'title': title,
                    'link': url,
                    'date': published_at,
                    'content': full_text,
                    'keywords': keywords,
                    'sentiment': sentiment,
                    'source': source_name
                })
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching news from API: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred during API news fetch: {e}")
        return articles

    def search_news(self) -> List[Dict[str, Any]]:
        """Search for relevant news articles using API or fallback scraping."""
        articles = self.fetch_news_from_api(self.query)
        if not articles:
            logger.warning("No articles from API. Falling back to Google News scraping.")
            self.fetch_google_news() # Fallback to existing scraping if API fails or not configured
            articles = self.articles # Get articles from scraping

        self.articles = articles
        return articles
    
    def analyze_news_impact(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the impact of news on the market"""
        if not articles:
            return {
                'overall_sentiment': 'neutral',
                'sentiment_distribution': {'positive': 0, 'neutral': 0, 'negative': 0},
                'key_topics': [],
                'impact_score': 0.0
            }
        
        # Calculate sentiment distribution
        sentiment_counts = defaultdict(int)
        for article in articles:
            sentiment_counts[article['sentiment']['sentiment']] += 1
        
        # Calculate overall sentiment
        total_articles = len(articles)
        sentiment_distribution = {
            sentiment: count/total_articles
            for sentiment, count in sentiment_counts.items()
        }
        
        # Determine overall sentiment
        if sentiment_distribution.get('positive', 0) > 0.4:
            overall_sentiment = 'positive'
        elif sentiment_distribution.get('negative', 0) > 0.4:
            overall_sentiment = 'negative'
        else:
            overall_sentiment = 'neutral'
        
        # Calculate impact score (-1 to 1)
        impact_score = (
            sentiment_distribution.get('positive', 0) -
            sentiment_distribution.get('negative', 0)
        )
        
        # Get most common keywords
        keyword_counts = defaultdict(int)
        for article in articles:
            for keyword in article['keywords']:
                keyword_counts[keyword] += 1
        
        key_topics = sorted(
            keyword_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return {
            'overall_sentiment': overall_sentiment,
            'sentiment_distribution': sentiment_distribution,
            'key_topics': key_topics,
            'impact_score': impact_score
        }
    
    def get_news_summary(self, articles: List[Dict[str, Any]]) -> str:
        """Generate a summary of the news analysis"""
        if not articles:
            return "No relevant news articles found."
        
        impact = self.analyze_news_impact(articles)
        
        summary = f"News Analysis Summary:\n"
        summary += f"Overall Sentiment: {impact['overall_sentiment'].upper()}\n"
        summary += f"Impact Score: {impact['impact_score']:.2f}\n\n"
        
        summary += "Key Topics:\n"
        for topic, count in impact['key_topics']:
            summary += f"- {topic}: {count} mentions\n"
        
        summary += "\nSentiment Distribution:\n"
        for sentiment, ratio in impact['sentiment_distribution'].items():
            summary += f"- {sentiment}: {ratio:.1%}\n"
        
        return summary

    def fetch_google_news(self):
        # Simple Google News scraping (for demo; for production use NewsAPI, GNews, etc.)
        search_url = f"https://news.google.com/search?q={self.query.replace(' ', '+')}&hl=en-US&gl=US&ceid=US:en"
        response = requests.get(search_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        articles = []
        for item in soup.find_all('article')[:self.max_articles]:
            title_tag = item.find('h3')
            if not title_tag:
                continue
            title = title_tag.text.strip()
            link_tag = item.find('a')
            link = 'https://news.google.com' + link_tag['href'][1:] if link_tag else ''
            articles.append({'title': title, 'link': link})
        self.articles = articles

    def llm_summarize_news(self, forecast_context: Optional[str] = None):
        # Combine all article titles for LLM summarization
        news_text = '\n'.join([f"- {a['title']}" for a in self.articles])
        prompt = f"""
You are a market analyst AI. Here are recent news headlines related to the following query: '{self.query}'.\n
{news_text}

Given the following forecast context, summarize the news and explain how it may relate to the forecasted trends.\nForecast context: {forecast_context}
"""
        self.llm_insights = call_llm(prompt, max_tokens=512, llm_config=self.llm_config)
        self.summary = self.llm_insights

    def run(self, forecast_context: Optional[str] = None) -> Dict[str, Any]:
        self.search_news() # Use the updated search_news which handles API and fallback
        self.llm_summarize_news(forecast_context)
        return {
            'articles': self.articles,
            'llm_summary': self.summary
        }

async def news_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Main news agent node function"""
    logger.info("Executing News Agent Node")
    try:
        user_query = state.get('query', "general market trends") 
        forecast_summary = state.get('forecast_summary', None)
        # Assuming API key might be in config, passed via state
        api_key = state.get('config', {}).get('news', {}).get('api_key', None)
        llm_config = state.get('config', {}).get('llm', {})

        agent = NewsAgent(user_query, api_key=api_key, llm_config=llm_config) # Pass API key and LLM config to agent
        news_output = agent.run(forecast_context=forecast_summary)

        state['news_results'] = news_output['articles']
        state['news_summary'] = news_output['llm_summary']
        
        return state
    except Exception as e:
        logger.error(f"Error in news agent node: {e}")
        state['error'] = str(e)
        return state
