import logging
from pathlib import Path
import yaml
import pandas as pd
from datetime import datetime
from typing import Dict, Optional
import json

from news_agent import NewsAgent

def setup_logging():
    """Configure logging."""
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f'news_collection_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )

def load_config(config_path: Optional[str] = None) -> Dict:
    """Load configuration from YAML file."""
    default_config = {
        'news': {
            'search_terms': [
                'spirits market',
                'alcohol industry',
                'beverage trends',
                'whiskey market',
                'vodka market',
                'rum market',
                'tequila market'
            ],
            'time_period': '1y',
            'sources': {
                'reddit': {
                    'subreddits': [
                        'r/alcohol',
                        'r/whiskey',
                        'r/cocktails',
                        'r/bartenders'
                    ],
                    'limit': 100
                },
                'google_trends': {
                    'timeframe': 'today 12-m',
                    'geo': 'US'
                },
                'trading_economics': {
                    'indicators': [
                        'US Consumer Price Index',
                        'US Retail Sales',
                        'US Personal Income'
                    ]
                }
            }
        },
        'output': {
            'news_dir': 'output/news',
            'reports_dir': 'output/reports',
            'logs_dir': 'logs'
        }
    }
    
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            return {**default_config, **yaml.safe_load(f)}
    return default_config

def create_output_dirs(config: Dict):
    """Create output directories."""
    for dir_path in config['output'].values():
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def save_news_results(results: Dict, output_dir: Path):
    """Save news collection results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save news data
    news_df = pd.DataFrame(results['news_data'])
    news_df.to_csv(output_dir / f'news_data_{timestamp}.csv', index=False)
    
    # Save market data
    market_df = pd.DataFrame(results['market_data'])
    market_df.to_csv(output_dir / f'market_data_{timestamp}.csv', index=False)
    
    # Save insights
    insights_file = output_dir / f'news_insights_{timestamp}.json'
    with open(insights_file, 'w') as f:
        json.dump(results['insights'], f, indent=4, default=str)
    
    logging.info(f"News collection results saved to {output_dir}")

def main():
    """Main entry point for news collection pipeline."""
    try:
        # Setup logging
        setup_logging()
        logger = logging.getLogger(__name__)
        
        # Load configuration
        config = load_config('config/config.yaml')
        logger.info("Configuration loaded successfully")
        
        # Create output directories
        create_output_dirs(config)
        logger.info("Output directories created")
        
        # Initialize news agent
        agent = NewsAgent(config)
        logger.info("News agent initialized")
        
        # Collect news
        logger.info("Collecting news and market data...")
        news_results = agent.collect_news(
            config['news']['search_terms'],
            config['news']['time_period']
        )
        
        # Save results
        logger.info("Saving news collection results...")
        save_news_results(
            news_results,
            Path(config['output']['news_dir'])
        )
        
        # Generate and save analysis
        logger.info("Generating news analysis...")
        analysis_results = agent.analyze_news(news_results)
        
        analysis_file = Path(config['output']['reports_dir']) / f'news_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(analysis_file, 'w') as f:
            json.dump(analysis_results, f, indent=4, default=str)
        
        logger.info("News collection completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in news collection pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main() 