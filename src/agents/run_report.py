import logging
from pathlib import Path
import yaml
import pandas as pd
from datetime import datetime
from typing import Dict, Optional
import json

from report_agent import ReportAgent

def setup_logging():
    """Configure logging."""
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f'report_generation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )

def load_config(config_path: Optional[str] = None) -> Dict:
    """Load configuration from YAML file."""
    default_config = {
        'data': {
            'forecast_results': 'output/forecasts/forecasts.csv',
            'news_results': 'output/news/news_data.csv',
            'simulation_results': 'output/simulations/scenario_results.csv'
        },
        'report': {
            'template_dir': 'templates',
            'output_dir': 'reports',
            'sections': [
                'executive_summary',
                'data_overview',
                'forecast_analysis',
                'market_insights',
                'simulation_results',
                'recommendations'
            ],
            'visualization': {
                'style': 'seaborn',
                'context': 'notebook',
                'palette': 'deep',
                'figsize': [12, 6]
            }
        },
        'output': {
            'reports_dir': 'output/reports',
            'presentations_dir': 'output/presentations',
            'notebooks_dir': 'output/notebooks',
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

def load_results(config: Dict) -> Dict:
    """Load all results for report generation."""
    results = {}
    
    # Load forecast results
    forecast_df = pd.read_csv(config['data']['forecast_results'])
    results['forecast_results'] = {
        'dates': forecast_df['date'].tolist(),
        'predictions': forecast_df['forecast'].tolist(),
        'confidence_intervals': {
            'lower': forecast_df['lower_bound'].tolist(),
            'upper': forecast_df['upper_bound'].tolist()
        }
    }
    
    # Load news results
    news_df = pd.read_csv(config['data']['news_results'])
    results['news_results'] = {
        'news_data': news_df.to_dict('records'),
        'market_data': pd.read_csv(config['data']['news_results'].replace('news_data', 'market_data')).to_dict('records')
    }
    
    # Load simulation results
    sim_df = pd.read_csv(config['data']['simulation_results'])
    results['simulation_results'] = {
        'scenarios': sim_df.to_dict('records'),
        'feature_importance': pd.read_csv(config['data']['simulation_results'].replace('scenario_results', 'feature_importance')).to_dict('records')
    }
    
    return results

def main():
    """Main entry point for report generation pipeline."""
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
        
        # Load results
        logger.info("Loading results...")
        results = load_results(config)
        
        # Initialize report agent
        agent = ReportAgent(config)
        logger.info("Report agent initialized")
        
        # Generate report
        logger.info("Generating report...")
        report_results = agent.generate_report(
            results['forecast_results'],
            results['news_results'],
            results['simulation_results']
        )
        
        # Print report locations
        logger.info("\nGenerated Reports:")
        for report_type, path in report_results['generated_reports'].items():
            logger.info(f"- {report_type}: {path}")
        
        logger.info("Report generation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in report generation pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main() 