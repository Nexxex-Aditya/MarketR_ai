import logging
from pathlib import Path
import yaml
import pandas as pd
from datetime import datetime
from typing import Dict, Optional
import json

from forecasting_agent import ForecastingAgent

def setup_logging():
    """Configure logging."""
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f'forecasting_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )

def load_config(config_path: Optional[str] = None) -> Dict:
    """Load configuration from YAML file."""
    default_config = {
        'data': {
            'processed_data': 'processed_data/integrated_data.parquet'
        },
        'forecasting': {
            'target_column': 'Sales',
            'forecast_horizon': 12,
            'confidence_level': 0.95,
            'model': {
                'type': 'timegpt',
                'params': {
                    'freq': 'M',
                    'seasonality': 12,
                    'prediction_interval': 0.95
                }
            }
        },
        'output': {
            'forecasts_dir': 'output/forecasts',
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

def save_forecast_results(results: Dict, output_dir: Path):
    """Save forecast results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save forecasts
    forecasts_df = pd.DataFrame({
        'date': results['dates'],
        'forecast': results['predictions'],
        'lower_bound': results['confidence_intervals']['lower'],
        'upper_bound': results['confidence_intervals']['upper']
    })
    forecasts_df.to_csv(output_dir / f'forecasts_{timestamp}.csv', index=False)
    
    # Save metrics
    metrics_file = output_dir / f'forecast_metrics_{timestamp}.json'
    with open(metrics_file, 'w') as f:
        json.dump(results['metrics'], f, indent=4, default=str)
    
    logging.info(f"Forecast results saved to {output_dir}")

def main():
    """Main entry point for forecasting pipeline."""
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
        
        # Load processed data
        logger.info("Loading processed data...")
        data = pd.read_parquet(config['data']['processed_data'])
        
        # Initialize forecasting agent
        agent = ForecastingAgent(config)
        logger.info("Forecasting agent initialized")
        
        # Generate forecasts
        logger.info("Generating forecasts...")
        forecast_results = agent.forecast(
            data,
            config['forecasting']['target_column'],
            config['forecasting']['forecast_horizon']
        )
        
        # Save results
        logger.info("Saving forecast results...")
        save_forecast_results(
            forecast_results,
            Path(config['output']['forecasts_dir'])
        )
        
        # Generate and save analysis
        logger.info("Generating forecast analysis...")
        analysis_results = agent.analyze_trends(forecast_results)
        
        analysis_file = Path(config['output']['reports_dir']) / f'forecast_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(analysis_file, 'w') as f:
            json.dump(analysis_results, f, indent=4, default=str)
        
        logger.info("Forecasting completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in forecasting pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main() 