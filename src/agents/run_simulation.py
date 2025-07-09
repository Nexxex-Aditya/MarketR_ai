import logging
from pathlib import Path
import yaml
import pandas as pd
from datetime import datetime
from typing import Dict, Optional
import json

from simulation_agent import SimulationAgent

def setup_logging():
    """Configure logging."""
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f'simulation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )

def load_config(config_path: Optional[str] = None) -> Dict:
    """Load configuration from YAML file."""
    default_config = {
        'data': {
            'processed_data': 'processed_data/integrated_data.parquet'
        },
        'simulation': {
            'feature_to_modify': 'Price',
            'changes': [-0.1, -0.05, 0, 0.05, 0.1],
            'model': {
                'n_estimators': 100,
                'max_depth': 10
            },
            'confidence_level': 0.95
        },
        'output': {
            'simulations_dir': 'output/simulations',
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

def save_simulation_results(results: Dict, output_dir: Path):
    """Save simulation results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save scenario results
    scenarios_df = pd.DataFrame(results['scenarios'])
    scenarios_df.to_csv(output_dir / f'scenario_results_{timestamp}.csv', index=False)
    
    # Save feature importance
    importance_df = pd.DataFrame(results['feature_importance'])
    importance_df.to_csv(output_dir / f'feature_importance_{timestamp}.csv', index=False)
    
    # Save correlations
    correlations_file = output_dir / f'correlations_{timestamp}.json'
    with open(correlations_file, 'w') as f:
        json.dump(results['correlations'], f, indent=4, default=str)
    
    logging.info(f"Simulation results saved to {output_dir}")

def main():
    """Main entry point for simulation pipeline."""
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
        
        # Initialize simulation agent
        agent = SimulationAgent(config)
        logger.info("Simulation agent initialized")
        
        # Run simulations
        logger.info("Running simulations...")
        simulation_results = agent.simulate_scenarios(
            data,
            config['simulation']['feature_to_modify'],
            config['simulation']['changes']
        )
        
        # Save results
        logger.info("Saving simulation results...")
        save_simulation_results(
            simulation_results,
            Path(config['output']['simulations_dir'])
        )
        
        # Generate and save analysis
        logger.info("Generating simulation analysis...")
        analysis_results = agent.analyze_correlations(data)
        
        analysis_file = Path(config['output']['reports_dir']) / f'simulation_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(analysis_file, 'w') as f:
            json.dump(analysis_results, f, indent=4, default=str)
        
        logger.info("Simulation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in simulation pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main() 