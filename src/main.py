import logging
from pathlib import Path
import yaml
import pandas as pd
from datetime import datetime
from typing import Dict, Optional

from agents.forecasting_agent import ForecastingAgent
from agents.news_agent import NewsAgent
from agents.simulation_agent import SimulationAgent
from agents.report_agent import ReportAgent
from data_processing.data_handler import DataHandler

class Orchestrator:
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Orchestrator.
        
        Args:
            config_path (str, optional): Path to configuration file
        """
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
        self.config = self.load_config(config_path)
        
        # Initialize agents
        llm_config = self.config.get('llm', {})
        self.forecasting_agent = ForecastingAgent(config_path=config_path)
        self.news_agent = NewsAgent(config_path=config_path, llm_config=llm_config)
        self.simulation_agent = SimulationAgent(llm_config=llm_config)
        self.report_agent = ReportAgent(llm_config=llm_config)
        self.data_handler = DataHandler(self.config, llm_config=llm_config)
    
    def setup_logging(self):
        """Configure logging."""
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f'orchestrator_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
    
    def load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from YAML file."""
        default_config = {
            'data': {
                'purchase_data': 'data/us_spirits_purchase_data.csv',
                'products': 'data/products.csv',
                'clusters': 'data/clusters.csv'
            },
            'forecasting': {
                'target_column': 'Sales',
                'forecast_horizon': 12,
                'confidence_level': 0.95
            },
            'simulation': {
                'feature_to_modify': 'Price',
                'changes': [-0.1, -0.05, 0, 0.05, 0.1]
            },
            'news': {
                'search_terms': ['spirits market', 'alcohol industry', 'beverage trends'],
                'time_period': '1y'
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
                # Merge default config with loaded config, prioritizing loaded values
                return self.merge_configs(default_config, loaded_config)
        return default_config

    def merge_configs(self, default: Dict, overrides: Dict) -> Dict:
        """Recursively merges override dictionary into default dictionary."""
        for key, value in overrides.items():
            if isinstance(value, dict) and key in default and isinstance(default[key], dict):
                default[key] = self.merge_configs(default[key], value)
            else:
                default[key] = value
        return default
    
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load all required datasets using DataHandler."""
        try:
            data = {
                'purchase_data': self.data_handler.load_and_preprocess_purchase_data(),
                'products': self.data_handler.load_products_data(),
                'clusters': self.data_handler.load_clusters_data()
            }
            return data
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
    
    def run_analysis(self, user_query: str = "general market trends", user_inputs: Optional[str] = None, business_rules: Optional[str] = None):
        """Run the complete analysis pipeline."""
        try:
            # Load data
            self.logger.info("Loading data...")
            data = self.load_data()
            # Check if data is loaded and purchase_data is not empty
            if not data or 'purchase_data' not in data or data['purchase_data'].empty:
                self.logger.error("Failed to load or preprocess purchase data. Aborting analysis.")
                raise ValueError("No purchase data available for analysis.")
            
            # Generate forecasts
            self.logger.info("Generating forecasts...")
            try:
                target_column = self.config['forecasting']['target_column']
                forecast_horizon = self.config['forecasting']['forecast_horizon']
                # Use provided user_query or fallback to default
                forecast_user_query = user_query # Use dynamic user_query

                forecast_results = self.forecasting_agent.forecast(
                    data['purchase_data'],
                    target_column,
                    forecast_horizon,
                    user_query=forecast_user_query
                )
                if not forecast_results.get('forecast_result'):
                    self.logger.warning("Forecasting agent returned no results. Proceeding with caution.")
            except Exception as e:
                self.logger.error(f"Error in forecasting agent: {str(e)}")
                forecast_results = {} # Ensure forecast_results is always a dict

            
            # Collect market insights
            self.logger.info("Collecting market insights...")
            market_insights = {}
            try:
                news_search_query = self.config['news'].get('search_terms', ["general market trends"])[0]
                news_api_key = self.config['news'].get('api_key', None)

                market_insights = self.news_agent.run(
                    forecast_context=forecast_results.get('llm_reasoning', '')
                )
                if not market_insights.get('articles') and not market_insights.get('llm_summary'):
                    self.logger.warning("News agent returned no insights. Proceeding with caution.")
            except Exception as e:
                self.logger.error(f"Error in news agent: {str(e)}")
                market_insights = {} # Ensure market_insights is always a dict

            
            # Run simulations
            self.logger.info("Running simulations...")
            simulation_results = {}
            try:
                # Use provided user_inputs for simulation
                simulation_user_inputs = user_inputs # Use dynamic user_inputs

                if forecast_results and market_insights.get('llm_summary'):
                    simulation_results = self.simulation_agent.run(
                        forecast_result=forecast_results,
                        news_summary=market_insights.get('llm_summary', ''),
                        purchase_data=data['purchase_data'],
                        products_data=data['products'],
                        clusters_data=data['clusters'],
                        user_inputs=simulation_user_inputs
                    )
                    if not simulation_results.get('simulation_results'):
                        self.logger.warning("Simulation agent returned no results. Proceeding with caution.")
                else:
                    self.logger.warning("Skipping simulation due to missing forecast or news summary.")
            except Exception as e:
                self.logger.error(f"Error in simulation agent: {str(e)}")
                simulation_results = {} # Ensure simulation_results is always a dict
            
            # Generate report
            self.logger.info("Generating report...")
            report_results = {}
            try:
                # Use provided user_query and business_rules for report
                report_user_query = user_query # Use dynamic user_query
                report_business_rules = business_rules # Use dynamic business_rules

                report_results = self.report_agent.generate_report(
                    forecast_result=forecast_results,
                    news_results=market_insights,
                    simulation_result=simulation_results,
                    user_query=report_user_query,
                    business_rules=report_business_rules
                )
                if not report_results.get('generated_reports'):
                    self.logger.warning("Report agent generated no reports.")
            except Exception as e:
                self.logger.error(f"Error in report agent: {str(e)}")
                report_results = {} # Ensure report_results is always a dict

            
            self.logger.info("Analysis completed successfully!")
            return report_results
            
        except Exception as e:
            self.logger.error(f"Error in analysis pipeline: {str(e)}")
            raise

def main():
    """Main function to run the analysis pipeline."""
    try:
        # Initialize orchestrator
        orchestrator = Orchestrator('config/config.yaml')
        
        # Load data
        logging.info("Loading data...")
        purchase_data = orchestrator.data_handler.load_purchase_data()
        products_data = orchestrator.data_handler.load_products_data()
        clusters_data = orchestrator.data_handler.load_clusters_data()
        
        # Generate forecasts
        logging.info("Generating forecasts...")
        forecast_results = orchestrator.forecasting_agent.forecast(
            data=purchase_data,
            target_column='sales_value'
        )
        
        # Collect market insights
        logging.info("Collecting market insights...")
        news_results = orchestrator.news_agent.run(forecast_context=str(forecast_results))
        
        if not news_results.get('articles'):
            logging.warning("News agent returned no insights. Proceeding with caution.")
        
        # Run simulations
        logging.info("Running simulations...")
        if forecast_results and news_results.get('articles'):
            simulation_results = orchestrator.simulation_agent.simulate_scenarios(
                data=purchase_data,
                target_column='sales_value',
                feature_to_modify='sales_volume',
                changes=[-0.1, 0, 0.1, 0.2]
            )
        else:
            logging.warning("Skipping simulation due to missing forecast or news summary.")
            simulation_results = {}
        
        # Generate report
        logging.info("Generating report...")
        results = orchestrator.report_agent.generate_report(
            forecast_results=forecast_results,
            market_insights=news_results,
            simulation_results=simulation_results
        )
        
        logging.info("Analysis completed successfully!")
        
        # Print report locations
        print("\nGenerated Reports:")
        if 'generated_reports' in results:
            for report_type, path in results['generated_reports'].items():
                print(f"{report_type}: {path}")
        else:
            print("No reports were generated.")
        
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 