from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pathlib import Path
import yaml
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

class SimulationAgent:
    def __init__(self, config_path: Optional[str] = None, llm_config: Optional[Dict] = None):
        """
        Initialize the Simulation Agent.
        
        Args:
            config_path (str, optional): Path to configuration file
            llm_config (dict, optional): Configuration for LLM
        """
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
        self.config = self.load_config(config_path)
        self.llm_config = llm_config or {}
        self.model = RandomForestRegressor(
            n_estimators=self.config['model']['n_estimators'],
            max_depth=self.config['model']['max_depth'],
            random_state=42
        )
        self.scaler = StandardScaler()
        
    def setup_logging(self):
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
    
    def load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from YAML file."""
        default_config = {
            'model': {
                'n_estimators': 100,
                'max_depth': 10
            },
            'simulation': {
                'scenarios': 5,
                'confidence_level': 0.95
            },
            'visualization': {
                'style': 'seaborn',
                'context': 'notebook',
                'palette': 'deep'
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return {**default_config, **yaml.safe_load(f)}
        return default_config
    
    def prepare_data(self, data: pd.DataFrame, target_column: str) -> tuple:
        """
        Prepare data for simulation.
        
        Args:
            data (pd.DataFrame): Input data
            target_column (str): Target column to simulate
            
        Returns:
            tuple: (X, y) prepared data
        """
        # Select features
        feature_cols = [col for col in data.columns if col != target_column]
        X = data[feature_cols]
        y = data[target_column]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y
    
    def train_model(self, X: np.ndarray, y: np.ndarray):
        """
        Train the simulation model.
        
        Args:
            X (np.ndarray): Features
            y (np.ndarray): Target
        """
        self.model.fit(X, y)
        
        # Calculate feature importance
        feature_importance = pd.DataFrame({
            'feature': [col for col in self.model.feature_names_in_],
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return feature_importance
    
    def generate_scenarios(self, X: np.ndarray, feature_index: int, 
                          changes: List[float]) -> List[np.ndarray]:
        """
        Generate scenarios by modifying a specific feature.
        
        Args:
            X (np.ndarray): Base features
            feature_index (int): Index of feature to modify
            changes (List[float]): List of changes to apply
            
        Returns:
            List[np.ndarray]: Modified feature sets
        """
        scenarios = []
        base_value = X[:, feature_index].mean()
        
        for change in changes:
            scenario = X.copy()
            scenario[:, feature_index] = base_value * (1 + change)
            scenarios.append(scenario)
        
        return scenarios
    
    def simulate_scenarios(self, data: pd.DataFrame, target_column: str,
                          feature_to_modify: str, changes: List[float]) -> Dict:
        """
        Simulate what-if scenarios.
        
        Args:
            data (pd.DataFrame): Historical data
            target_column (str): Target column to simulate
            feature_to_modify (str): Feature to modify
            changes (List[float]): List of changes to apply
            
        Returns:
            Dict: Simulation results
        """
        try:
            # Prepare data
            X, y = self.prepare_data(data, target_column)
            
            # Train model
            feature_importance = self.train_model(X, y)
            
            # Get feature index
            feature_index = list(self.model.feature_names_in_).index(feature_to_modify)
            
            # Generate scenarios
            scenarios = self.generate_scenarios(X, feature_index, changes)
            
            # Make predictions
            predictions = []
            for scenario in scenarios:
                pred = self.model.predict(scenario)
                predictions.append(pred)
            
            # Calculate statistics
            results = {
                'feature_importance': feature_importance.to_dict(),
                'scenarios': []
            }
            
            for i, (change, pred) in enumerate(zip(changes, predictions)):
                scenario_results = {
                    'scenario_id': i + 1,
                    'feature_change': change,
                    'predictions': {
                        'mean': float(np.mean(pred)),
                        'std': float(np.std(pred)),
                        'min': float(np.min(pred)),
                        'max': float(np.max(pred))
                    },
                    'confidence_interval': {
                        'lower': float(np.percentile(pred, (1 - self.config['simulation']['confidence_level']) * 100 / 2)),
                        'upper': float(np.percentile(pred, (1 + self.config['simulation']['confidence_level']) * 100 / 2))
                    }
                }
                results['scenarios'].append(scenario_results)
            
            # Generate visualizations
            self.generate_visualizations(data, target_column, feature_to_modify, 
                                       changes, predictions, results)
            
            # Save results
            self.save_results(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in simulation: {str(e)}")
            raise
    
    def generate_visualizations(self, data: pd.DataFrame, target_column: str,
                              feature_to_modify: str, changes: List[float],
                              predictions: List[np.ndarray], results: Dict):
        """Generate visualization of simulation results."""
        # Set style
        plt.style.use(self.config['visualization']['style'])
        sns.set_context(self.config['visualization']['context'])
        
        # Create output directory
        output_dir = Path('output/simulations')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(data=pd.DataFrame(results['feature_importance']),
                   x='importance', y='feature')
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig(output_dir / 'feature_importance.png')
        plt.close()
        
        # Plot scenario predictions
        plt.figure(figsize=(12, 6))
        for i, (change, pred) in enumerate(zip(changes, predictions)):
            plt.plot(pred, label=f'{change*100:.1f}% change')
        plt.title(f'Scenario Predictions for {target_column}')
        plt.xlabel('Time')
        plt.ylabel(target_column)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / 'scenario_predictions.png')
        plt.close()
        
        # Plot confidence intervals
        plt.figure(figsize=(12, 6))
        changes_percent = [f'{c*100:.1f}%' for c in changes]
        means = [s['predictions']['mean'] for s in results['scenarios']]
        lower = [s['confidence_interval']['lower'] for s in results['scenarios']]
        upper = [s['confidence_interval']['upper'] for s in results['scenarios']]
        
        plt.errorbar(changes_percent, means, yerr=[means[i] - lower[i] for i in range(len(means))],
                    fmt='o-', capsize=5)
        plt.title(f'Confidence Intervals for Different {feature_to_modify} Changes')
        plt.xlabel('Change in Feature')
        plt.ylabel(target_column)
        plt.tight_layout()
        plt.savefig(output_dir / 'confidence_intervals.png')
        plt.close()
    
    def save_results(self, results: Dict):
        """Save simulation results."""
        output_dir = Path('output/simulations')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f'simulation_results_{timestamp}.json'
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4, default=str)
        
        self.logger.info(f"Simulation results saved to {output_file}")
    
    def analyze_correlations(self, data: pd.DataFrame) -> Dict:
        """
        Analyze correlations between features.
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            Dict: Correlation analysis results
        """
        # Calculate correlations
        corr_matrix = data.corr()
        
        # Find strong correlations
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.7:  # Strong correlation threshold
                    strong_correlations.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': float(corr_matrix.iloc[i, j])
                    })
        
        return {
            'correlation_matrix': corr_matrix.to_dict(),
            'strong_correlations': strong_correlations
        } 