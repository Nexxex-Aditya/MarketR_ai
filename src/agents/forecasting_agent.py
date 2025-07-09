from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
import yaml
import json
import os
from dotenv import load_dotenv
from nixtla import NixtlaClient
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from utils.llm_utils import call_llm


class ForecastingAgent:
    def __init__(self, config: Optional[Dict] = None, config_path: Optional[str] = None):
        """
        Initialize the Forecasting Agent with configuration and logging.
        
        Args:
            config (dict, optional): Configuration dictionary
            config_path (str, optional): Path to configuration file
        """
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
        
        # Load configuration
        if config:
            self.config = config
        else:
            self.config = self.load_config(config_path)
        
        # Load environment variables from .env file
        load_dotenv()
        
        # Get Nixtla API key from environment variables
        api_key = os.getenv("NIXTLA_API_KEY")
        if not api_key:
            raise ValueError("Nixtla API key not found in .env file. Please set NIXTLA_API_KEY in your .env file")
        
        self.client = NixtlaClient()
        
        # Initialize ML components
        self.scaler = StandardScaler()
        self.model = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )

    def setup_logging(self):
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

    def load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from a YAML file or use default settings."""
        default_config = {
            'forecast': {
                'horizon': 30,
                'confidence_level': 0.95,
                'frequency': 'D'
            },
            'anomaly_detection': {
                'threshold': 2.0,
                'window_size': 7
            }
        }
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
                # Merge default config with loaded config
                if loaded_config and 'forecast' in loaded_config:
                    default_config['forecast'].update(loaded_config['forecast'])
                if loaded_config and 'anomaly_detection' in loaded_config:
                    default_config['anomaly_detection'].update(loaded_config['anomaly_detection'])
        return default_config

    def prepare_data(self, data: pd.DataFrame, target_column: str) -> Tuple[np.ndarray, pd.Series]:
        """
        Prepare data for simulation.
        
        Args:
            data (pd.DataFrame): Input data
            target_column (str): Target column to simulate
            
        Returns:
            tuple: (X_scaled, y) prepared data
        """
        # Make a copy to avoid modifying original data
        data = data.copy()
        
        # Ensure date column is datetime
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
            data.set_index('date', inplace=True)
        
        # Handle NaN values in target column
        if data[target_column].isnull().any():
            # Replace NaN with median
            median_value = data[target_column].median()
            data[target_column] = data[target_column].fillna(median_value)
            self.logger.info(f"Replaced NaN values in {target_column} with median: {median_value}")
        
        # Select features
        feature_cols = [col for col in data.columns if col != target_column]
        X = data[feature_cols]
        y = data[target_column]
        
        # Handle NaN values in features
        if X.isnull().any().any():
            # Replace NaN with median for each feature
            for col in X.columns:
                median_value = X[col].median()
                X[col] = X[col].fillna(median_value)
                self.logger.info(f"Replaced NaN values in feature {col} with median: {median_value}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y

    def detect_anomalies(self, df: pd.DataFrame) -> Dict:
        """
        Detect anomalies in the target column using rolling z-score.
        """
        anomalies = {}
        rolling_mean = df['y'].rolling(window=self.config['anomaly_detection']['window_size']).mean()
        rolling_std = df['y'].rolling(window=self.config['anomaly_detection']['window_size']).std()
        z_scores = (df['y'] - rolling_mean) / rolling_std
        anomaly_indices = df.index[abs(z_scores) > self.config['anomaly_detection']['threshold']]
        if not anomaly_indices.empty:
            anomalies['y'] = {
                'dates': df.loc[anomaly_indices, 'ds'].astype(str).tolist(),
                'values': df.loc[anomaly_indices, 'y'].tolist(),
                'z_scores': z_scores[anomaly_indices].tolist()
            }
        return anomalies

    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            'forecast': {
                'horizon': 30,
                'confidence_level': 0.95,
                'frequency': 'D'
            },
            'anomaly_detection': {
                'threshold': 2.0,
                'window_size': 7
            }
        }

    def forecast(self, data: pd.DataFrame, target_column: str, user_query: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate forecasts using machine learning.
        
        Args:
            data (pd.DataFrame): Input data
            target_column (str): Target column to forecast
            user_query (str, optional): User's specific query or requirements
            
        Returns:
            dict: Forecast results
        """
        try:
            # Prepare data
            X, y = self.prepare_data(data, target_column)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = self.model.predict(X_test)
            
            # Calculate metrics
            mse = np.mean((y_test - y_pred) ** 2)
            rmse = np.sqrt(mse)
            
            # Generate future predictions
            last_date = pd.to_datetime(data.index[-1])
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=self.config['forecast']['horizon'],
                freq='D'
            )
            
            # Prepare future features (using last known values)
            future_X = np.tile(X[-1], (len(future_dates), 1))
            future_X_scaled = self.scaler.transform(future_X)
            
            # Make future predictions
            future_pred = self.model.predict(future_X_scaled)
            
            # Calculate prediction intervals
            std_pred = np.std(y_test - y_pred)
            lower_bound = future_pred - 1.96 * std_pred
            upper_bound = future_pred + 1.96 * std_pred
            
            # Save results
            results = {
                'forecast': {
                    'dates': future_dates.tolist(),
                    'predictions': future_pred.tolist(),
                    'lower_bound': lower_bound.tolist(),
                    'upper_bound': upper_bound.tolist()
                },
                'metrics': {
                    'rmse': float(rmse),
                    'mse': float(mse)
                },
                'model_info': {
                    'type': 'RandomForestRegressor',
                    'n_estimators': self.model.n_estimators,
                    'feature_importance': dict(zip(
                        [col for col in data.columns if col != target_column],
                        self.model.feature_importances_
                    ))
                },
                'user_query': user_query
            }
            self.save_results(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in forecasting: {str(e)}")
            raise

    def save_results(self, results: Dict):
        """
        Save the forecast and anomaly detection results to a JSON file.
        """
        output_dir = Path('output/forecasts')
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f'forecast_results_{timestamp}.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4, default=str)
        self.logger.info(f"Forecast results saved to {output_file}")

    def analyze_trends(self, data: pd.DataFrame, target_column: str) -> Dict:
        """
        Analyze the overall trend, volatility, and seasonality of the data.
        """
        data = data.copy()
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.sort_values('Date')
        values = data[target_column]

        stats = values.describe()
        trend = np.polyfit(range(len(values)), values, 1)[0]
        seasonal = values.diff(12).mean() if len(values) >= 12 else None

        return {
            'statistics': stats.to_dict(),
            'trend': float(trend),
            'seasonality': float(seasonal) if seasonal is not None else None,
            'trend_direction': 'increasing' if trend > 0 else 'decreasing',
            'volatility': float(values.std())
        } 