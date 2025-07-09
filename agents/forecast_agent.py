import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
from utils.llm_utils import call_llm
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ForecastAgent:
    def __init__(self, config_path: Optional[str] = None, llm_config: Optional[Dict[str, Any]] = None):
        self.model_choice = None
        self.llm_reasoning = None
        self.forecast_result = None
        self.llm_config = llm_config

    def llm_select_model(self, data: pd.DataFrame, user_query: str):
        prompt = f"""
You are a forecasting expert. Given the following data columns and a sample, recommend the best forecasting method (Prophet, SARIMAX, RandomForest, etc.) and explain your reasoning.
Data columns: {list(data.columns)}
Sample data:
{data.head().to_string()}
Business context: {user_query}
"""
        response = call_llm(prompt, llm_config=self.llm_config)
        self.llm_reasoning = response
        # Simple extraction: look for model name in response
        if 'prophet' in response.lower():
            self.model_choice = 'prophet'
        elif 'sarimax' in response.lower():
            self.model_choice = 'sarimax'
        elif 'randomforest' in response.lower() or 'random forest' in response.lower():
            self.model_choice = 'randomforest'
        else:
            self.model_choice = 'prophet'  # default

    def run_forecast(self, data: pd.DataFrame, target_column: str, horizon: int = 12):
        # Assume time series column is 'date' and target is dynamic via target_column
        df = data.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in data.")

        if self.model_choice == 'prophet':
            prophet_df = df.rename(columns={'date': 'ds', target_column: 'y'})
            model = Prophet()
            model.fit(prophet_df)
            future = model.make_future_dataframe(periods=horizon, freq='M')
            forecast = model.predict(future)
            self.forecast_result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(horizon)
        elif self.model_choice == 'sarimax':
            y = df[target_column].values
            model = SARIMAX(y, order=(1,1,1), seasonal_order=(1,1,1,12))
            results = model.fit(disp=False)
            forecast = results.get_forecast(steps=horizon)
            pred = forecast.predicted_mean
            ci = forecast.conf_int()
            self.forecast_result = pd.DataFrame({
                'date': pd.date_range(df['date'].iloc[-1], periods=horizon+1, freq='M')[1:],
                'yhat': pred,
                'yhat_lower': ci.iloc[:,0],
                'yhat_upper': ci.iloc[:,1]
            })
        elif self.model_choice == 'randomforest':
            # Simple lag-based RF
            df['lag1'] = df[target_column].shift(1)
            df = df.dropna()
            X = df[['lag1']]
            y = df[target_column]
            model = RandomForestRegressor()
            model.fit(X, y)
            last_val = df[target_column].iloc[-1]
            preds = []
            for _ in range(horizon):
                pred = model.predict(np.array([[last_val]]))[0]
                preds.append(pred)
                last_val = pred
            self.forecast_result = pd.DataFrame({
                'date': pd.date_range(df['date'].iloc[-1], periods=horizon+1, freq='M')[1:],
                'yhat': preds
            })
        else:
            raise ValueError('Unknown model choice')

    def detect_unusual_trends(self, forecast_df: pd.DataFrame, target_column: str = 'yhat') -> Dict[str, Any]:
        """Detects unusual trends (significant increases or decreases) in the forecast."""
        if forecast_df.empty or target_column not in forecast_df.columns:
            logger.warning("Forecast data is empty or target column missing for trend detection.")
            return {'unusual_trend_detected': False, 'trend_description': 'N/A'}

        # Ensure forecast_df is sorted by date
        forecast_df = forecast_df.sort_values(by='date')

        # Focus on the most recent trends, e.g., last 3 data points
        if len(forecast_df) < 3:
            return {'unusual_trend_detected': False, 'trend_description': 'Not enough data for trend analysis.'}

        recent_forecast = forecast_df[target_column].tail(3)
        last_value = recent_forecast.iloc[-1]
        previous_value = recent_forecast.iloc[-2]
        two_ago_value = recent_forecast.iloc[-3]

        # Simple trend detection: check for consistent increase or decrease
        trend_detected = False
        trend_description = "No unusual trend detected."

        # Threshold for 'unusual' change, e.g., 5% increase/decrease over previous two periods
        change_threshold = 0.05 

        # Check for unusual incremental trend
        if (last_value > previous_value * (1 + change_threshold)) and \
           (previous_value > two_ago_value * (1 + change_threshold)):
            trend_detected = True
            trend_description = f"Unusual incremental trend detected in {target_column}. Recent values increased by more than {change_threshold*100}% consecutively."
        
        # Check for unusual decremental trend
        elif (last_value < previous_value * (1 - change_threshold)) and \
             (previous_value < two_ago_value * (1 - change_threshold)):
            trend_detected = True
            trend_description = f"Unusual decremental trend detected in {target_column}. Recent values decreased by more than {change_threshold*100}% consecutively."
            
        return {
            'unusual_trend_detected': trend_detected,
            'trend_description': trend_description,
            'last_forecast_values': recent_forecast.tolist()
        }

    def forecast(self, data: pd.DataFrame, target_column: str, forecast_horizon: int, user_query: str) -> Dict:
        self.llm_select_model(data, user_query)
        self.run_forecast(data, target_column, forecast_horizon)
        
        # Perform trend detection on the generated forecast
        trend_analysis = self.detect_unusual_trends(pd.DataFrame(self.forecast_result), target_column='yhat')

        return {
            'model_choice': self.model_choice,
            'llm_reasoning': self.llm_reasoning,
            'forecast_result': self.forecast_result.to_dict(orient='records'),
            'trend_analysis': trend_analysis # Add trend analysis to the output
        }

async def forecast_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Forecast agent node for LangGraph workflow."""
    logger.info("Executing Forecast Agent Node")
    try:
        # Expect preprocessed data from the state
        purchase_data = state['data']['purchase_data']
        user_query = state['query']
        config = state['config'] # Access config from state
        target_column = config['forecasting']['target_column']
        forecast_horizon = config['forecasting']['forecast_horizon']
        llm_config = config.get('llm', {}) # Get LLM config from state

        agent = ForecastAgent(llm_config=llm_config)
        forecast_output = agent.forecast(
            purchase_data, 
            target_column, 
            forecast_horizon, 
            user_query
        )

        state['forecast_result'] = forecast_output['forecast_result']
        state['forecast_summary'] = forecast_output['llm_reasoning'] 
        state['model_choice'] = forecast_output['model_choice']
        state['forecast_trend_analysis'] = forecast_output['trend_analysis'] # Add trend analysis to state

        return state
    except Exception as e:
        logger.error(f"Error in forecast agent node: {e}")
        state['error'] = str(e)
        return state 