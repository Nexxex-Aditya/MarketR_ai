# Simulation Agent using Correlation - based logic

# This model outcomes based on price anomalies using simple business rules.
# each scenario inlcudes :
#     - Date
#     - Price
#     - Imapct Assessment

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from scipy import stats
import random
import logging
from datetime import datetime, timedelta
from utils.llm_utils import call_llm
from pathlib import Path
import json
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimulationAgent:
    """Enhanced market simulation agent"""
    
    def __init__(self, config_path: Optional[str] = None, llm_config: Optional[Dict[str, Any]] = None):
        self.llm_scenarios = None
        self.simulation_results = None
        self.llm_summary = None
        self.llm_config = llm_config # Store LLM configuration
    
    def generate_price_elasticity(self, purchase_data: pd.DataFrame) -> float:
        """Generates a hypothetical price elasticity based on historical data."""
        if 'PurchaseValue' in purchase_data.columns and 'PurchaseVolume' in purchase_data.columns and not purchase_data[['PurchaseValue', 'PurchaseVolume']].isnull().all().any():
            temp_df = purchase_data[['PurchaseValue', 'PurchaseVolume']].dropna()
            if not temp_df.empty and len(temp_df) > 1:
                # Simple price elasticity calculation: (% Change in Quantity Demanded) / (% Change in Price)
                # Using log-log model for a simplified elasticity estimate
                # ln(Q2/Q1) / ln(P2/P1) or regression coefficient in a log-log model
                try:
                    # Avoid division by zero or log of zero/negative values
                    temp_df = temp_df[(temp_df['PurchaseValue'] > 0) & (temp_df['PurchaseVolume'] > 0)]
                    if len(temp_df) > 1:
                        log_price = np.log(temp_df['PurchaseValue'])
                        log_volume = np.log(temp_df['PurchaseVolume'])
                        # Simple linear regression to get the coefficient
                        slope, intercept, r_value, p_value, std_err = stats.linregress(log_price, log_volume)
                        logger.info(f"Calculated price elasticity: {slope:.2f}")
                        return slope
                    else:
                        logger.warning("Not enough non-zero data points for elasticity calculation.")
                        return -0.5 # Default elasticity
                except Exception as e:
                    logger.warning(f"Error calculating price elasticity: {e}. Returning default.")
                    return -0.5 # Default elasticity if calculation fails
            else:
                logger.warning("Not enough data to calculate price elasticity.")
                return -0.5 # Default elasticity
        else:
            logger.warning("Missing 'PurchaseValue' or 'PurchaseVolume' columns for price elasticity calculation. Returning default.")
            return -0.5 # Default elasticity if columns are missing

    def llm_generate_scenarios(self, forecast_result: Dict[str, Any], news_summary: str, user_inputs: Optional[str] = None) -> List[Dict[str, Any]]:
        prompt = f"""
You are a retail analytics AI. Given the following forecast results and news summary, generate a set of what-if scenarios for:
- Sales forecasts under different promotions, price changes, and competition actions
- Cross-category impact (e.g., if eggs go up, what happens to substitutes?)
- Consumer segmentation (who is price sensitive, who is not)
- Suggest optimal promotions and pricing strategies
- Accept user/retailer input for custom scenarios and simulate outcomes

Forecast result:
{forecast_result}

News summary:
{news_summary}

User inputs (if any):
{user_inputs}

Return a JSON list of scenario descriptions and the logic for simulating each.
"""
        response = call_llm(prompt, max_tokens=1024, llm_config=self.llm_config)
        try:
            scenarios = json.loads(response) if response.strip().startswith('[') else []
        except Exception:
            scenarios = []
        self.llm_scenarios = scenarios
        return scenarios
    
    def simulate(self, scenarios: List[Dict[str, Any]], forecast_result: Dict[str, Any], 
                 purchase_data: pd.DataFrame, products_data: pd.DataFrame, clusters_data: pd.DataFrame) -> List[Dict[str, Any]]:
        # A more sophisticated simulation based on correlations and user inputs
        # Assumptions:
        # - 'date' column in purchase_data/merged_data for time series
        # - 'sales_value' column for target variable (e.g., 'PurchaseValue')
        # - 'price_per_unit' or similar in products_data or merged data for price (e.g., 'PurchaseValue')
        # - 'category' or 'brand' for product attributes (from products_data)
        # - 'cluster_name' or similar for cluster attributes (from clusters_data)

        # Merge data for comprehensive view
        merged_data = purchase_data.copy()
        if not products_data.empty and 'SingleBuyProductItemId' in merged_data.columns and 'ProductItemId' in products_data.columns:
            merged_data = pd.merge(merged_data, products_data.rename(columns={'ProductItemId': 'SingleBuyProductItemId'}), on='SingleBuyProductItemId', how='left')
        else:
            logger.warning("Products data not merged: missing required columns or empty products data.")

        if not clusters_data.empty and 'ClusterId' in merged_data.columns and 'ClusterId' in clusters_data.columns:
            merged_data = pd.merge(merged_data, clusters_data, on='ClusterId', how='left')
        else:
            logger.warning("Clusters data not merged: missing required columns or empty clusters data.")

        logger.info(f"Data merged for simulation. Final columns: {merged_data.columns.tolist()}")

        results = []
        base_sales = forecast_result.get('forecast_result', [])[-1].get('yhat', 100) if forecast_result.get('forecast_result') else 100 # Last forecasted sales

        # Ensure 'date' column is datetime and sort
        merged_data['date'] = pd.to_datetime(merged_data['date'])
        merged_data = merged_data.sort_values('date')

        # Calculate overall price elasticity for more sophisticated price simulations
        price_elasticity = self.generate_price_elasticity(merged_data)

        for scenario in scenarios:
            scenario_description = scenario.get('scenario', 'Generic Scenario')
            simulated_sales = base_sales # Start with base sales
            impact_assessment = "Neutral/Uncertain Impact (Simulated)"
            
            # Example: Simulate price increase impact on sales using calculated elasticity
            if "price change" in scenario_description.lower() or "price increase" in scenario_description.lower() or "price decrease" in scenario_description.lower():
                # Extract price change from scenario description if possible, else default
                price_change_factor = 1.0 # Default to no change
                match_increase = re.search(r'(\d+\.?\d*)% price increase', scenario_description.lower())
                match_decrease = re.search(r'(\d+\.?\d*)% price decrease', scenario_description.lower())
                if match_increase:
                    price_change_factor = 1 + float(match_increase.group(1)) / 100
                elif match_decrease:
                    price_change_factor = 1 - float(match_decrease.group(1)) / 100
                
                if price_change_factor != 1.0:
                    # Apply elasticity: % change in quantity = elasticity * % change in price
                    percent_price_change = price_change_factor - 1
                    percent_volume_change = price_elasticity * percent_price_change
                    simulated_sales = base_sales * (1 + percent_volume_change)
                    
                    if percent_volume_change < 0:
                        impact_assessment = "Negative Impact (Simulated)"
                    elif percent_volume_change > 0:
                        impact_assessment = "Positive Impact (Simulated)"
                    else:
                        impact_assessment = "Neutral Impact (Simulated)"
                else:
                    impact_assessment = "No explicit price change specified." # If regex didn't find specific % change

            elif "promotion" in scenario_description.lower():
                promotion_effect = 1.10 # Example: 10% sales lift due to promotion
                simulated_sales = base_sales * promotion_effect
                impact_assessment = "Positive Impact (Simulated)"
            elif "cross-category impact" in scenario_description.lower():
                # Placeholder for more complex cross-category logic using products_data
                # E.g., if coffee prices rise, tea sales might increase.
                # This would require more sophisticated modeling beyond this demo's scope.
                simulated_sales = base_sales * np.random.uniform(0.95, 1.05) # Small random fluctuation for demo
                impact_assessment = "Potential Cross-Category Impact (Simulated)"
            else:
                simulated_sales = base_sales * np.random.uniform(0.9, 1.1) 
                impact_assessment = "Neutral/Uncertain Impact (Simulated)"
            
            results.append({
                'scenario': scenario_description,
                'simulated_sales': simulated_sales,
                'impact_assessment': impact_assessment,
                'price_elasticity_used': price_elasticity # Add elasticity to results for transparency
            })
        self.simulation_results = results
        return results
    
    def llm_summarize_simulation(self, simulation_results: List[Dict[str, Any]]) -> str:
        prompt = f"""
You are a business analyst AI. Given the following simulation results, summarize the key findings, including:
- Which scenarios are most beneficial or risky
- What actions are recommended for the retailer
- Any cross-category or consumer insights

Simulation results:
{json.dumps(simulation_results, indent=2)}
"""
        self.llm_summary = call_llm(prompt, max_tokens=512, llm_config=self.llm_config)
        return self.llm_summary
    
    def run(self, forecast_result: Dict[str, Any], news_summary: str, 
            purchase_data: pd.DataFrame, products_data: pd.DataFrame, clusters_data: pd.DataFrame, 
            user_inputs: Optional[str] = None) -> Dict[str, Any]:
        
        scenarios = self.llm_generate_scenarios(forecast_result, news_summary, user_inputs)
        sim_results = self.simulate(scenarios, forecast_result, purchase_data, products_data, clusters_data)
        summary = self.llm_summarize_simulation(sim_results)
        return {
            'scenarios': scenarios,
            'simulation_results': sim_results,
            'llm_summary': summary
        }

async def simulation_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Main simulation agent node function"""
    logger.info("Executing Simulation Agent Node")
    try:
        # Expect preprocessed data from the state
        data = state['data']
        purchase_data = data['purchase_data']
        products_data = data['products']
        clusters_data = data['clusters']

        forecast_result = state.get('forecast_result', {})
        news_summary = state.get('news_summary', None)
        user_inputs = state.get('user_inputs', None)
        config = state['config'] # Access config from state
        llm_config = config.get('llm', {}) # Get LLM config from state

        if not forecast_result or not news_summary:
            raise ValueError("Missing forecast or news results for simulation.")

        agent = SimulationAgent(llm_config=llm_config)
        simulation_output = agent.run(
            forecast_result, 
            news_summary, 
            purchase_data, 
            products_data, 
            clusters_data,
            user_inputs
        )

        state['simulation_result'] = simulation_output
        return state
        
    except Exception as e:
        logger.error(f"Error in simulation agent node: {e}")
        state['error'] = str(e)
        return state