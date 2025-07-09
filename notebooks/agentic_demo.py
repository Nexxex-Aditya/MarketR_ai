# ===== CELL 1: Setup =====
import os
import pandas as pd
from core.agentic_pipeline import AgenticPipeline
from agents.forecast_agent import ForecastAgent
from agents.news_agent import NewsAgent
from agents.simulation_agent import SimulationAgent
from agents.summary_agent import SummaryAgent

# Path to your data
DATA_PATH = '../data/us_spirits_purchase_data.csv'
USER_QUERY = 'Analyze trends in premium spirits sales and their correlation with market clusters'
USER_INPUTS = None
BUSINESS_RULES = '''
- Always mention % growth in sales and % impact on sales
- Highlight any region with >5% impact
- Summarize cross-category effects if present
- Provide actionable recommendations
'''

# ===== CELL 2: Forecast Agent =====
forecast_agent = ForecastAgent(DATA_PATH)
forecast_out = forecast_agent.run(USER_QUERY)
print('Model chosen:', forecast_out['model_choice'])
print('LLM reasoning:', forecast_out['llm_reasoning'])
pd.DataFrame(forecast_out['forecast_result']).head()

# ===== CELL 3: News Agent =====
news_agent = NewsAgent(USER_QUERY)
news_out = news_agent.run(forecast_context=forecast_out['llm_reasoning'])
print('News summary:', news_out['llm_summary'])
for art in news_out['articles'][:3]:
    print('-', art['title'])

# ===== CELL 4: Simulation Agent =====
simulation_agent = SimulationAgent(DATA_PATH)
simulation_out = simulation_agent.run(forecast_out['forecast_result'], news_out['llm_summary'], USER_INPUTS)
print('Simulation summary:', simulation_out['llm_summary'])
for sc in simulation_out['scenarios'][:2]:
    print('Scenario:', sc)

# ===== CELL 5: Summary Agent =====
summary_agent = SummaryAgent()
summary_out = summary_agent.run(forecast_out['forecast_result'], news_out['llm_summary'], simulation_out['llm_summary'], BUSINESS_RULES)
print('Executive summary:', summary_out['executive_summary'])
print('Headlines:', summary_out['headlines'])

# ===== CELL 6: Full Pipeline =====
pipeline = AgenticPipeline(DATA_PATH)
results = pipeline.run(USER_QUERY, USER_INPUTS, BUSINESS_RULES)
print('PPT generated at:', results['ppt_path'])

# ===== CELL 7: Download PPT =====
from IPython.display import FileLink
FileLink(results['ppt_path']) 