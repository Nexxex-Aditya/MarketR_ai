import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import json
from pathlib import Path
import os

from workflows.synergistic_workflow import build_synergistic_workflow

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

async def run_agentic_pipeline(query: str,
                             data_path: str,
                             max_retries: int = 3,
                             timeout: int = 30) -> Dict[str, Any]:
    """Run the enhanced agentic pipeline using LangGraph."""
    try:
        # Initialize the LangGraph workflow
        workflow = build_synergistic_workflow()

        # Initialize state for the LangGraph workflow
        state = {
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'max_retries': max_retries,
            'timeout': timeout,
            'data_path': data_path  # Pass data_path to the state
        }
        
        # Execute the workflow
        logger.info("Starting LangGraph workflow execution...")
        # Assuming the workflow.invoke method exists and processes the state
        final_state = await workflow.ainvoke(state)

        # Extract results from the final state
        summary = {
            'query': query,
            'timestamp': final_state['timestamp'],
            'forecast': final_state.get('forecast_result', {}),
            'news': final_state.get('news_results', {}),
            'simulation': final_state.get('simulation_result', {}),
            'ppt_path': final_state.get('ppt_file', None)
        }

        return summary
        
    except Exception as e:
        logger.error(f"Pipeline error: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    query = "Egg prices in US"
    # For example usage, you might need a dummy data_path or adapt
    dummy_data_path = "data/us_spirits_purchase_data.csv" # Or any other data file
    result = asyncio.run(run_agentic_pipeline(query, dummy_data_path))
    print(json.dumps(result, indent=2)) 