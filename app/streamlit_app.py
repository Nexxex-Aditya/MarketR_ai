import streamlit as st
import pandas as pd
import os
import asyncio
# from core.agentic_pipeline import run_agentic_pipeline # Old import
import logging
from pathlib import Path
from src.main import Orchestrator # New import

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None

# Set up data paths (these are defaults, actual paths come from config.yaml)
DATA_DIR = Path("data")
PURCHASE_DATA_PATH = DATA_DIR / "us_spirits_purchase_data.csv"
PRODUCTS_DATA_PATH = DATA_DIR / "products.csv"
CLUSTERS_DATA_PATH = DATA_DIR / "clusters.csv"
CONFIG_PATH = Path("config/config.yaml") # Path to configuration file

st.set_page_config(page_title="NIQ Spirits Analysis", layout="wide")
st.title("NIQ Spirits Market Analysis System")
st.write("Analyze spirits purchase data, market clusters, and generate insights.")

# Verify data files exist (and config file)
if not all(path.exists() for path in [PURCHASE_DATA_PATH, PRODUCTS_DATA_PATH, CLUSTERS_DATA_PATH, CONFIG_PATH]):
    st.error("Required data or configuration files are missing. Please ensure all data files are present in the data directory and config/config.yaml exists.")
    st.stop()

# Load and display data summary (using a simplified method for UI)
@st.cache_data
def load_data_summary():
    try:
        # These paths are for display only, actual loading uses DataHandler with config paths
        purchase_data = pd.read_csv(PURCHASE_DATA_PATH, nrows=5)  # Just for preview
        products_data = pd.read_csv(PRODUCTS_DATA_PATH, nrows=5)
        clusters_data = pd.read_csv(CLUSTERS_DATA_PATH, nrows=5)
        
        return {
            'purchase_columns': purchase_data.columns.tolist(),
            'products_columns': products_data.columns.tolist(),
            'clusters_columns': clusters_data.columns.tolist()
        }
    except Exception as e:
        logger.error(f"Error loading data for summary display: {str(e)}")
        st.warning(f"Could not load data summary for display: {e}")
        return None

# Display data summary
data_summary = load_data_summary()
if data_summary:
    st.subheader("Available Data Preview (Columns)")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("Purchase Data Columns:")
        st.write(data_summary['purchase_columns'])
    
    with col2:
        st.write("Products Data Columns:")
        st.write(data_summary['products_columns'])
    
    with col3:
        st.write("Clusters Data Columns:")
        st.write(data_summary['clusters_columns'])
else:
    st.warning("Data summary could not be loaded. Please check data files.")


# Analysis options
st.subheader("Analysis Options")

# Business query input with predefined options
query_options = [
    "Analyze premium spirits sales trends and market clusters",
    "Identify growth opportunities in specific market segments",
    "Analyze price sensitivity across different clusters",
    "Forecast sales for next quarter and identify key drivers",
    "Custom query..."
]

selected_query = st.selectbox("Select analysis type or enter custom query", query_options)

if selected_query == "Custom query...":
    user_query = st.text_input("Enter your custom business query here:")
    if not user_query:
        st.warning("Please enter a custom query to proceed.")
        st.stop()
else:
    user_query = selected_query

# Optional: user inputs for simulation
user_inputs = st.text_area("(Optional) Enter custom scenario inputs for simulation (e.g., 'What if there is a 5% price increase for premium spirits?'):", value="")

# Business rules
default_rules = """
- Focus on premium spirits categories
- Analyze cluster-specific trends
- Highlight significant price points
- Consider seasonal patterns
- Provide actionable market insights
"""
business_rules = st.text_area("Business rules for analysis (optional, e.g., 'Focus on actionable insights for increasing sales volume.'):", value=default_rules)

async def run_orchestrator_pipeline(query, user_inputs, business_rules):
    try:
        orchestrator = Orchestrator(config_path=str(CONFIG_PATH))
        results = orchestrator.run_analysis(
            user_query=query,
            user_inputs=user_inputs if user_inputs else None,
            business_rules=business_rules if business_rules else None
        )
        return results
    except Exception as e:
        logger.error(f"Orchestrator pipeline error: {str(e)}")
        st.error(f"An error occurred during analysis: {str(e)}")
        return {'error': str(e)}

if st.button("Run Analysis Pipeline", help="Click to start the multi-agent AI analysis."):
    st.session_state.results = None # Clear previous results
    if not user_query:
        st.error("Please select or enter a business query.")
    else:
        with st.spinner("Running comprehensive analysis... This may take a few minutes."):
            results = asyncio.run(run_orchestrator_pipeline(user_query, user_inputs, business_rules))
            st.session_state.results = results

if st.session_state.results:
    if 'error' in st.session_state.results:
        st.error(f"Analysis failed: {st.session_state.results['error']}")
    else:
        st.success("Analysis complete! Download your report below.")
        report_output = st.session_state.results.get('report_output', {})
        
        # Download button for PPT
        if report_output.get('generated_reports') and report_output['generated_reports'].get('powerpoint'):
            report_path = report_output['generated_reports']['powerpoint']
            if Path(report_path).exists():
                with open(report_path, 'rb') as f:
                    st.download_button(
                        label="Download Analysis Report (PowerPoint)",
                        data=f.read(),
                        file_name=os.path.basename(report_path),
                        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
                    )
            else:
                st.warning(f"Generated report file not found at {report_path}.")
        else:
            st.info("No PowerPoint report was generated.")
        
        # Display key summary from LLM
        st.subheader("Executive Summary")
        summary_data = report_output.get('summary_data', {})
        if summary_data.get('executive_summary'):
            st.write(summary_data['executive_summary'])
        else:
            st.write("No executive summary available.")

        st.subheader("Key Headlines")
        if summary_data.get('headlines'):
            for headline in summary_data['headlines']:
                st.markdown(f"- {headline}")
        else:
            st.write("No key headlines available.")

        st.subheader("Detailed Analysis Insights")
        st.info("Please download the PowerPoint report for comprehensive details on forecasts, news analysis, and simulation results.")

        # Optional: Display raw agent outputs for debugging/transparency (can be toggled)
        with st.expander("View Raw Agent Outputs (For Debugging)"):
            st.json(st.session_state.results) 