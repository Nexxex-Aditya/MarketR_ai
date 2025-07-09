import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import sys
import logging
from dotenv import load_dotenv

# Add src to Python path
sys.path.append(str(Path(__file__).parent))

from data_processing.data_handler import DataHandler
from agents.forecasting_agent import ForecastingAgent
from agents.report_agent import ReportAgent
from utils.llm_utils import call_llm, format_llm_prompt
from utils.presentation_generator import PresentationGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class StreamlitApp:
    def __init__(self):
        """Initialize the Streamlit application."""
        # Initialize configuration
        self.config = {
            'data': {
                'purchase_data': 'data/us_spirits_purchase_data.csv',
                'products_data': 'data/products.csv',
                'clusters_data': 'data/clusters.csv'
            },
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
        
        # Initialize components with config
        self.data_handler = DataHandler(self.config)
        self.forecasting_agent = ForecastingAgent()
        self.presentation_generator = PresentationGenerator()
        self._initialize_session_state()
        self._setup_page_config()
        
    def _initialize_session_state(self):
        """Initialize session state variables."""
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'purchase_data' not in st.session_state:
            st.session_state.purchase_data = None
        if 'products_data' not in st.session_state:
            st.session_state.products_data = None
        if 'clusters_data' not in st.session_state:
            st.session_state.clusters_data = None
        if 'forecast_results' not in st.session_state:
            st.session_state.forecast_results = None
        if 'llm_connection' not in st.session_state:
            st.session_state.llm_connection = None
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
            
    def _setup_page_config(self):
        """Set up the Streamlit page configuration."""
        st.set_page_config(
            page_title="NIQ Hackfest - Data Analysis & Forecasting",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
    def _initialize_llm_connection(self):
        """Initialize and maintain LLM connection."""
        try:
            if st.session_state.llm_connection is None:
                # Test connection with a simple prompt
                test_response = call_llm("Test connection")
                if test_response:
                    st.session_state.llm_connection = True
                    logger.info("LLM connection established successfully")
                else:
                    st.session_state.llm_connection = False
                    logger.error("Failed to establish LLM connection")
        except Exception as e:
            st.session_state.llm_connection = False
            logger.error(f"Error initializing LLM connection: {str(e)}")
            
    def _maintain_llm_connection(self):
        """Maintain LLM connection throughout the session."""
        if not st.session_state.llm_connection:
            self._initialize_llm_connection()
            
        # Periodically check connection
        if st.session_state.llm_connection:
            try:
                # Send a heartbeat request
                call_llm("Heartbeat check")
            except Exception as e:
                logger.error(f"LLM connection lost: {str(e)}")
                st.session_state.llm_connection = False
                self._initialize_llm_connection()
                
    def _handle_user_query(self, query: str):
        """Handle user queries with continuous LLM connection."""
        self._maintain_llm_connection()
        
        if not st.session_state.llm_connection:
            st.error("LLM connection is not available. Please try again.")
            return
            
        try:
            # Add query to chat history
            st.session_state.chat_history.append({"role": "user", "content": query})
            
            # Get response from LLM
            response = call_llm(query)
            
            # Add response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response['analysis']})
            
            return response
        except Exception as e:
            logger.error(f"Error handling user query: {str(e)}")
            st.error("Error processing your query. Please try again.")
            return None
            
    def load_data(self):
        """Load and preprocess data."""
        try:
            # Load data
            purchase_data = self.data_handler.load_purchase_data()
            products_data = self.data_handler.load_products_data()
            clusters_data = self.data_handler.load_clusters_data()
            
            # Store in session state
            st.session_state.purchase_data = purchase_data
            st.session_state.products_data = products_data
            st.session_state.clusters_data = clusters_data
            st.session_state.data_loaded = True
            
            st.success("Data loaded successfully!")
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            
    def generate_forecast(self, data: pd.DataFrame):
        """Generate forecasts using the forecasting agent."""
        try:
            # Create progress bar
            progress_bar = st.progress(0)
            st.text('Initializing forecasting model...')
            progress_bar.progress(25)
            
            # Prepare data
            st.text('Preparing data...')
            progress_bar.progress(50)
            
            # Generate forecast
            st.text('Running forecasting model...')
            forecast_results = self.forecasting_agent.forecast(data, 'sales_value')
            progress_bar.progress(75)
            
            # Save results
            st.session_state.forecast_results = forecast_results
            progress_bar.progress(100)
            
            st.success("Forecast generated successfully!")
            
        except Exception as e:
            st.error(f"Error generating forecast: {str(e)}")
            
    def display_forecast(self):
        """Display forecast results."""
        if st.session_state.forecast_results:
            results = st.session_state.forecast_results
            
            # Create forecast plot
            fig = go.Figure()
            
            # Add forecast line
            fig.add_trace(go.Scatter(
                x=results['forecast']['dates'],
                y=results['forecast']['predictions'],
                name='Forecast',
                line=dict(color='blue')
            ))
            
            # Add confidence intervals
            fig.add_trace(go.Scatter(
                x=results['forecast']['dates'],
                y=results['forecast']['upper_bound'],
                name='Upper Bound',
                line=dict(color='rgba(0,0,255,0.2)'),
                fill=None
            ))
            
            fig.add_trace(go.Scatter(
                x=results['forecast']['dates'],
                y=results['forecast']['lower_bound'],
                name='Lower Bound',
                line=dict(color='rgba(0,0,255,0.2)'),
                fill='tonexty'
            ))
            
            # Update layout
            fig.update_layout(
                title='Sales Value Forecast',
                xaxis_title='Date',
                yaxis_title='Sales Value',
                hovermode='x unified'
            )
            
            # Display plot
            st.plotly_chart(fig, use_container_width=True)
            
            # Display metrics
            st.subheader("Forecast Metrics")
            metrics_df = pd.DataFrame({
                'Metric': ['RMSE', 'MSE'],
                'Value': [results['metrics']['rmse'], results['metrics']['mse']]
            })
            st.table(metrics_df)
            
    def generate_presentation(self):
        """Generate PowerPoint presentation with market analysis."""
        try:
            if not st.session_state.data_loaded:
                st.error("Please load data first!")
                return
                
            with st.spinner("Generating presentation..."):
                # Get data and results
                data = st.session_state.purchase_data
                forecast_results = st.session_state.forecast_results
                analysis_results = self.data_handler.analyze_data(data)
                
                # Generate presentation
                output_path = self.presentation_generator.generate_presentation(
                    data=data,
                    forecast_results=forecast_results,
                    analysis_results=analysis_results
                )
                
                st.success(f"Presentation generated successfully! Saved to: {output_path}")
                
                # Provide download link
                with open(output_path, 'rb') as f:
                    st.download_button(
                        label="Download Presentation",
                        data=f,
                        file_name=os.path.basename(output_path),
                        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
                    )
                    
        except Exception as e:
            st.error(f"Error generating presentation: {str(e)}")
            
    def run(self):
        """Run the Streamlit application."""
        st.title("NIQ Hackfest - Data Analysis & Forecasting")
        
        # Initialize LLM connection
        self._initialize_llm_connection()
        
        # Sidebar
        with st.sidebar:
            st.header("Data Loading")
            if st.button("Load Data"):
                self.load_data()
                
            st.header("Forecasting")
            if st.button("Generate Forecast"):
                if not st.session_state.data_loaded:
                    st.error("Please load data first!")
                else:
                    self.generate_forecast(st.session_state.purchase_data)
                    
            st.header("Presentation")
            if st.button("Generate Market Analysis Presentation"):
                self.generate_presentation()
                
            st.header("Chat Interface")
            user_query = st.text_input("Ask a question about the data:")
            if user_query:
                response = self._handle_user_query(user_query)
                if response:
                    st.write("Analysis:", response['analysis'])
                    st.write("Recommendations:", response['recommendations'])
                    
        # Main content
        if st.session_state.data_loaded:
            # Display data
            st.header("Data Preview")
            st.dataframe(st.session_state.purchase_data.head())
            
            # Display forecast if available
            if st.session_state.forecast_results:
                st.header("Forecast Results")
                self.display_forecast()
                
            # Display chat history
            st.header("Chat History")
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.write("You:", message["content"])
                else:
                    st.write("Assistant:", message["content"])
                    
if __name__ == "__main__":
    app = StreamlitApp()
    app.run() 