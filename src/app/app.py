import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import yaml
import json
from datetime import datetime, timedelta
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.forecasting_agent import ForecastingAgent
from agents.news_agent import NewsAgent
from agents.simulation_agent import SimulationAgent
from agents.report_agent import ReportAgent
from data_processing.data_processor import DataProcessor

def load_config():
    """Load configuration from YAML file."""
    config_path = Path('config/config.yaml')
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}

def load_data():
    """Load processed data and results."""
    data = {}
    
    # Load processed data
    processed_data_path = Path('processed_data/processed_data.parquet')
    if processed_data_path.exists():
        data['processed'] = pd.read_parquet(processed_data_path)
    
    # Load forecast results
    forecast_path = Path('output/forecasts/forecasts.csv')
    if forecast_path.exists():
        data['forecasts'] = pd.read_csv(forecast_path)
    
    # Load news data
    news_path = Path('output/news/news_data.csv')
    if news_path.exists():
        data['news'] = pd.read_csv(news_path)
    
    # Load simulation results
    sim_path = Path('output/simulations/scenario_results.csv')
    if sim_path.exists():
        data['simulations'] = pd.read_csv(sim_path)
    
    return data

def main():
    st.set_page_config(
        page_title="Spirits Market Analysis",
        page_icon="🥃",
        layout="wide"
    )
    
    st.title("Spirits Market Analysis Dashboard")
    
    # Load configuration and data
    config = load_config()
    data = load_data()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Overview", "Data Analysis", "Forecasting", "Market Insights", "Simulation", "Reports"]
    )
    
    if page == "Overview":
        show_overview(data)
    elif page == "Data Analysis":
        show_data_analysis(data)
    elif page == "Forecasting":
        show_forecasting(data, config)
    elif page == "Market Insights":
        show_market_insights(data)
    elif page == "Simulation":
        show_simulation(data, config)
    elif page == "Reports":
        show_reports(data)

def show_overview(data):
    """Display overview dashboard."""
    st.header("Market Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    if 'processed' in data:
        with col1:
            st.metric(
                "Total Sales",
                f"${data['processed']['sales'].sum():,.2f}",
                f"{data['processed']['sales'].pct_change().mean()*100:.1f}%"
            )
        with col2:
            st.metric(
                "Average Price",
                f"${data['processed']['price'].mean():,.2f}",
                f"{data['processed']['price'].pct_change().mean()*100:.1f}%"
            )
        with col3:
            st.metric(
                "Total Products",
                f"{data['processed']['product_id'].nunique():,}",
                f"{data['processed']['product_id'].nunique()/data['processed']['product_id'].nunique()*100:.1f}%"
            )
        with col4:
            st.metric(
                "Total Territories",
                f"{data['processed']['territory'].nunique():,}",
                f"{data['processed']['territory'].nunique()/data['processed']['territory'].nunique()*100:.1f}%"
            )
    
    # Sales trend
    if 'processed' in data:
        st.subheader("Sales Trend")
        sales_trend = data['processed'].groupby('date')['sales'].sum().reset_index()
        fig = px.line(sales_trend, x='date', y='sales', title='Daily Sales Trend')
        st.plotly_chart(fig, use_container_width=True)
    
    # Forecast vs Actual
    if 'forecasts' in data and 'processed' in data:
        st.subheader("Forecast vs Actual")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data['forecasts']['date'],
            y=data['forecasts']['forecast'],
            name='Forecast',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=data['forecasts']['date'],
            y=data['forecasts']['actual'],
            name='Actual',
            line=dict(color='red')
        ))
        st.plotly_chart(fig, use_container_width=True)

def show_data_analysis(data):
    """Display data analysis page."""
    st.header("Data Analysis")
    
    if 'processed' not in data:
        st.warning("No processed data available. Please run the data processing pipeline first.")
        return
    
    # Data filters
    st.subheader("Filters")
    col1, col2 = st.columns(2)
    
    with col1:
        territories = st.multiselect(
            "Select Territories",
            options=data['processed']['territory'].unique(),
            default=data['processed']['territory'].unique()[:3]
        )
    
    with col2:
        date_range = st.date_input(
            "Select Date Range",
            value=(
                data['processed']['date'].min(),
                data['processed']['date'].max()
            )
        )
    
    # Filter data
    filtered_data = data['processed'][
        (data['processed']['territory'].isin(territories)) &
        (data['processed']['date'].between(date_range[0], date_range[1]))
    ]
    
    # Sales by territory
    st.subheader("Sales by Territory")
    territory_sales = filtered_data.groupby('territory')['sales'].sum().reset_index()
    fig = px.bar(territory_sales, x='territory', y='sales', title='Total Sales by Territory')
    st.plotly_chart(fig, use_container_width=True)
    
    # Price distribution
    st.subheader("Price Distribution")
    fig = px.histogram(filtered_data, x='price', title='Price Distribution')
    st.plotly_chart(fig, use_container_width=True)
    
    # Sales by product category
    if 'category' in filtered_data.columns:
        st.subheader("Sales by Category")
        category_sales = filtered_data.groupby('category')['sales'].sum().reset_index()
        fig = px.pie(category_sales, values='sales', names='category', title='Sales by Category')
        st.plotly_chart(fig, use_container_width=True)

def show_forecasting(data, config):
    """Display forecasting page."""
    st.header("Forecasting")
    
    if 'forecasts' not in data:
        st.warning("No forecast data available. Please run the forecasting pipeline first.")
        return
    
    # Forecast parameters
    st.subheader("Forecast Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        forecast_horizon = st.slider(
            "Forecast Horizon (days)",
            min_value=7,
            max_value=90,
            value=config.get('forecasting', {}).get('forecast_horizon', 30)
        )
    
    with col2:
        confidence_level = st.slider(
            "Confidence Level",
            min_value=0.8,
            max_value=0.99,
            value=config.get('forecasting', {}).get('confidence_level', 0.95),
            step=0.01
        )
    
    # Generate new forecast
    if st.button("Generate New Forecast"):
        with st.spinner("Generating forecast..."):
            forecaster = ForecastingAgent(config)
            new_forecast = forecaster.forecast(data['processed'])
            data['forecasts'] = new_forecast
            st.success("Forecast generated successfully!")
    
    # Display forecast
    st.subheader("Forecast Results")
    fig = go.Figure()
    
    # Add actual values
    fig.add_trace(go.Scatter(
        x=data['forecasts']['date'],
        y=data['forecasts']['actual'],
        name='Actual',
        line=dict(color='blue')
    ))
    
    # Add forecast
    fig.add_trace(go.Scatter(
        x=data['forecasts']['date'],
        y=data['forecasts']['forecast'],
        name='Forecast',
        line=dict(color='red')
    ))
    
    # Add confidence intervals
    fig.add_trace(go.Scatter(
        x=data['forecasts']['date'],
        y=data['forecasts']['upper_bound'],
        name='Upper Bound',
        line=dict(color='rgba(255,0,0,0.2)'),
        fill=None
    ))
    
    fig.add_trace(go.Scatter(
        x=data['forecasts']['date'],
        y=data['forecasts']['lower_bound'],
        name='Lower Bound',
        line=dict(color='rgba(255,0,0,0.2)'),
        fill='tonexty'
    ))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Forecast metrics
    st.subheader("Forecast Metrics")
    metrics = pd.DataFrame({
        'Metric': ['MAE', 'MSE', 'RMSE', 'MAPE'],
        'Value': [
            data['forecasts']['mae'].mean(),
            data['forecasts']['mse'].mean(),
            data['forecasts']['rmse'].mean(),
            data['forecasts']['mape'].mean()
        ]
    })
    st.table(metrics)

def show_market_insights(data):
    """Display market insights page."""
    st.header("Market Insights")
    
    if 'news' not in data:
        st.warning("No news data available. Please run the news collection pipeline first.")
        return
    
    # News sentiment
    st.subheader("News Sentiment Analysis")
    sentiment_data = data['news'].groupby('date')['sentiment'].mean().reset_index()
    fig = px.line(sentiment_data, x='date', y='sentiment', title='News Sentiment Trend')
    st.plotly_chart(fig, use_container_width=True)
    
    # Recent news
    st.subheader("Recent News")
    for _, row in data['news'].sort_values('date', ascending=False).head(10).iterrows():
        with st.expander(f"{row['date']} - {row['title']}"):
            st.write(row['content'])
            st.write(f"Source: {row['source']}")
            st.write(f"Sentiment: {row['sentiment']:.2f}")
    
    # Market trends
    if 'market_data' in data:
        st.subheader("Market Trends")
        market_data = pd.DataFrame(data['market_data'])
        fig = px.line(market_data, x='date', y='value', color='indicator', title='Market Indicators')
        st.plotly_chart(fig, use_container_width=True)

def show_simulation(data, config):
    """Display simulation page."""
    st.header("Scenario Simulation")
    
    if 'simulations' not in data:
        st.warning("No simulation data available. Please run the simulation pipeline first.")
        return
    
    # Simulation parameters
    st.subheader("Simulation Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        feature = st.selectbox(
            "Select Feature",
            options=['price', 'promotion', 'seasonality']
        )
    
    with col2:
        change = st.slider(
            "Change Percentage",
            min_value=-50,
            max_value=50,
            value=10,
            step=5
        )
    
    # Run simulation
    if st.button("Run Simulation"):
        with st.spinner("Running simulation..."):
            simulator = SimulationAgent(config)
            new_simulation = simulator.run_simulation(data['processed'], feature, change)
            data['simulations'] = new_simulation
            st.success("Simulation completed successfully!")
    
    # Display simulation results
    st.subheader("Simulation Results")
    
    # Impact analysis
    impact_data = data['simulations'].groupby('scenario')['impact'].mean().reset_index()
    fig = px.bar(impact_data, x='scenario', y='impact', title='Impact of Scenarios')
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance
    if 'feature_importance' in data['simulations']:
        st.subheader("Feature Importance")
        importance_data = pd.DataFrame(data['simulations']['feature_importance'])
        fig = px.bar(importance_data, x='feature', y='importance', title='Feature Importance')
        st.plotly_chart(fig, use_container_width=True)

def show_reports(data):
    """Display reports page."""
    st.header("Generated Reports")
    
    # Report selection
    report_type = st.selectbox(
        "Select Report Type",
        options=['Executive Summary', 'Detailed Analysis', 'Presentation']
    )
    
    if report_type == 'Executive Summary':
        st.subheader("Executive Summary")
        if 'report_results' in data:
            st.write(data['report_results']['executive_summary'])
        else:
            st.warning("No executive summary available. Please generate reports first.")
    
    elif report_type == 'Detailed Analysis':
        st.subheader("Detailed Analysis")
        if 'report_results' in data:
            for section in data['report_results']['sections']:
                with st.expander(section['title']):
                    st.write(section['content'])
        else:
            st.warning("No detailed analysis available. Please generate reports first.")
    
    elif report_type == 'Presentation':
        st.subheader("Presentation")
        if 'report_results' in data and 'presentation_url' in data['report_results']:
            st.write(f"Presentation available at: {data['report_results']['presentation_url']}")
        else:
            st.warning("No presentation available. Please generate reports first.")
    
    # Generate new report
    if st.button("Generate New Report"):
        with st.spinner("Generating report..."):
            reporter = ReportAgent(config)
            new_report = reporter.generate_report(
                data.get('forecasts', {}),
                data.get('news', {}),
                data.get('simulations', {})
            )
            data['report_results'] = new_report
            st.success("Report generated successfully!")

if __name__ == "__main__":
    main() 