import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import json
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional, Union

def load_config(config_path: Optional[str] = None) -> Dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path('config/config.yaml')
    
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}

def load_data(data_path: Union[str, Path]) -> pd.DataFrame:
    """Load data from various file formats."""
    data_path = Path(data_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    if data_path.suffix == '.csv':
        return pd.read_csv(data_path)
    elif data_path.suffix == '.parquet':
        return pd.read_parquet(data_path)
    elif data_path.suffix == '.json':
        return pd.read_json(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}")

def save_data(data: pd.DataFrame, output_path: Union[str, Path], format: str = 'csv'):
    """Save data to various file formats."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'csv':
        data.to_csv(output_path, index=False)
    elif format == 'parquet':
        data.to_parquet(output_path, index=False)
    elif format == 'json':
        data.to_json(output_path, orient='records')
    else:
        raise ValueError(f"Unsupported format: {format}")

def create_time_series_plot(
    data: pd.DataFrame,
    date_col: str,
    value_col: str,
    title: str,
    confidence_intervals: Optional[Dict[str, List[float]]] = None
) -> go.Figure:
    """Create a time series plot with optional confidence intervals."""
    fig = go.Figure()
    
    # Add main line
    fig.add_trace(go.Scatter(
        x=data[date_col],
        y=data[value_col],
        name=value_col,
        line=dict(color='blue')
    ))
    
    # Add confidence intervals if provided
    if confidence_intervals:
        fig.add_trace(go.Scatter(
            x=data[date_col],
            y=confidence_intervals['upper'],
            name='Upper Bound',
            line=dict(color='rgba(255,0,0,0.2)'),
            fill=None
        ))
        
        fig.add_trace(go.Scatter(
            x=data[date_col],
            y=confidence_intervals['lower'],
            name='Lower Bound',
            line=dict(color='rgba(255,0,0,0.2)'),
            fill='tonexty'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Value',
        hovermode='x unified'
    )
    
    return fig

def create_bar_plot(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    color_col: Optional[str] = None
) -> go.Figure:
    """Create a bar plot with optional color grouping."""
    fig = px.bar(
        data,
        x=x_col,
        y=y_col,
        color=color_col,
        title=title
    )
    
    fig.update_layout(
        xaxis_title=x_col,
        yaxis_title=y_col,
        hovermode='x unified'
    )
    
    return fig

def create_pie_plot(
    data: pd.DataFrame,
    names_col: str,
    values_col: str,
    title: str
) -> go.Figure:
    """Create a pie plot."""
    fig = px.pie(
        data,
        names=names_col,
        values=values_col,
        title=title
    )
    
    return fig

def calculate_metrics(
    actual: np.ndarray,
    predicted: np.ndarray
) -> Dict[str, float]:
    """Calculate common regression metrics."""
    mae = np.mean(np.abs(actual - predicted))
    mse = np.mean((actual - predicted) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mape': mape
    }

def format_currency(value: float) -> str:
    """Format number as currency."""
    return f"${value:,.2f}"

def format_percentage(value: float) -> str:
    """Format number as percentage."""
    return f"{value:.1f}%"

def get_date_range(
    start_date: datetime,
    end_date: datetime,
    freq: str = 'D'
) -> pd.DatetimeIndex:
    """Generate date range with specified frequency."""
    return pd.date_range(start=start_date, end=end_date, freq=freq)

def filter_data_by_date_range(
    data: pd.DataFrame,
    date_col: str,
    start_date: datetime,
    end_date: datetime
) -> pd.DataFrame:
    """Filter data by date range."""
    return data[
        (data[date_col] >= start_date) &
        (data[date_col] <= end_date)
    ]

def filter_data_by_categories(
    data: pd.DataFrame,
    category_col: str,
    categories: List[str]
) -> pd.DataFrame:
    """Filter data by categories."""
    return data[data[category_col].isin(categories)]

def aggregate_data(
    data: pd.DataFrame,
    group_cols: List[str],
    agg_cols: Dict[str, str]
) -> pd.DataFrame:
    """Aggregate data by specified columns."""
    return data.groupby(group_cols).agg(agg_cols).reset_index()

def calculate_growth_rate(
    data: pd.DataFrame,
    value_col: str,
    date_col: str,
    period: str = 'D'
) -> pd.DataFrame:
    """Calculate growth rate for specified period."""
    return data.set_index(date_col)[value_col].pct_change(periods=1).reset_index()

def create_summary_statistics(data: pd.DataFrame) -> pd.DataFrame:
    """Create summary statistics for numeric columns."""
    return data.describe().T.reset_index()

def detect_anomalies(
    data: pd.DataFrame,
    value_col: str,
    threshold: float = 3.0
) -> pd.DataFrame:
    """Detect anomalies using z-score method."""
    z_scores = np.abs((data[value_col] - data[value_col].mean()) / data[value_col].std())
    return data[z_scores > threshold]

def create_correlation_matrix(
    data: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """Create correlation matrix for numeric columns."""
    if numeric_cols is None:
        numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    return data[numeric_cols].corr()

def create_heatmap(
    data: pd.DataFrame,
    title: str
) -> go.Figure:
    """Create correlation heatmap."""
    fig = go.Figure(data=go.Heatmap(
        z=data.values,
        x=data.columns,
        y=data.index,
        colorscale='RdBu'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Features',
        yaxis_title='Features'
    )
    
    return fig 