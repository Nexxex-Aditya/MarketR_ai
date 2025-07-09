# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer, KNNImputer
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('seaborn')
sns.set_palette('husl')

# ===== CELL 1: Data Loading and Initial Exploration =====
# Load the datasets
spirits_data = pd.read_csv('../data/us_spirits_purchase_data.csv')
products_data = pd.read_csv('../data/products.csv')
clusters_data = pd.read_csv('../data/clusters.csv')

# Display basic information about each dataset
print("=== Spirits Purchase Data ===")
print(f"Shape: {spirits_data.shape}")
print("\nSample data:")
print(spirits_data.head())

print("\n=== Products Data ===")
print(f"Shape: {products_data.shape}")
print("\nSample data:")
print(products_data.head())

print("\n=== Clusters Data ===")
print(f"Shape: {clusters_data.shape}")
print("\nSample data:")
print(clusters_data.head())

# ===== CELL 2: Missing Data Analysis and Treatment =====
def analyze_missing_data(df, title):
    """Analyze missing values in a dataframe"""
    missing_data = pd.DataFrame({
        'Missing Values': df.isnull().sum(),
        'Percentage': (df.isnull().sum() / len(df)) * 100
    })
    print(f"=== Missing Data Analysis for {title} ===")
    print(missing_data[missing_data['Missing Values'] > 0])
    
    # Visualize missing data
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
    plt.title(f'Missing Data Heatmap - {title}')
    plt.show()
    
    return missing_data

def handle_missing_data(df, missing_analysis):
    """
    Handle missing data based on column types and missing percentages
    """
    df_cleaned = df.copy()
    
    for column in df.columns:
        missing_pct = missing_analysis.loc[column, 'Percentage']
        
        # Skip if no missing values
        if missing_pct == 0:
            continue
            
        # Get column data type
        dtype = df[column].dtype
        
        # Handle different types of missing data
        if missing_pct > 30:  # High percentage of missing values
            if dtype in ['int64', 'float64']:
                # For numerical columns with high missing values, use median
                df_cleaned[column] = df_cleaned[column].fillna(df_cleaned[column].median())
            else:
                # For categorical columns with high missing values, use mode
                df_cleaned[column] = df_cleaned[column].fillna(df_cleaned[column].mode()[0])
                
        elif missing_pct > 5:  # Medium percentage of missing values
            if dtype in ['int64', 'float64']:
                # Use KNN imputation for numerical columns
                imputer = KNNImputer(n_neighbors=5)
                df_cleaned[column] = imputer.fit_transform(df_cleaned[[column]])
            else:
                # Use forward fill for categorical columns
                df_cleaned[column] = df_cleaned[column].fillna(method='ffill')
                
        else:  # Low percentage of missing values
            if dtype in ['int64', 'float64']:
                # Use mean for numerical columns
                df_cleaned[column] = df_cleaned[column].fillna(df_cleaned[column].mean())
            else:
                # Use backward fill for categorical columns
                df_cleaned[column] = df_cleaned[column].fillna(method='bfill')
    
    return df_cleaned

def handle_time_series_missing_data(df, date_column, value_column):
    """
    Handle missing data in time series
    """
    # Sort by date
    df_sorted = df.sort_values(date_column)
    
    # Forward fill for short gaps
    df_filled = df_sorted.fillna(method='ffill', limit=3)
    
    # Backward fill for remaining gaps
    df_filled = df_filled.fillna(method='bfill', limit=3)
    
    # For remaining gaps, use interpolation
    df_filled[value_column] = df_filled[value_column].interpolate(method='time')
    
    return df_filled

# Analyze missing data in each dataset
spirits_missing = analyze_missing_data(spirits_data, 'Spirits Purchase Data')
products_missing = analyze_missing_data(products_data, 'Products Data')
clusters_missing = analyze_missing_data(clusters_data, 'Clusters Data')

# Handle missing data in each dataset
spirits_data_cleaned = handle_missing_data(spirits_data, spirits_missing)
products_data_cleaned = handle_missing_data(products_data, products_missing)
clusters_data_cleaned = handle_missing_data(clusters_data, clusters_missing)

# Handle time series missing data in spirits data
if 'purchase_date' in spirits_data_cleaned.columns and 'sales_amount' in spirits_data_cleaned.columns:
    spirits_data_cleaned = handle_time_series_missing_data(
        spirits_data_cleaned, 
        'purchase_date', 
        'sales_amount'
    )

# Verify missing data handling
print("\n=== Missing Data After Treatment ===")
print("\nSpirits Data:")
print(spirits_data_cleaned.isnull().sum())
print("\nProducts Data:")
print(products_data_cleaned.isnull().sum())
print("\nClusters Data:")
print(clusters_data_cleaned.isnull().sum())

# ===== CELL 3: Data Integration and Feature Engineering =====
# Merge datasets using cleaned data
merged_data = pd.merge(spirits_data_cleaned, products_data_cleaned, on='product_id', how='left')
merged_data = pd.merge(merged_data, clusters_data_cleaned, on='cluster_id', how='left')

# Convert date columns to datetime
date_columns = [col for col in merged_data.columns if 'date' in col.lower()]
for col in date_columns:
    merged_data[col] = pd.to_datetime(merged_data[col])

# Create time-based features
merged_data['year'] = merged_data['purchase_date'].dt.year
merged_data['month'] = merged_data['purchase_date'].dt.month
merged_data['quarter'] = merged_data['purchase_date'].dt.quarter

# Display the merged dataset
print("=== Merged Dataset ===")
print(f"Shape: {merged_data.shape}")
print(merged_data.head())

# ===== CELL 4: Time Series Analysis and Anomaly Detection =====
def detect_anomalies(series, window=30, threshold=3):
    """Detect anomalies using rolling statistics"""
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    
    upper_bound = rolling_mean + (threshold * rolling_std)
    lower_bound = rolling_mean - (threshold * rolling_std)
    
    anomalies = series[(series > upper_bound) | (series < lower_bound)]
    return anomalies, upper_bound, lower_bound

# Aggregate data by date
daily_sales = merged_data.groupby('purchase_date')['sales_amount'].sum().reset_index()

# Detect anomalies
anomalies, upper_bound, lower_bound = detect_anomalies(daily_sales['sales_amount'])

# Visualize time series with anomalies
plt.figure(figsize=(15, 7))
plt.plot(daily_sales['purchase_date'], daily_sales['sales_amount'], label='Daily Sales')
plt.scatter(anomalies.index, anomalies.values, color='red', label='Anomalies')
plt.plot(daily_sales['purchase_date'], upper_bound, '--', color='gray', label='Upper Bound')
plt.plot(daily_sales['purchase_date'], lower_bound, '--', color='gray', label='Lower Bound')
plt.title('Daily Sales with Anomalies')
plt.xlabel('Date')
plt.ylabel('Sales Amount')
plt.legend()
plt.show()

# ===== CELL 5: Forecasting with Prophet =====
# Prepare data for Prophet
prophet_data = daily_sales.rename(columns={'purchase_date': 'ds', 'sales_amount': 'y'})

# Initialize and fit Prophet model
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    changepoint_prior_scale=0.05
)
model.fit(prophet_data)

# Make future predictions
future = model.make_future_dataframe(periods=90)  # Forecast 90 days ahead
forecast = model.predict(future)

# Plot the forecast
fig = model.plot(forecast)
plt.title('Sales Forecast')
plt.show()

# Plot forecast components
fig = model.plot_components(forecast)
plt.show()

# ===== CELL 6: Cluster Analysis and Market Segmentation =====
# Analyze cluster performance
cluster_analysis = merged_data.groupby('cluster_id').agg({
    'sales_amount': ['sum', 'mean', 'count'],
    'product_id': 'nunique'
}).reset_index()

# Visualize cluster performance
plt.figure(figsize=(12, 6))
sns.barplot(data=cluster_analysis, x='cluster_id', y=('sales_amount', 'sum'))
plt.title('Total Sales by Cluster')
plt.xlabel('Cluster ID')
plt.ylabel('Total Sales')
plt.show()

# Product category analysis
category_analysis = merged_data.groupby('category').agg({
    'sales_amount': ['sum', 'mean'],
    'product_id': 'nunique'
}).reset_index()

# Visualize category performance
plt.figure(figsize=(12, 6))
sns.barplot(data=category_analysis, x='category', y=('sales_amount', 'sum'))
plt.title('Total Sales by Category')
plt.xticks(rotation=45)
plt.show()

# ===== CELL 7: Additional Analysis - Price Trends =====
# Analyze price trends over time
price_trends = merged_data.groupby(['purchase_date', 'category'])['price'].mean().reset_index()

# Plot price trends by category
plt.figure(figsize=(15, 7))
for category in price_trends['category'].unique():
    category_data = price_trends[price_trends['category'] == category]
    plt.plot(category_data['purchase_date'], category_data['price'], label=category)

plt.title('Price Trends by Category')
plt.xlabel('Date')
plt.ylabel('Average Price')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# ===== CELL 8: Additional Analysis - Sales Volume Distribution =====
# Analyze sales volume distribution
plt.figure(figsize=(12, 6))
sns.histplot(data=merged_data, x='sales_amount', bins=50)
plt.title('Distribution of Sales Amount')
plt.xlabel('Sales Amount')
plt.ylabel('Frequency')
plt.show()

# ===== CELL 9: Additional Analysis - Correlation Analysis =====
# Select numeric columns for correlation analysis
numeric_cols = merged_data.select_dtypes(include=[np.number]).columns
correlation_matrix = merged_data[numeric_cols].corr()

# Plot correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# ===== CELL 10: Additional Analysis - Seasonal Patterns =====
# Analyze seasonal patterns
monthly_sales = merged_data.groupby(['year', 'month'])['sales_amount'].sum().reset_index()
monthly_sales['date'] = pd.to_datetime(monthly_sales[['year', 'month']].assign(day=1))

# Plot monthly sales trends
plt.figure(figsize=(15, 7))
plt.plot(monthly_sales['date'], monthly_sales['sales_amount'], marker='o')
plt.title('Monthly Sales Trends')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.grid(True)
plt.show()

# ===== CELL 11: Additional Analysis - Product Performance =====
# Analyze top performing products
top_products = merged_data.groupby('product_id').agg({
    'sales_amount': 'sum',
    'product_name': 'first'
}).sort_values('sales_amount', ascending=False).head(10)

# Plot top products
plt.figure(figsize=(12, 6))
sns.barplot(data=top_products.reset_index(), x='product_name', y='sales_amount')
plt.title('Top 10 Products by Sales')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# ===== CELL 12: Additional Analysis - Geographic Analysis =====
# If location data is available
if 'location' in merged_data.columns:
    location_analysis = merged_data.groupby('location').agg({
        'sales_amount': ['sum', 'mean'],
        'product_id': 'nunique'
    }).reset_index()
    
    # Plot location performance
    plt.figure(figsize=(12, 6))
    sns.barplot(data=location_analysis, x='location', y=('sales_amount', 'sum'))
    plt.title('Sales by Location')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show() 