import pandas as pd
import logging
from typing import Dict, Any, Optional
from utils.llm_utils import call_llm
import json

logger = logging.getLogger(__name__)

class DataHandler:
    """Handles data loading, preprocessing, and analysis with LLM guidance."""
    
    def __init__(self, config: dict, llm_config: dict = None):
        """Initialize DataHandler with configuration."""
        self.config = config
        self.llm_config = llm_config
        self.logger = logging.getLogger(__name__)
        
        # Initialize LLM connection if config provided
        if self.llm_config:
            self.llm = self._initialize_llm()
        else:
            self.llm = None
        
    def _initialize_llm(self):
        """Initialize LLM connection."""
        try:
            from utils.llm_utils import initialize_llm
            return initialize_llm(self.llm_config)
        except Exception as e:
            self.logger.warning(f"Failed to initialize LLM: {str(e)}")
            return None
        
    def load_purchase_data(self) -> pd.DataFrame:
        """Load and preprocess purchase data."""
        try:
            # Load data
            data_path = self.config['data']['purchase_data']
            self.logger.info(f"Loading purchase data from {data_path}")
            
            # Read only necessary columns and optimize dtypes
            data = pd.read_csv(data_path, 
                             usecols=['DateKey', 'ClusterId', 'SingleBuyProductItemId', 
                                    'PurchaseValue', 'PurchaseVolume', 'NumOutletsPurchasedw'],
                             dtype={
                                 'ClusterId': 'int32',
                                 'SingleBuyProductItemId': 'int32',
                                 'PurchaseValue': 'float32',
                                 'PurchaseVolume': 'float32',
                                 'NumOutletsPurchasedw': 'int32'
                             })
            
            # Rename columns to match expected format
            column_mapping = {
                'DateKey': 'date',
                'PurchaseValue': 'sales_value',
                'PurchaseVolume': 'sales_volume',
                'NumOutletsPurchasedw': 'num_outlets'
            }
            data = data.rename(columns=column_mapping)
            
            # Convert date column to datetime
            data['date'] = pd.to_datetime(data['date'])
            data.set_index('date', inplace=True)
            
            # Handle missing values and anomalies in a single pass
            data = self._preprocess_data(data)
            
            self.logger.info("Purchase data loaded and preprocessed successfully.")
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading purchase data: {str(e)}")
            raise
            
    def load_products_data(self) -> pd.DataFrame:
        """Load products data."""
        try:
            data_path = self.config['data']['products_data']
            self.logger.info(f"Loading products data from {data_path}")
            data = pd.read_csv(data_path)
            self.logger.info("Products data loaded successfully.")
            return data
        except Exception as e:
            self.logger.error(f"Error loading products data: {str(e)}")
            raise
            
    def load_clusters_data(self) -> pd.DataFrame:
        """Load clusters data."""
        try:
            data_path = self.config['data']['clusters_data']
            self.logger.info(f"Loading clusters data from {data_path}")
            data = pd.read_csv(data_path)
            self.logger.info("Clusters data loaded successfully.")
            return data
        except Exception as e:
            self.logger.error(f"Error loading clusters data: {str(e)}")
            raise
            
    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Efficiently preprocess data in a single pass."""
        # Handle missing values
        data['sales_value'] = data['sales_value'].fillna(0)
        data['sales_volume'] = data['sales_volume'].fillna(0)
        
        # Get anomaly detection settings with defaults
        threshold = self.config.get('anomaly_detection', {}).get('threshold', 2.0)
        window_size = self.config.get('anomaly_detection', {}).get('window_size', 7)
        
        # Handle anomalies using vectorized operations
        rolling_mean = data['sales_value'].rolling(window=window_size, min_periods=1).mean()
        rolling_std = data['sales_value'].rolling(window=window_size, min_periods=1).std()
        
        # Identify anomalies
        z_scores = (data['sales_value'] - rolling_mean) / rolling_std
        anomalies = abs(z_scores) > threshold
        
        # Replace anomalies with rolling mean
        data.loc[anomalies, 'sales_value'] = rolling_mean[anomalies]
        
        # Add basic features without creating multiple copies
        data['month'] = data.index.month
        data['day_of_week'] = data.index.dayofweek
        data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
        
        return data
        
    def analyze_data(self, data: pd.DataFrame) -> dict:
        """Analyze data and return key metrics."""
        try:
            # Calculate basic statistics
            analysis = {
                'total_sales': data['sales_value'].sum(),
                'total_volume': data['sales_volume'].sum(),
                'avg_sales': data['sales_value'].mean(),
                'avg_volume': data['sales_volume'].mean(),
                'num_products': data['SingleBuyProductItemId'].nunique(),
                'num_clusters': data['ClusterId'].nunique(),
                'avg_outlets': data['num_outlets'].mean()
            }
            
            # Calculate trends
            monthly_sales = data.groupby(data.index.month)['sales_value'].mean()
            monthly_volume = data.groupby(data.index.month)['sales_volume'].mean()
            
            analysis['sales_trend'] = "Increasing" if monthly_sales.iloc[-1] > monthly_sales.iloc[0] else "Decreasing"
            analysis['volume_trend'] = "Increasing" if monthly_volume.iloc[-1] > monthly_volume.iloc[0] else "Decreasing"
            
            # Calculate cluster statistics
            cluster_stats = data.groupby('ClusterId').agg({
                'sales_value': 'sum',
                'sales_volume': 'sum',
                'num_outlets': 'mean'
            }).round(2)
            
            analysis['cluster_stats'] = cluster_stats.to_dict()
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing data: {str(e)}")
            raise

    def llm_analyze_data_and_suggest_treatment(self, df: pd.DataFrame, data_context: str) -> Dict[str, Any]:
        """Uses LLM to analyze data and suggest treatment for null values and feature engineering."""
        df_info = df.info(verbose=True, show_counts=True, buf=None)
        df_describe = df.describe().to_string()
        null_counts = df.isnull().sum()
        null_percentage = (df.isnull().sum() / len(df)) * 100
        null_summary = pd.DataFrame({'Null Count': null_counts, 'Null Percentage': null_percentage}).to_string()

        prompt = f"""
You are an expert data engineer and analyst. I am providing you with information about a Pandas DataFrame. Your task is to analyze this data and provide recommendations for:
1.  **Null Value Treatment:** For each column with null values, suggest the best imputation strategy (e.g., mean, median, mode, forward-fill, backward-fill, specific value, or drop rows/columns) and explain why.
2.  **Feature Engineering:** Based on the existing columns and the context, suggest new features that could be engineered to improve forecasting models. Provide the logic or type of feature (e.g., time-based features, lag features, rolling statistics).

Here's the DataFrame information:
-   `df.info()` output (showing non-null counts and dtypes):
    {df_info}
-   `df.describe()` output (descriptive statistics):
    {df_describe}
-   Null value summary:
    {null_summary}

Data Context: {data_context}

Provide your recommendations as a JSON object with two main keys: 'null_treatment_suggestions' (a dictionary where keys are column names and values are dictionaries with 'strategy' and 'reasoning') and 'feature_engineering_suggestions' (a list of dictionaries, each with 'feature_name', 'type', and 'logic').
Example for null_treatment_suggestions:
{{"ColumnA": {{"strategy": "mean_imputation", "reasoning": "Numerical, normally distributed."}}, "ColumnB": {{"strategy": "mode_imputation", "reasoning": "Categorical data."}}}}
Example for feature_engineering_suggestions:
[{{"feature_name": "day_of_week", "type": "time_based", "logic": "df['date'].dt.dayofweek"}}]
"""
        try:
            response = call_llm(prompt, config=self.config)
            return json.loads(response)
        except json.JSONDecodeError as e:
            self.logger.error(f"LLM response not valid JSON for data analysis: {e}. Response: {response[:500]}...")
            return {}
        except Exception as e:
            self.logger.error(f"Error calling LLM for data analysis: {e}")
            return {}

    def apply_null_treatment(self, df: pd.DataFrame, suggestions: Dict[str, Any]) -> pd.DataFrame:
        """Applies null treatment strategies based on LLM suggestions."""
        if 'null_treatment_suggestions' not in suggestions:
            return df

        for col, details in suggestions['null_treatment_suggestions'].items():
            strategy = details.get('strategy')
            if col in df.columns:
                if strategy == 'mean_imputation' and pd.api.types.is_numeric_dtype(df[col]):
                    df[col].fillna(df[col].mean(), inplace=True)
                    self.logger.info(f"Applied mean imputation to column: {col}")
                elif strategy == 'median_imputation' and pd.api.types.is_numeric_dtype(df[col]):
                    df[col].fillna(df[col].median(), inplace=True)
                    self.logger.info(f"Applied median imputation to column: {col}")
                elif strategy == 'mode_imputation':
                    df[col].fillna(df[col].mode()[0], inplace=True)
                    self.logger.info(f"Applied mode imputation to column: {col}")
                elif strategy == 'ffill':
                    df[col].fillna(method='ffill', inplace=True)
                    self.logger.info(f"Applied forward-fill to column: {col}")
                elif strategy == 'bfill':
                    df[col].fillna(method='bfill', inplace=True)
                    self.logger.info(f"Applied backward-fill to column: {col}")
                elif strategy == 'drop_rows':
                    df.dropna(subset=[col], inplace=True)
                    self.logger.info(f"Dropped rows with nulls in column: {col}")
                elif strategy == 'drop_column':
                    df.drop(columns=[col], inplace=True)
                    self.logger.info(f"Dropped column: {col} due to nulls.")
                elif strategy == 'fill_value':
                    fill_val = details.get('value', 0) # Default to 0 if not specified
                    df[col].fillna(fill_val, inplace=True)
                    self.logger.info(f"Filled nulls in column: {col} with value: {fill_val}")
                else:
                    self.logger.warning(f"Unknown or unsupported null treatment strategy '{strategy}' for column: {col}. Skipping.")
            else:
                self.logger.warning(f"Column {col} not found for null treatment. Skipping.")
        return df

    def perform_feature_engineering(self, df: pd.DataFrame, suggestions: Dict[str, Any]) -> pd.DataFrame:
        """Performs feature engineering based on LLM suggestions."""
        if 'feature_engineering_suggestions' not in suggestions:
            return df

        for feature in suggestions['feature_engineering_suggestions']:
            feature_name = feature.get('feature_name')
            feature_type = feature.get('type')
            feature_logic = feature.get('logic')

            if not feature_name or not feature_type or not feature_logic:
                self.logger.warning(f"Invalid feature engineering suggestion: {feature}. Skipping.")
                continue

            try:
                if feature_type == 'time_based' and 'date' in df.columns and 'dt.' in feature_logic:
                    # Basic check for time-based features, assumes 'date' column
                    df[feature_name] = eval(f"df['date'].{feature_logic}")
                    self.logger.info(f"Created time-based feature: {feature_name}")
                elif feature_type == 'lag_feature' and 'shift(' in feature_logic:
                    # Example: df['sales'].shift(1)
                    # Needs to be careful with eval here for security; for demo it's fine
                    # In a real app, parse this more safely
                    base_col = feature_logic.split('.')[0].replace("df['", "").replace("']", "")
                    if base_col in df.columns:
                        df[feature_name] = eval(f"df['{base_col}'].{feature_logic}")
                        self.logger.info(f"Created lag feature: {feature_name}")
                elif feature_type == 'rolling_statistics' and 'rolling(' in feature_logic:
                    # Example: df['sales'].rolling(window=7).mean()
                    base_col = feature_logic.split('.')[0].replace("df['", "").replace("']", "")
                    if base_col in df.columns:
                        df[feature_name] = eval(f"df['{base_col}'].{feature_logic}")
                        self.logger.info(f"Created rolling feature: {feature_name}")
                else:
                    self.logger.warning(f"Unsupported feature type or logic for {feature_name}: {feature_type}, {feature_logic}. Skipping.")
            except Exception as e:
                self.logger.error(f"Error engineering feature {feature_name}: {e}. Skipping.")
        return df

    def detect_and_treat_anomalies(self, df: pd.DataFrame, target_column: str, method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        """Detects and treats anomalies in a specified target column."""
        if target_column not in df.columns:
            self.logger.warning(f"Target column '{target_column}' not found for anomaly detection. Skipping.")
            return df
        if not pd.api.types.is_numeric_dtype(df[target_column]):
            self.logger.warning(f"Target column '{target_column}' is not numeric for anomaly detection. Skipping.")
            return df

        df_cleaned = df.copy()

        if method == 'iqr':
            Q1 = df_cleaned[target_column].quantile(0.25)
            Q3 = df_cleaned[target_column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            anomalies = df_cleaned[(df_cleaned[target_column] < lower_bound) | (df_cleaned[target_column] > upper_bound)]
            if not anomalies.empty:
                self.logger.warning(f"Detected {len(anomalies)} anomalies in {target_column} using IQR method. Lower bound: {lower_bound:.2f}, Upper bound: {upper_bound:.2f}.")
                # Treatment: Replace anomalies with median of the column (or other strategy)
                median_val = df_cleaned[target_column].median()
                df_cleaned.loc[anomalies.index, target_column] = median_val
                self.logger.info(f"Treated anomalies in {target_column} by replacing with median: {median_val:.2f}")
            else:
                self.logger.info(f"No anomalies detected in {target_column} using IQR method.")
        elif method == 'isolation_forest':
            try:
                from sklearn.ensemble import IsolationForest
                model = IsolationForest(contamination='auto', random_state=42)
                df_cleaned['anomaly'] = model.fit_predict(df_cleaned[[target_column]])
                anomalies = df_cleaned[df_cleaned['anomaly'] == -1]

                if not anomalies.empty:
                    self.logger.warning(f"Detected {len(anomalies)} anomalies in {target_column} using Isolation Forest.")
                    median_val = df_cleaned[target_column].median()
                    df_cleaned.loc[anomalies.index, target_column] = median_val
                    self.logger.info(f"Treated anomalies in {target_column} by replacing with median: {median_val:.2f}")
                else:
                    self.logger.info(f"No anomalies detected in {target_column} using Isolation Forest.")
                df_cleaned.drop(columns=['anomaly'], inplace=True)
            except ImportError:
                self.logger.error("sklearn.ensemble.IsolationForest not found. Please install scikit-learn.")
            except Exception as e:
                self.logger.error(f"Error during Isolation Forest anomaly detection for {target_column}: {e}")
        else:
            self.logger.warning(f"Unsupported anomaly detection method: {method}. Skipping.")
        
        return df_cleaned

    def load_and_preprocess_purchase_data(self) -> pd.DataFrame:
        """Load and preprocess the main purchase data, applying LLM-guided treatments and feature engineering."""
        file_path = self.config['data']['purchase_data']
        self.logger.info(f"Loading purchase data from {file_path}")
        try:
            df = pd.read_csv(file_path)
            
            # Standardize column names
            column_mapping = {
                'DateKey': 'date',
                'PurchaseValue': 'sales_value',
                'PurchaseVolume': 'sales_volume'
            }
            df = df.rename(columns=column_mapping)
            
            # Convert 'date' to datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # LLM-guided data analysis and suggestions
            data_context = "This is spirits purchase data. 'sales_value' is the monetary value of purchases, and 'sales_volume' is the quantity purchased. 'ClusterId' and 'SingleBuyProductItemId' are identifiers. The goal is to forecast sales."
            llm_suggestions = self.llm_analyze_data_and_suggest_treatment(df.head(50), data_context) # Pass a sample to LLM to save tokens
            
            # Apply null treatment based on LLM suggestions
            df = self.apply_null_treatment(df, llm_suggestions)

            # Perform feature engineering based on LLM suggestions
            df = self.perform_feature_engineering(df, llm_suggestions)

            # Anomaly detection and treatment (can be made LLM-guided later)
            # For now, applying to 'sales_value' and 'sales_volume' with IQR method as a default
            df = self.detect_and_treat_anomalies(df, target_column='sales_value', method='iqr')
            df = self.detect_and_treat_anomalies(df, target_column='sales_volume', method='iqr')

            self.logger.info("Purchase data loaded and preprocessed successfully with LLM guidance.")
            return df
        except Exception as e:
            self.logger.error(f"Error loading or preprocessing purchase data: {e}")
            raise 