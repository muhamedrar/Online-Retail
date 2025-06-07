from statsmodels.tsa.seasonal import seasonal_decompose
import pickle
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_preprocessing import load_data, preprocess_data, feature_engineering, remove_outliers
import warnings
warnings.filterwarnings("ignore")
import configparser
import ast

config = configparser.ConfigParser()
config.read('./config.ini')


rolling_window = int(config['DataPreprocessing']['rolling_window'])
decomposition_period = int(config['DataPreprocessing']['decomposition_period'])
data_clustered_path = config['KmeansClustering']['data_export_path']

order = tuple(map(int, config['ArimaForecasting']['order'].strip('()').split(',')))
# Parse seasonal_order to support (2, 0, [1, 2], 7) format
seasonal_order = ast.literal_eval(config['ArimaForecasting']['seasonal_order'])




def filter_data_by_country(df, country):
    if country is not None:
        print(f"Filtering data for country: {country}")
        df = df[df['Country'] == country].copy()
        print(f"Data filtered for country {country}, new shape: {df.shape}")
    else:
        print("No country filter applied.")
    return df


def filter_data_by_cluster(df, cluster):
    if cluster is not None:
        print(f"Filtering data for cluster: {cluster}")
        df = df[df['cluster'] == cluster].copy()
        print(f"Data filtered for cluster {cluster}, new shape: {df.shape}")
    else:
        print("No cluster filter applied.")
    return df




def prep_data_for_sales_forecasting(file_path, cluster=None,Country = None,  rolling_window=rolling_window, decomposition_period=decomposition_period):

    print(f"executing prep_data_for_sales_forecasting with file_path: {file_path}, rolling_window: {rolling_window}, decomposition_period: {decomposition_period}")
    

    
    if cluster is not None:
        print("Loading data with clustering enabled.")
          
        df = load_data(file_path, with_cluster=True)
        print(f"Data loaded with shape: {df.shape}")
        # df = df[df['cluster'] == cluster].copy()
        df = filter_data_by_country(df, Country)
        print(f"Data filtered by country {Country}, new shape: {df.shape}")
        df = filter_data_by_cluster(df, cluster)
        print(f"Data filtered for cluster {cluster}, new shape: {df.shape}")
    
    else:
        print("Loading data with clustering disabled.")
        df = load_data(file_path, with_cluster=False)
        df = filter_data_by_country(df, Country)
 
    print(f"before preprocessing, data shape: {df.shape}")
    df = preprocess_data(df)
    
    df = feature_engineering(df)

    
    df = remove_outliers(df, 'Quantity')
    df = remove_outliers(df, 'UnitPrice')
    df = remove_outliers(df, 'UnitPrice')
    print(f"removed outliers and feature_engineering and preprocess_data, data shape: {df.shape}")

    # Aggregate and smooth data
    df = df.groupby(df['InvoiceDate'].dt.date)['TotalPrice'].sum().to_frame()
    df['TotalPrice'] = df['TotalPrice'].rolling(window=rolling_window).mean()
    df.dropna(inplace=True) 
    # Seasonal decomposition
    decomposition_smooth = seasonal_decompose(df, model='additive', period=decomposition_period)
    
    print(f"Data prepared for sales forecasting with shape: {df.shape}")
    return decomposition_smooth.trend.dropna(), decomposition_smooth.seasonal.dropna(), decomposition_smooth.resid.dropna()







class SalesForecaster:
    def __init__(self, filePath, order=order, seasonal_order=seasonal_order,cluster=None, Country=None):
        trend, seasonal, resid = prep_data_for_sales_forecasting(filePath, cluster=cluster,Country=Country)
        self.trend = trend.dropna()
        self.seasonal = seasonal.dropna()
        self.resid = resid.dropna()
        self.model = ARIMA(self.resid, order=order, seasonal_order=seasonal_order)
        self.fitted_model = self.model.fit()

    def forecast(self, future_steps=30):
        future_forecast_resid = self.fitted_model.forecast(steps=future_steps)
        future_trend = self.trend[-future_steps:]
        future_seasonal = self.seasonal[-future_steps:]
        future_final_forecast = future_forecast_resid + future_trend.values + future_seasonal.values
        # future dates
        future_dates = pd.date_range(start=self.trend.index[-1] + pd.Timedelta(days=1), periods=future_steps, freq='D')
        future_final_forecast = pd.Series(future_final_forecast.values, index=future_dates)
        return future_final_forecast 
    
    
    
        
        
        

    







