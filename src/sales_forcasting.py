from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
from src.data_preprocessing import load_data, preprocess_data, feature_engineering, remove_outliers


def prep_data_for_sales_forecasting(file_path, cluster=None, outlier_columns=None, rolling_window=7, decomposition_period=7):

    
    # Default columns for outlier removal
    if outlier_columns is None:
        outlier_columns = ['TotalPrice', 'Quantity', 'UnitPrice']

    # Load data
    df = load_data(file_path, with_cluster=True)
    if df is None:
        print("Failed to load data.")
        return None, None, None

    # Filter by cluster if specified
    if cluster is not None:
        df = df[df['cluster'] == cluster]

    # Preprocess and engineer features
    df = preprocess_data(df)
    df = feature_engineering(df)

    # Remove outliers
    for column in outlier_columns:
        df = remove_outliers(df, column)

    # Aggregate and smooth data
    df = df.groupby(['InvoiceDate']).agg({'TotalPrice': 'sum'})
    df['TotalPrice'] = df['TotalPrice'].rolling(window=rolling_window).mean()
    df.dropna(inplace=True)

    # Seasonal decomposition
    decomposition_smooth = seasonal_decompose(df, model='additive', period=decomposition_period)

    return decomposition_smooth.trend, decomposition_smooth.seasonal, decomposition_smooth.resid







class SalesForecaster:
    def __init__(self, filePath, order=(0, 1, 1), seasonal_order=(2, 0, [1, 2], 7)):
        trend, seasonal, resid = prep_data_for_sales_forecasting(filePath, cluster=None)
        self.trend = trend.dropna()
        self.seasonal = seasonal
        self.resid = resid.dropna()
        self.model = ARIMA(self.resid, order=order, seasonal_order=seasonal_order)
        self.fitted_model = self.model.fit()

    def forecast(self, future_steps=30):
        future_forecast_resid = self.fitted_model.forecast(steps=future_steps)
        future_trend = self.trend[-future_steps:]
        future_seasonal = self.seasonal[-future_steps:]
        future_final_forecast = future_forecast_resid + future_trend.values + future_seasonal.values
        return future_final_forecast 
