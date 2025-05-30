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

def prep_data_for_sales_forecasting(file_path, cluster=None,  rolling_window=7, decomposition_period=7):

    print(f"executing prep_data_for_sales_forecasting with file_path: {file_path}, cluster: {cluster}, rolling_window: {rolling_window}, decomposition_period: {decomposition_period}")
   
    if cluster is not None:
        print("Loading data with clustering enabled.")
          

        df = load_data(file_path, with_cluster=True)
        print(f"Data loaded with shape: {df.shape}")
        df = df[df['cluster'] == cluster].copy
        print('filtered data')
        
        
    else:
        print("Loading data with clustering disabled.")
        df = load_data(file_path, with_cluster=False)
        
 
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
    def __init__(self, filePath, order=(0, 1, 1), seasonal_order=(2, 0, [1, 2], 7),cluster=None):
        trend, seasonal, resid = prep_data_for_sales_forecasting(filePath, cluster=cluster)
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
    




######################## final ############################
# forecaster = SalesForecaster('./Data/Online_Retail.csv')

# # export model
# with open('./Model/sales_forecaster.pkl', 'wb') as f:
#     pickle.dump(forecaster, f)




forecaster = SalesForecaster('./Data/Online_Retail_Clustered.csv',cluster=True)

