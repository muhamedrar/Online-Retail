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
from statsmodels.tsa.stattools import adfuller

config = configparser.ConfigParser()
config.read('../config.ini')

rolling_window = int(config['DataPreprocessing']['rolling_window'])
decomposition_period = int(config['DataPreprocessing']['decomposition_period'])
data_clustered_path = config['KmeansClustering']['data_export_path']

order = tuple(map(int, config['ArimaForecasting']['order'].strip('()').split(',')))
seasonal_order = ast.literal_eval(config['ArimaForecasting']['seasonal_order'])

def check_stationarity(series):
    result = adfuller(series.dropna())
    print(f"ADF Statistic: {result[0]}, p-value: {result[1]}")
    if result[1] >= 0.05:
        print("Data is not stationary. Applying first difference.")
        differenced_series = series.diff().dropna()
        result = adfuller(differenced_series)
        print(f"After differencing - ADF Statistic: {result[0]}, p-value: {result[1]}")
        return differenced_series, 1
    return series, 0

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

def prep_data_for_sales_forecasting(file_path, cluster=None, Country=None, rolling_window=rolling_window, decomposition_period=decomposition_period):
    print(f"Executing prep_data_for_sales_forecasting with file_path: {file_path}, rolling_window: {rolling_window}, decomposition_period: {decomposition_period}")

    if cluster is not None:
        if Country is not None:
            print("Loading data with clustering enabled.")
            df = load_data(file_path, with_cluster=True)
            print(f"Data loaded with shape: {df.shape}")
            df = filter_data_by_country(df, Country)
            df = filter_data_by_cluster(df, cluster)
        else:
            print("Loading data with clustering enabled but no country filter.")
            df = load_data(file_path, with_cluster=True)
            df = filter_data_by_cluster(df, cluster)
    else:
        if Country is not None:
            print("Loading data with clustering disabled.")
            df = load_data(file_path, with_cluster=False)
            df = filter_data_by_country(df, Country)
        else:
            print("Loading data with clustering disabled and no country filter.")
            df = load_data(file_path, with_cluster=False)

    print(f"Before preprocessing, data shape: {df.shape}")
    df = preprocess_data(df)
    df = feature_engineering(df)
    df = remove_outliers(df, 'Quantity')
    df = remove_outliers(df, 'UnitPrice')
    print(f"After outlier removal and feature engineering, data shape: {df.shape}")

    # Aggregate and smooth data
    df = df.groupby(df['InvoiceDate'].dt.date)['TotalPrice'].sum().to_frame()
    df['TotalPrice'] = df['TotalPrice'].rolling(window=rolling_window, min_periods=1).mean()
    df.dropna(inplace=True)
    print(f"After aggregation and smoothing, data shape: {df.shape}")

    # Seasonal decomposition
    decomposition_smooth = seasonal_decompose(df, model='additive', period=decomposition_period)
    trend = decomposition_smooth.trend
    seasonal = decomposition_smooth.seasonal
    resid = decomposition_smooth.resid

    # Align all components to common index
    common_index = df.index.intersection(trend.index).intersection(seasonal.index).intersection(resid.index)
    actual_data = df.loc[common_index].copy()
    trend = trend.loc[common_index].copy()
    seasonal = seasonal.loc[common_index].copy()
    resid = resid.loc[common_index].copy()
    print(f"After alignment, data shape: {actual_data.shape}, trend shape: {trend.shape}, seasonal shape: {seasonal.shape}, resid shape: {resid.shape}")

    return actual_data, trend, seasonal, resid

class SalesForecaster:
    def __init__(self, file_path, order=order, seasonal_order=seasonal_order, cluster=None, Country=None):
        self.df, self.trend, self.seasonal, self.resid = prep_data_for_sales_forecasting(file_path, cluster=cluster, Country=Country)
        self.trend = self.trend.dropna()
        self.seasonal = self.seasonal.dropna()
        self.resid = self.resid.dropna()
        self.model = ARIMA(self.resid, order=order, seasonal_order=seasonal_order)
        self.fitted_model = self.model.fit()

    def forecast(self, future_steps=30):
        future_forecast_resid = self.fitted_model.forecast(steps=future_steps)
        print(f"future_forecast_resid length: {len(future_forecast_resid)}")

        # Ensure future_trend and future_seasonal match future_steps
        trend_length = min(future_steps, len(self.trend))
        seasonal_length = min(future_steps, len(self.seasonal))
        future_trend = pd.Series(self.trend.iloc[-trend_length:].values, index=range(future_steps))
        future_seasonal = pd.Series(self.seasonal.iloc[-seasonal_length:].values, index=range(future_steps))

        # Pad with last values to match future_steps
        if len(future_trend) < future_steps:
            last_trend = future_trend.iloc[-1] if not future_trend.empty else 0
            future_trend = pd.concat([future_trend, pd.Series([last_trend] * (future_steps - len(future_trend)), index=range(len(future_trend), future_steps))])
        if len(future_seasonal) < future_steps:
            last_seasonal = future_seasonal.iloc[-1] if not future_seasonal.empty else 0
            future_seasonal = pd.concat([future_seasonal, pd.Series([last_seasonal] * (future_steps - len(future_seasonal)), index=range(len(future_seasonal), future_steps))])

        print(f"future_trend length: {len(future_trend)}, future_seasonal length: {len(future_seasonal)}")

        # Combine components
        future_final_forecast = future_forecast_resid.values + future_trend.values[:future_steps] + future_seasonal.values[:future_steps]
        future_dates = pd.date_range(start=self.df.index[-1] + pd.Timedelta(days=1), periods=future_steps, freq='D')
        return pd.Series(future_final_forecast, index=future_dates)

    def get_metrics(self):
        # Generate in-sample predictions for the residual series
        resid_predictions = self.fitted_model.predict(start=self.resid.index[0], end=self.resid.index[-1])
        print(f"resid_predictions length: {len(resid_predictions)}, index: {resid_predictions.index}")

        # Align trend and seasonal components with resid_predictions index
        trend_aligned = self.trend.reindex(resid_predictions.index, method='ffill').values
        seasonal_aligned = self.seasonal.reindex(resid_predictions.index, method='ffill').values
        full_predictions = resid_predictions + trend_aligned + seasonal_aligned

        # Get actual data for the same index range
        actual_data = self.df.loc[full_predictions.index]
        if len(actual_data) == 0 or len(full_predictions) == 0:
            print("Warning: No valid data for metric calculation.")
            return {'rmse': 0, 'mape': 0, 'r2': 0}

        residuals = actual_data['TotalPrice'] - full_predictions
        rmse = (residuals ** 2).mean() ** 0.5 if not residuals.isna().all() else 0
        mape = (abs(residuals / (actual_data['TotalPrice'] + 1e-10)) * 100).mean() if not residuals.isna().all() else 0
        ss_tot = ((actual_data['TotalPrice'] - actual_data['TotalPrice'].mean()) ** 2).sum()
        ss_res = ((actual_data['TotalPrice'] - full_predictions) ** 2).sum()
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 and not residuals.isna().all() else 0
        return {'rmse': rmse, 'mape': mape, 'r2': r2}