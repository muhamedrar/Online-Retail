import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
import configparser
import os
import sys
from sales_forcasting import SalesForecaster
from data_preprocessing import remove_outliers, load_data, preprocess_data, feature_engineering

# Debug path resolution
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
src_path = os.path.join(project_root, 'src')
print(f"Appending path to src: {src_path}")
sys.path.append(src_path)

def prep_historical_data(df, rolling_window):
    df = preprocess_data(df)
    df = feature_engineering(df)
    df = remove_outliers(df, 'Quantity')
    df = remove_outliers(df, 'UnitPrice')
    df = remove_outliers(df, 'UnitPrice')
    df = df.groupby(df['InvoiceDate'].dt.date)['TotalPrice'].sum().to_frame()
    df['TotalPrice'] = df['TotalPrice'].rolling(window=rolling_window).mean()
    df.dropna(inplace=True)
    return df

def show_forecasting_page():
    st.title("📈 Sales Forecasting")
    data_file = './Data/Online_Retail_Clustered.csv'
    df = load_data(data_file, with_cluster=True)
    df = prep_historical_data(df, 7).reset_index()

    # Load cluster labels from config
    config = configparser.ConfigParser()
    config.read('./config.ini')
    cluster_labels = [label.strip() for label in config['KmeansClustering']['cluster_labels'].split(',')]
    countries = ['United Kingdom', 'Germany', 'France']

    # User inputs
    country = st.selectbox("Choose a Country", countries)
    cluster = st.selectbox("Choose a Cluster", cluster_labels + ["All Clusters"])

    # Determine model file based on selections
    if cluster == "All Clusters":
        model_key = f"sales_forecaster_{country.replace(' ', '_')}"
    else:
        model_key = f"sales_forecaster_{cluster.replace(' ', '_')}_{country.replace(' ', '_')}"

    model_path = f"./Models/{model_key}.pkl"

    # Load and forecast
    try:
        if not os.path.exists(model_path):
            st.error(f"No enough data for this combination ({country} - {cluster}). Please train a model for this combination.")
        else:
            with open(model_path, 'rb') as f:
                forecaster = pickle.load(f)

            # Generate forecast (use full forecast from SalesForecaster)
            forecast_values = forecaster.forecast(future_steps = 180)
            if isinstance(forecast_values, pd.Series):
                forecast_df = pd.DataFrame({
                    'date': forecast_values.index,
                    'Sales': forecast_values.values
                })
            else:
                # Assume forecast_values is a list or array with dates implied by model
                last_date = df['InvoiceDate'].max()
                forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(forecast_values), freq='D')
                forecast_df = pd.DataFrame({
                    'date': forecast_dates,
                    'Sales': forecast_values
                })

            print(forecast_df)

            # Display forecast
            st.subheader(f"Forecast for {country} - {cluster}")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=forecast_df['date'],
                y=forecast_df['Sales'],
                mode='lines',
                name='Forecast',
                line=dict(color='#4CAF50')
            ))
            fig.update_layout(
                title=f"Sales Forecast for {country} - {cluster}",
                xaxis_title="Date",
                yaxis_title="Sales",
                template="plotly_white"
            )
            st.plotly_chart(fig)

            # Display metrics (if available)
            if hasattr(forecaster, 'get_metrics'):
                metrics = forecaster.get_metrics()
                st.metric("RMSE", round(metrics.get('rmse', 0), 2))
            else:
                st.write("No evaluation metrics available.")

    except Exception as e:
        st.error(f"Error generating forecast: {str(e)}")

    # Display historical data
    try:
        # Plot historical data
        st.subheader("Historical Sales")
        fig_hist = px.line(df, x='InvoiceDate', y='TotalPrice', title=f"Historical Sales for {country} - {cluster}")
        fig_hist.update_layout(
            xaxis_title="Date",
            yaxis_title="Sales",
            template="plotly_white"
        )
        st.plotly_chart(fig_hist)

    except Exception as e:
        st.warning(f"Could not display historical data: {str(e)}")