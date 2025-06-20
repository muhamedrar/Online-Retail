import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
import configparser
import os
import sys

# Add src directory to system path for custom modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
src_path = os.path.join(project_root, 'src')
print(f"Appending path to src: {src_path}")
sys.path.append(src_path)

from sales_forcasting import SalesForecaster
from data_preprocessing import load_data, preprocess_data, feature_engineering, remove_outliers

def prep_historical_data(df, rolling_window=7):
    """Prepare historical data by preprocessing, engineering features, removing outliers, and smoothing."""
    df = preprocess_data(df)
    df = feature_engineering(df)
    df = remove_outliers(df, 'Quantity')
    df = remove_outliers(df, 'UnitPrice')
    df = df.groupby(df['InvoiceDate'].dt.date)['TotalPrice'].sum().to_frame()
    df['TotalPrice'] = df['TotalPrice'].rolling(window=rolling_window).mean()
    df.dropna(inplace=True)
    return df.reset_index()

def show_forecasting_page():
    """Display the sales forecasting page with interactive plots and metrics."""
    st.title("📈 Sales Forecasting")

    # Load and prepare data
    data_file = './Data/Online_Retail_Clustered.csv'
    df = load_data(data_file, with_cluster=True)
    historical_df = prep_historical_data(df)

    # Load cluster labels from config
    config = configparser.ConfigParser()
    config.read('./config.ini')
    cluster_labels = [label.strip() for label in config['KmeansClustering']['cluster_labels'].split(',')]

    # User inputs
    cluster = st.selectbox("Choose a Cluster", ["All Clusters"] + cluster_labels)

    # Determine model file based on selection
    if cluster == "All Clusters":
        model_key = "sales_forecaster_global"
    else:
        model_key = f"sales_forecaster_cluster_{cluster.replace(' ', '_')}"
    model_path = f"./Models/{model_key}.pkl"

    # Load and forecast
    try:
        if not os.path.exists(model_path):
            st.error(f"No model available for {cluster}. Please train the model first.")
        else:
            with open(model_path, 'rb') as f:
                forecaster = pickle.load(f)

            # Generate forecast for 293 days
            forecast_values = forecaster.forecast(future_steps=290)
            print(f"Forecast values length: {len(forecast_values)}, first 5: {forecast_values.head()}")  # Debug print
            if isinstance(forecast_values, pd.Series):
                forecast_df = pd.DataFrame({
                    'date': forecast_values.index,
                    'Sales': forecast_values.values
                })
            else:
                last_date = historical_df['InvoiceDate'].max()
                forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(forecast_values), freq='D')
                forecast_df = pd.DataFrame({
                    'date': forecast_dates,
                    'Sales': forecast_values
                })

            # Display forecast plot
            st.subheader(f"Forecast for {cluster}")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=forecast_df['date'],
                y=forecast_df['Sales'],
                mode='lines',
                name='Forecast',
                line=dict(color='#4CAF50')
            ))
            fig.update_layout(
                title=f"Sales Forecast for {cluster}",
                xaxis_title="Date",
                yaxis_title="Sales",
                template="plotly_white"
            )
            st.plotly_chart(fig)

            # Display metrics
            if hasattr(forecaster, 'get_metrics'):
                metrics = forecaster.get_metrics()
                col1, col2, col3= st.columns(3)
                with col1:
                    st.metric("RMSE", f"{metrics.get('rmse', 0):.2f}")
                with col2:
                    st.metric("MAPE", f"{metrics.get('mape', 0):.2f}%")
                with col3:
                    st.metric("R2", f"{metrics.get('r2', 0):.2f}%")
                 
            else:
                st.write("No evaluation metrics available.")

    except Exception as e:
        st.error(f"Error generating forecast: {str(e)}")

    # Display historical data
    try:
        st.subheader("Historical Sales")
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(
            x=historical_df['InvoiceDate'],
            y=historical_df['TotalPrice'],
            mode='lines',
            name='Historical Sales',
            line=dict(color='#2196F3')
        ))
        fig_hist.update_layout(
            title=f"Historical Sales for {cluster}",
            xaxis_title="Date",
            yaxis_title="Sales",
            template="plotly_white"
        )
        st.plotly_chart(fig_hist)
    except Exception as e:
        st.warning(f"Could not display historical data: {str(e)}")
