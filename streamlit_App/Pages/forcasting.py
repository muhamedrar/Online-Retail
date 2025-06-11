import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
import configparser
import os
import sys

# Adjust the path to src directory relative to the script location
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

def show_forecasting_page():
    st.title("📈 Sales Forecasting")

    # Load cluster labels from config
    config = configparser.ConfigParser()
    config.read('../config.ini')
    cluster_labels = [label.strip() for label in config['KmeansClustering']['cluster_labels'].split(',')]
    countries = ['United Kingdom', 'Germany', 'France']

    # User inputs
    country = st.selectbox("Choose a Country", countries)
    cluster = st.selectbox("Choose a Cluster", cluster_labels + ["All Clusters"])
    horizon = st.slider("Forecast Horizon (Months)", min_value=1, max_value=24, value=12)

    # Determine model file based on selections
    if cluster == "All Clusters":
        model_key = f"sales_forecaster_{country.replace(' ', '_')}"
    else:
        model_key = f"sales_forecaster_{cluster.replace(' ', '_')}_{country.replace(' ', '_')}"

    model_path = f"../Models/{model_key}.pkl"

    # Load and forecast
    try:
        with open(model_path, 'rb') as f:
            forecaster = pickle.load(f)

        # Convert horizon from months to approximate weeks (4 weeks per month on average)
        weeks_horizon = horizon * 4

        # Generate forecast (adjust based on your SalesForecaster's predict method)
        # Assuming predict returns a series or array; adjust if it returns a DataFrame
        forecast_values = forecaster.predict(future_steps=weeks_horizon)
        last_date = pd.to_datetime('2025-06-11')  # Current date
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=weeks_horizon, freq='D')
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'forecast': forecast_values
        })

        # Resample forecast to weekly for consistency
        forecast_df = forecast_df.set_index('date')['forecast'].resample('W').sum().reset_index()

        # Display forecast
        st.subheader(f"Forecast for {country} - {cluster}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=forecast_df['date'],
            y=forecast_df['forecast'],
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

    except FileNotFoundError:
        st.error(f"No model found for {country} - {cluster}. Please ensure the model was trained and saved to {model_path}.")
    except Exception as e:
        st.error(f"Error generating forecast: {str(e)}")

    # Display historical data
    try:
        data_file = '../Data/Online_Retail_Clustered.csv'
        df = pd.read_csv(data_file)

        # Standardize sales column
        if 'TotalPrice' in df.columns:
            df['total_sales'] = df['TotalPrice']
        elif 'Quantity' in df.columns and 'UnitPrice' in df.columns:
            df['total_sales'] = df['Quantity'] * df['UnitPrice']
        elif 'total_sales' not in df.columns:
            raise ValueError("No valid sales column found (expected 'total_sales', 'TotalPrice', or 'Quantity' and 'UnitPrice').")

        # Ensure date column exists and is in datetime format
        if 'InvoiceDate' in df.columns:
            df['date'] = pd.to_datetime(df['InvoiceDate'])
        else:
            raise ValueError("No valid date column found (expected 'InvoiceDate' or 'date').")

        # Filter data
        if cluster != "All Clusters":
            df = df[(df['Country'] == country) & (df['cluster'] == cluster)]
        else:
            df = df[df['Country'] == country]

        # Aggregate and resample by week
        df = df.groupby('date')['total_sales'].sum().reset_index()
        df = df.set_index('date')['total_sales'].resample('W').sum().reset_index().dropna()

        # Plot historical data
        st.subheader("Historical Sales")
        fig_hist = px.line(df, x='date', y='total_sales', title=f"Historical Sales for {country} - {cluster}")
        fig_hist.update_layout(
            xaxis_title="Date",
            yaxis_title="Sales",
            template="plotly_white"
        )
        st.plotly_chart(fig_hist)

    except Exception as e:
        st.warning(f"Could not display historical data: {str(e)}")