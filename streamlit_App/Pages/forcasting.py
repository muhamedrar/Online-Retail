import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
import configparser
import os
import sys
from datetime import datetime, timedelta

# Add src directory to system path for custom modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
src_path = os.path.join(project_root, 'src')
sys.path.append(src_path)

from sales_forcasting import SalesForecaster
from data_preprocessing import load_data, preprocess_data, feature_engineering, remove_outliers

def prep_historical_data(df, rolling_window=7):
    df = preprocess_data(df)
    df = feature_engineering(df)
    df = remove_outliers(df, 'Quantity')
    df = remove_outliers(df, 'UnitPrice')
    df = df.groupby(df['InvoiceDate'].dt.date)['TotalPrice'].sum().to_frame()
    df['TotalPrice'] = df['TotalPrice'].rolling(window=rolling_window).mean()
    df.dropna(inplace=True)
    return df.reset_index().rename(columns={'index': 'InvoiceDate'})

def prepare_heatmap_data(df):
    heatmap_df = df.groupby(['cluster', 'Month'])['TotalPrice'].sum().reset_index()
    pivot_df = heatmap_df.pivot(index='cluster', columns='Month', values='TotalPrice').fillna(0)
    return pivot_df

def render_metric_cards(metrics):
    st.markdown("""
        <style>
            .kpi-row {
                display: flex;
                flex-direction: row;
                justify-content: space-around;
                gap: 16px;
                margin-top: 1rem;
                margin-bottom: 1rem;
            }
            .kpi-card {
                background: linear-gradient(135deg, #ffffff 50%, #f1f5f9 100%);
                border-radius: 12px;
                padding: 1.25rem;
                box-shadow: 0 3px 12px rgba(0, 0, 0, 0.05);
                text-align: center;
                flex: 1;
                transition: transform 0.2s ease, box-shadow 0.2s ease;
                cursor: help;
            }
            .kpi-card:hover {
                transform: translateY(-3px);
                box-shadow: 0 6px 16px rgba(0, 0, 0, 0.1);
            }
            .kpi-label {
                font-size: 1rem;
                font-weight: 500;
                color: #64748b;
                margin-bottom: 0.5rem;
            }
            .kpi-value {
                font-size: 1.75rem;
                font-weight: 700;
                color: #1e293b;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown(f"""
        <div class="kpi-row">
            <div class="kpi-card" title="Root Mean Square Error">
                <div class="kpi-label">RMSE</div>
                <div class="kpi-value">{metrics.get('rmse', 0):.2f}</div>
            </div>
            <div class="kpi-card" title="Mean Absolute Percentage Error">
                <div class="kpi-label">MAPE</div>
                <div class="kpi-value">{metrics.get('mape', 0):.2f}%</div>
            </div>
            <div class="kpi-card" title="R-squared Value">
                <div class="kpi-label">R²</div>
                <div class="kpi-value">{metrics.get('r2', 0) * 100:.2f}%</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

def show_forecasting_page():
    st.title("Cluster-Based Sales Forecasting")

    data_file = '../Data/Online_Retail_Clustered.csv'
    df = load_data(data_file, with_cluster=True)
    config = configparser.ConfigParser()
    config.read('../config.ini')
    cluster_labels = [label.strip() for label in config['KmeansClustering']['cluster_labels'].split(',')]

    with st.sidebar:
        st.header("Settings")
        cluster = st.selectbox("Choose a Cluster", ["All Clusters"] + cluster_labels)
        start_date = st.date_input("Start Date", value=pd.to_datetime(df['InvoiceDate']).min().date())
        end_date = st.date_input("End Date", value=pd.to_datetime(df['InvoiceDate']).max().date())
        forecast_months = st.slider("Forecast Months", min_value=1, max_value=10, value=10)

    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    if cluster != "All Clusters":
        df = df[df['cluster'] == cluster]

    historical_df = prep_historical_data(df)
    heatmap_df = prepare_heatmap_data(df)

    model_key = "sales_forecaster_global" if cluster == "All Clusters" else f"sales_forecaster_cluster_{cluster.replace(' ', '_')}"
    model_path = f"../Models/{model_key}.pkl"

    try:
        if not os.path.exists(model_path):
            st.error(f"No model available for {cluster}. Please train the model first.")
            return

        with open(model_path, 'rb') as f:
            forecaster = pickle.load(f)

        forecast_values = forecaster.forecast(future_steps=299)

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

        historical_df['InvoiceDate'] = pd.to_datetime(historical_df['InvoiceDate'])
        filtered_hist = historical_df[
            (historical_df['InvoiceDate'].dt.date >= start_date) &
            (historical_df['InvoiceDate'].dt.date <= end_date)
        ]
        start_forecast = forecast_df['date'].min()
        end_forecast = start_forecast + pd.DateOffset(months=forecast_months)
        filtered_forecast = forecast_df[
            (forecast_df['date'] >= start_forecast) &
            (forecast_df['date'] < end_forecast)
        ]

        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(
            x=filtered_forecast['date'],
            y=filtered_forecast['Sales'],
            mode='lines+markers',
            name='Forecast',
            line=dict(color='#4CAF50', width=3),
            marker=dict(size=5)
        ))
        fig_forecast.update_layout(
            title=f"Forecasted Sales for {cluster}",
            xaxis_title="Date",
            yaxis_title="Sales",
            template="plotly_white",
            hovermode="x unified",
            legend=dict(orientation="h", x=0.5, xanchor="center")
        )
        st.plotly_chart(fig_forecast)

        if hasattr(forecaster, 'get_metrics'):
            metrics = forecaster.get_metrics()
            st.markdown("### Model Evaluation")
            render_metric_cards(metrics)

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(
            x=filtered_hist['InvoiceDate'],
            y=filtered_hist['TotalPrice'],
            mode='lines+markers',
            name='Historical Sales',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=5)
        ))
        fig_hist.update_layout(
            title=f"Historical Sales for {cluster}",
            xaxis_title="Date",
            yaxis_title="Sales",
            template="plotly_white",
            hovermode="x unified",
            legend=dict(orientation="h", x=0.5, xanchor="center")
        )
        st.plotly_chart(fig_hist)

    except Exception as e:
        st.error(f"Error generating forecast: {str(e)}")

    st.markdown("### Cluster-Month Sales Heatmap")
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=heatmap_df.values,
        x=heatmap_df.columns,
        y=heatmap_df.index,
        colorscale='Viridis',
        colorbar_title="Total Sales",
        text=heatmap_df.round(0).astype(int).astype(str),
        hoverinfo='text'
    ))
    fig_heatmap.update_traces(
        texttemplate="%{text}",
        textfont={"size": 12}
    )
    fig_heatmap.update_layout(
        title="Total Sales by Cluster & Month",
        xaxis_title="Month",
        yaxis_title="Cluster",
        template="plotly_white"
    )
    st.plotly_chart(fig_heatmap)
