import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
import configparser
import os
import sys
from datetime import datetime

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

def prepare_heatmap_data(df):
    """Prepare data for heatmap showing sales by month and cluster."""
    heatmap_df = df.groupby(['cluster', 'Month'])['TotalPrice'].sum().reset_index()
    pivot_df = heatmap_df.pivot(index='cluster', columns='Month', values='TotalPrice').fillna(0)
    return pivot_df

def show_forecasting_page():
    """Display the sales forecasting page with interactive plots and metrics."""
    # Custom CSS for styling with dark mode support and inverted fonts
    dark_mode = st.get_option("theme.base") == "dark"
    st.markdown(
        f"""
        <style>
        /* General page styling */
        .stApp {{
           
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }}
        /* Plot container styling */
        .css-1aumxhk, .heatmap-container {{
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, {0.1 if not dark_mode else 0.3});
            padding: 10px;
        }}
        /* Metric card styling */
        .stMetric {{
            
            border-radius: 10px;
            padding: 15px;
            text-align: center;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            flex: 1 1 0;
            min-width: 0;
        }}
        .stMetric:hover {{
            transform: translateY(-2px);
        }}
        .stMetric .stMetricLabel {{
            font-size: 1rem;
            font-weight: 500;
            margin-bottom: 0.5rem;
        }}
        .stMetric .stMetricValue {{
            font-size: 1.75rem;
            font-weight: 700;
        }}
        /* Skills section styling */
        .skills-section {{
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
        }}
        .skills-title {{
            font-size: 22px;
            font-weight: 700;
            text-align: center;
            margin-bottom: 15px;
        }}
        .skills-list {{
            font-size: 16px;
            line-height: 1.6;
        }}
        /* Header and subheader styling */
        .css-1d391kg {{ /* Title styling */
            font-size: 2.5rem;
            font-weight: 700;
            
        }}
        .css-10trblm {{ /* Subheader styling */
            font-size: 1.5rem;
            font-weight: 600;
            
        }}
        /* Selectbox styling */
        .stSelectbox label {{
            font-size: 1rem;
            font-weight: 500;
        }}
        .stSelectbox div[role="combobox"] {{
            border-radius: 8px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("📈 Sales Forecasting")

    # Load and prepare data
    data_file = './Data/Online_Retail_Clustered.csv'
    df = load_data(data_file, with_cluster=True)
    historical_df = prep_historical_data(df)
    heatmap_df = prepare_heatmap_data(df)

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

            # Generate forecast for 290 days
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
                line=dict(color='#4CAF50' )
            ))
            fig.update_layout(
                title=f"Sales Forecast for {cluster}",
                xaxis_title="Date",
                yaxis_title="Sales",
                template="plotly_white" if not dark_mode else "plotly_dark"
            )
            st.plotly_chart(fig)

            # Display metrics
            if hasattr(forecaster, 'get_metrics'):
                metrics = forecaster.get_metrics()
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("RMSE", f"{metrics.get('rmse', 0):.2f}")
                with col2:
                    st.metric("MAPE", f"{metrics.get('mape', 0):.2f}%")
                with col3:
                    st.metric("R²", f"{metrics.get('r2', 0)*100:.2f}%")

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
        ))
        fig_hist.update_layout(
            title=f"Historical Sales for {cluster}",
            xaxis_title="Date",
            yaxis_title="Sales",
        )
        st.plotly_chart(fig_hist)
    except Exception as e:
        st.warning(f"Could not display historical data: {str(e)}")

    # Display heatmap to showcase skills
    st.subheader("Sales Heatmap by Cluster and Month")
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=heatmap_df.values,
        x=heatmap_df.columns,
        y=heatmap_df.index,
        colorscale='Viridis',
        colorbar_title="Total Sales"
    ))
    fig_heatmap.update_layout(
        title="Sales Distribution by Cluster and Month",
        xaxis_title="Month",
        yaxis_title="Cluster",
    )
    st.markdown('<div class="heatmap-container">', unsafe_allow_html=True)
    st.plotly_chart(fig_heatmap)

