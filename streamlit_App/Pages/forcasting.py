import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


def show_forecasting_page():
    st.title("📈 Sales Forecasting")

    country = st.selectbox("Choose a Country", ["United Kingdom", "Germany", "France"])
    # df = pd.read_csv(f"data/sales_forecaster_{country}.csv")

    # st.line_chart(df.set_index("date")[["actual", "forecast"]])

    # st.metric("RMSE", round(df['rmse'].iloc[-1], 2))
