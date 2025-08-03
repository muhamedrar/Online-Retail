# streamlit_app.py
import streamlit as st
from Pages import clustering, forcasting, insights ,Executive_Summary ,chatbot

st.set_page_config(page_title="Online Retail Analytics", layout="wide")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Executive Summary",
    "Customer Clustering",
    "Sales Forecasting",
    "Strategic Recommendations"
])

# Page Routing
if page == "Executive Summary":
    Executive_Summary.executive_summary_page()

elif page == "Customer Clustering":
    clustering.show_clustering_page()

elif page == "Sales Forecasting":
    forcasting.show_forecasting_page()

elif page == "Strategic Recommendations":
    insights.show_insights_page()
    chatbot.chatbot_ui()
