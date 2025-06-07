# streamlit_app.py
import streamlit as st
from Pages import clustering, forcasting, insights

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
    st.title("🛍️ Online Retail Analysis - Executive Summary")
    st.markdown("""
    ### What This App Does
    - Segments customers into behavioral clusters using unsupervised learning.
    - Forecasts future sales across top-performing countries and customer segments.

    ### Why It Matters
    - Enables targeted marketing and retention strategies.
    - Supports inventory planning and demand forecasting.

    ### Key Insights
    - Cluster 2 drives 45% of revenue but shows signs of churn.
    - Germany sales peak earlier than the UK — plan promotions accordingly.
    - December is the highest demand month — align supply chain.
    """)

elif page == "Customer Clustering":
    clustering.show_clustering_page()

elif page == "Sales Forecasting":
    forcasting.show_forecasting_page()

elif page == "Strategic Recommendations":
    insights.show_insights_page()
