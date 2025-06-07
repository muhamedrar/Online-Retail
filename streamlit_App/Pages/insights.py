import streamlit as st

def show_insights_page():
    st.title("📌 Strategic Recommendations")

    st.markdown("""
    ### 🎯 Cluster-Specific Actions
    - **Cluster 0:** High-value, loyal customers → Launch loyalty rewards.
    - **Cluster 1:** New buyers → Trigger welcome campaigns.
    - **Cluster 2:** Low frequency, high spend → Offer seasonal reactivation.

    ### 🌍 Country-Specific Sales Trends
    - **UK:** Peaks in December → Boost logistics readiness.
    - **Germany:** Peaks early → Launch campaigns in Nov.
    - **France:** Stable year-round → Maintain current approach.
    """)