import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Inject custom CSS for typography / card styling
st.markdown("""
    <style>
        /* Typography */
        h1 { font-size: 2.8rem; margin-bottom: 0.2rem; }
        h2 { font-size: 2.2rem; margin-top: 1.5rem; }
        h3 { font-size: 1.8rem; margin-top: 1rem; }
        .summary-card {
            background: #f9fafb;
            border-radius: 14px;
            padding: 18px 22px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.06);
            margin-bottom: 16px;
        }
        .small-label {
            font-size: 0.85rem;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        .badge {
            display: inline-block;
            background: #2563eb;
            color: white;
            padding: 4px 12px;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-right: 6px;
        }
        .insight-box {
            background: #e0f2fe;
            border-left: 4px solid #0284c7;
            padding: 12px 16px;
            border-radius: 8px;
            margin-bottom: 12px;
        }
    </style>
""", unsafe_allow_html=True)

# Cluster info mapping
CLUSTER_INFO = {
    "Cluster 0": {"name": "Regular Shoppers", "desc": "Frequent shoppers purchasing moderately priced items in small quantities per transaction. They contribute steadily to sales with consistent, medium-sized purchases.", "insight": "Encourage upselling through bundles or loyalty programs to increase transaction value."},
    "Cluster 1": {"name": "Premium Shoppers", "desc": "Selective buyers making fewer transactions with high-value, premium-priced items. They buy expensive products in small quantities.", "insight": "Target with premium product campaigns to increase transaction frequency."},
    "Cluster 2": {"name": "Wholesale Buyers", "desc": "Infrequent, high-volume purchases of low-cost items, likely wholesalers or businesses.", "insight": "Offer bulk discounts or B2B programs to encourage more frequent purchases."},
    "Cluster 3": {"name": "Core Shoppers", "desc": "Frequent, high-volume purchases of moderately priced items, driving the majority of sales and quantity.", "insight": "Prioritize retention and upselling, as they are the core revenue drivers."},
    "Cluster 4": {"name": "Budget Shoppers", "desc": "Frequent shoppers purchasing small quantities of low-cost items, likely bargain seekers.", "insight": "Use promotions to maintain high frequency, and explore low-cost add-ons for upselling."}
}


def compute_cohort_retention(df, freq='M'):
    """
    Returns:
      retention: DataFrame of retention rates (proportions)
      counts: raw counts of active customers per cohort index
      cohort_sizes: size of each cohort at index 0
    """
    df = df.copy()
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['CohortStart'] = df.groupby('CustomerID')['InvoiceDate'].transform('min')
    if freq == 'Q':
        df['CohortPeriod'] = df['CohortStart'].dt.to_period('Q')
        df['InvoicePeriod'] = df['InvoiceDate'].dt.to_period('Q')
    else:
        df['CohortPeriod'] = df['CohortStart'].dt.to_period('M')
        df['InvoicePeriod'] = df['InvoiceDate'].dt.to_period('M')

    df['CohortIndex'] = (df['InvoicePeriod'] - df['CohortPeriod']).apply(lambda x: x.n)
    cohort_counts = df.groupby(['CohortPeriod', 'CohortIndex'])['CustomerID'].nunique().reset_index()
    pivot_counts = cohort_counts.pivot(index='CohortPeriod', columns='CohortIndex', values='CustomerID').fillna(0)
    cohort_size = pivot_counts.iloc[:, 0]
    retention = pivot_counts.divide(cohort_size, axis=0).fillna(0)
    return retention, pivot_counts, cohort_size


def plot_cohort_heatmap(retention, counts, freq_label):
    retention_text = retention.round(3).applymap(lambda v: f"{v:.0%}")
    counts_text = counts.astype(int).astype(str)

    fig = go.Figure()

    # Retention heatmap
    fig.add_trace(go.Heatmap(
        z=retention.values,
        x=[f"Period {i}" for i in retention.columns],
        y=[str(c) for c in retention.index.astype(str)],
        text=retention_text.values,
        hovertemplate="<b>Cohort:</b> %{y}<br><b>Period:</b> %{x}<br><b>Retention:</b> %{text}<extra></extra>",
        colorscale="Blues",
        colorbar=dict(title="Retention", tickformat=".0%")
    ))

    # Raw counts heatmap (hidden initially)
    fig.add_trace(go.Heatmap(
        z=counts.values,
        x=[f"Period {i}" for i in counts.columns],
        y=[str(c) for c in counts.index.astype(str)],
        text=counts_text.values,
        hovertemplate="<b>Cohort:</b> %{y}<br><b>Period:</b> %{x}<br><b>Customers:</b> %{text}<extra></extra>",
        colorscale="Greys",
        showscale=True,
        visible=False,
        colorbar=dict(title="Customers")
    ))

    # Dropdown buttons
    fig.update_layout(
        title=f"Cohort Retention ({freq_label})",
        xaxis_title="Periods Since First Purchase",
        yaxis_title=f"Cohort Start ({'Quarter' if freq_label=='Quarterly' else 'Month'})",
        template="plotly_white",
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.5,
                y=1.16,
                showactive=True,
                buttons=[
                    dict(label="Retention %",
                         method="update",
                         args=[{"visible": [True, False]},
                               {"title": f"Cohort Retention (%) ({freq_label})"}]),
                    dict(label="Raw Counts",
                         method="update",
                         args=[{"visible": [False, True]},
                               {"title": f"Cohort Sizes (Counts) ({freq_label})"}]),
                ],
                pad={"r": 10, "t": 10},
            )
        ],
        margin=dict(t=100, l=100)
    )

    fig.add_annotation(
        text="Toggle between percentage retention and raw customer counts.",
        showarrow=False,
        xref="paper",
        yref="paper",
        x=0,
        y=1.09,
        align="left",
        font=dict(size=12, color="gray")
    )

    return fig


def resolve_cluster_name(cluster_label):
    key = f"Cluster {cluster_label}"
    if key in CLUSTER_INFO:
        return CLUSTER_INFO[key]["name"]
    for k, v in CLUSTER_INFO.items():
        if v["name"] == cluster_label:
            return v["name"]
    return str(cluster_label)


def executive_summary_page():
    st.markdown("# Executive Summary")
    st.markdown("## Project Overview")
    st.markdown(
        """
        This project segments customers into actionable archetypes and forecasts their future purchasing behavior.
        The goal is to enable **targeted marketing**, **improve early retention**, and **optimize inventory planning** based on segment-level demand characteristics.
        """
    )

    # Load data
    data_file = './Data/Online_Retail_Clustered.csv'
    df = pd.read_csv(
        data_file,
        dtype={'InvoiceNo': str, 'StockCode': str, 'Description': str,
               'Quantity': int, 'InvoiceDate': str, 'UnitPrice': float,
               'CustomerID': 'float64', 'Country': str, 'cluster': str}
    )
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

    # Aggregations
    cluster_groups = df.groupby('cluster')
    avg_price_per_item = (df.assign(price_per_item=df['TotalPrice'] / df['Quantity'])
                            .groupby('cluster')['price_per_item']
                            .mean()
                            .round(2))
    avg_quantity_per_invoice = cluster_groups['Quantity'].mean().round(2)
    revenue_by_cluster = cluster_groups['TotalPrice'].sum()
    customer_counts = cluster_groups['CustomerID'].nunique()
    total_revenue = df['TotalPrice'].sum()

    top_cluster = revenue_by_cluster.idxmax()
    top_cluster_name = resolve_cluster_name(top_cluster)

    # High-level KPIs
    st.markdown("### High-Level Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Revenue", f"£{total_revenue:,.0f}")
    c2.metric("Top Revenue Cluster", f"{top_cluster_name}")
    c3.metric("Distinct Customers", f"{df['CustomerID'].nunique():,}")
    st.markdown("---")
    # Key insights cards
    st.markdown("### Key Insights")
    
    insight_col1, insight_col2 = st.columns(2)
    with insight_col1:
        st.markdown('<div class="summary-card">', unsafe_allow_html=True)
        st.markdown('<div class="small-label">Price Differentiation</div>', unsafe_allow_html=True)
        st.markdown(f"<h3 style='margin:4px 0;'>Average Price / Item by Segment</h3>", unsafe_allow_html=True)
        for cid, price in avg_price_per_item.items():
            name = resolve_cluster_name(cid)
            st.markdown(f"<b>{name}:</b> £{price:.2f} per item", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with insight_col2:
        st.markdown('<div class="summary-card">', unsafe_allow_html=True)
        st.markdown('<div class="small-label">Purchase Volume</div>', unsafe_allow_html=True)
        st.markdown(f"<h3 style='margin:4px 0;'>Quantity per Transaction</h3>", unsafe_allow_html=True)
        for cid, qty in avg_quantity_per_invoice.items():
            name = resolve_cluster_name(cid)
            st.markdown(f"<b>{name}:</b> {qty:.2f} items/invoice", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")
    # Segment profiles
    st.markdown("## Customer Segment Profiles")
    def sort_key(c):
        try:
            return int(c)
        except:
            return str(c).lower()
    clusters = sorted(revenue_by_cluster.index.tolist(), key=sort_key)

    for cid in clusters:
        info_key = f"Cluster {cid}"
        name = CLUSTER_INFO.get(info_key, {}).get("name", resolve_cluster_name(cid))
        desc = CLUSTER_INFO.get(info_key, {}).get("desc", "")
        insight = CLUSTER_INFO.get(info_key, {}).get("insight", "")
        segment_revenue = revenue_by_cluster.loc[cid]
        num_customers = customer_counts.loc[cid]

        st.markdown(f"### {name}")
        st.markdown(f"- **Revenue Contribution:** £{segment_revenue:,.0f}  ")
        st.markdown(f"- **Distinct Customers:** {num_customers:,}  ")
        st.markdown("---")

    # Cohort section with dropdown
    st.markdown("## Retention Analysis (Cohort)")
    granularity = st.selectbox("Cohort Granularity", ["Monthly", "Quarterly"])
    freq = 'M' if granularity == "Monthly" else 'Q'
    retention, counts, cohort_sizes = compute_cohort_retention(df, freq=freq)
    with st.expander("View interactive cohort retention heatmap", expanded=True):
        if retention.empty:
            st.info("Not enough data to compute cohort retention.")
        else:
            fig = plot_cohort_heatmap(retention, counts, freq_label=granularity)
            st.plotly_chart(fig, use_container_width=True)
            if retention.shape[1] > 1:
                period1_avg = retention.iloc[:, 1].mean()
                st.markdown(
                    f"**Retention Insight:** After the first period, average retention drops to **{period1_avg:.0%}**, "
                    "suggesting early reactivation or onboarding campaigns could meaningfully increase customer lifetime."
                )

    # Executive summary narrative
    st.markdown("## Executive Summary")
    st.markdown(f"""
    - Total revenue is **£{total_revenue:,.0f}**, driven primarily by **{top_cluster_name}** (Cluster {top_cluster}).  
    - Segmentation reveals distinct levers: retention for core shoppers, frequency growth for premium segments, and structured B2B engagement for wholesale buyers.  
    - Customer retention drops sharply after the first period; improving early engagement could expand the active customer base.  
    """)

    # Recommendations
    st.markdown("## Recommendations")
    st.markdown(
        """
        - Launch tiered loyalty and upsell programs for **Regular** and **Core Shoppers** to increase lifetime value.  
        - Deploy personalized premium campaigns to boost frequency among **Premium Shoppers**.  
        - Build structured B2B engagement (volume discounts, onboarding) for **Wholesale Buyers**.  
        - Introduce onboarding/reactivation sequences for new cohorts to reduce early churn.  
        """
    )
