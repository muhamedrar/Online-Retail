import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.subplots as sp
import plotly.express as px
from datetime import datetime
import base64
import os

def show_clustering_page():
    st.title("👥 Customer Segmentation Dashboard")

    # Load CSS
    css_path = "/home/mohamed/Desktop/Projects/Online-Retail/streamlit_App/Pages/styling/clustering.css"
    if os.path.exists(css_path):
        with open(css_path, "r") as f:
            css_content = f.read()
            st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    else:
        st.warning("CSS file not found at the specified path. Using default inline styles.")
        st.markdown("""
        <style>
        .main {
            background-color: #f9fbfd;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        }
        .kpi-card {
            background: linear-gradient(90deg, #f8fafc 60%, #e0e7ef 100%);
            border-radius: 12px;
            padding: 18px 20px;
            margin-bottom: 15px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
            text-align: left;
            transition: transform 0.2s;
        }
        .kpi-card:hover {
            transform: scale(1.02);
        }
        .kpi-label {
            font-size: 1.1rem;
            color: #6c757d;
            margin-bottom: 5px;
        }
        .kpi-value {
            font-size: 2.1rem;
            font-weight: 700;
            color: #1a237e;
        }
        .kpi-sub-row {
            display: flex;
            gap: 20px;
            margin-top: 20px;
            margin-bottom: 10px;
            justify-content: center;
        }
        .kpi-sub-card {
            background: linear-gradient(90deg, #e3f2fd 60%, #fce4ec 100%);
            border-radius: 10px;
            padding: 15px 20px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.07);
            text-align: center;
            min-width: 200px;
            transition: transform 0.2s;
        }
        .kpi-sub-card:hover {
            transform: scale(1.02);
        }
        .kpi-sub-label {
            font-size: 1.1rem;
            color: #607d8b;
            margin-bottom: 5px;
        }
        .kpi-sub-value {
            font-size: 2.1rem;
            font-weight: 700;
            color: #ad1457;
        }
        .stButton>button {
            background-color: #1a237e;
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 1rem;
            border: none;
            transition: background-color 0.3s;
        }
        .stButton>button:hover {
            background-color: #2c3e50;
        }
        .stDateInput > div > input {
            width: 200px !important;
            padding: 8px;
            border: 1px solid #ced4da;
            border-radius: 4px;
            font-size: 14px;
            color: #495057;
        }
        .stDateInput > label {
            font-size: 14px;
            color: #6c757d;
            margin-bottom: 5px;
        }
        .stMultiSelect > div > div {
            background-color: #f8fafc;
            border-radius: 4px;
            padding: 5px;
        }
        </style>
        """, unsafe_allow_html=True)

    # Load data
    df = pd.read_csv("./Data/Online_Retail_Clustered.csv")
    df = df.rename(columns={'cluster': 'segment'})
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    # --- Date Selector ---
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=pd.to_datetime("2024-12-01").date(), key="start_date")
    with col2:
        end_date = st.date_input("End Date", value=pd.to_datetime("2025-09-30").date(), key="end_date")
    if start_date > end_date:
        st.error("End date must be after start date.")
        return

    # Filter data based on date range
    mask = (df['InvoiceDate'].dt.date >= start_date) & (df['InvoiceDate'].dt.date <= end_date)
    filtered_df = df[mask].copy()

    # --- KPIs and Segment Selection ---
    segment_summary = filtered_df.groupby('segment').agg({
        'InvoiceNo': 'count',
        'TotalPrice': 'sum',
        'Quantity': 'sum'
    }).rename(columns={'InvoiceNo': 'Frequency'}).sort_index()

    segment_summary['Avg price per transaction'] = segment_summary['TotalPrice'] / segment_summary['Frequency']
    segment_summary['Avg quantity per transaction'] = segment_summary['Quantity'] / segment_summary['Frequency']
    segment_summary['Avg price per item'] = segment_summary['TotalPrice'] / segment_summary['Quantity']

    all_segments = sorted(filtered_df['segment'].unique())

    selected_segments = st.multiselect(
        "Select Segment(s)",
        options=all_segments,
        default=all_segments,
        key="segment_multiselect"
    )

    filtered_df = filtered_df[filtered_df['segment'].isin(selected_segments)]
    filtered_summary = segment_summary.loc[selected_segments].fillna(0)

    total_sales = filtered_df['TotalPrice'].sum()
    total_transactions = filtered_df['InvoiceNo'].count()
    total_quantity = filtered_df['Quantity'].sum()
    avg_price_per_item = filtered_df['TotalPrice'].sum() / filtered_df['Quantity'].sum() if filtered_df['Quantity'].sum() > 0 else 0
    avg_quantity_per_transaction = filtered_df['Quantity'].sum() / filtered_df['InvoiceNo'].count() if filtered_df['InvoiceNo'].count() > 0 else 0

    # --- KPI Display ---
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Total Sales</div>
            <div class="kpi-value">${total_sales:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Total Transactions</div>
            <div class="kpi-value">{total_transactions:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Total Quantity</div>
            <div class="kpi-value">{total_quantity:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Avg Price per Item</div>
            <div class="kpi-value">${avg_price_per_item:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    # --- Pie and Line Charts Side by Side ---
    left_col, right_col = st.columns([1, 2])

    with left_col:
        better_colors = px.colors.qualitative.Pastel + px.colors.qualitative.Set2
        pie_fig = px.pie(
            filtered_summary.reset_index(),
            names='segment',
            values='TotalPrice',
            color='segment',
            color_discrete_sequence=better_colors,
            hole=0
        )
        pie_fig.update_traces(
            textinfo='percent+label',
            textfont=dict(size=14, color='#fff'),
            marker=dict(line=dict(color='rgba(0,0,0,0)', width=0)),
            pull=[0.02]*len(filtered_summary)
        )
        pie_fig.update_layout(
            title=dict(
                text='Revenue Contribution by Segment',
                font=dict(size=22, color='#1a237e'),
                x=0.5,
                xanchor='center'
            ),
            legend=dict(
                font=dict(size=12, color='#6c757d'),
                orientation='h',
                yanchor='bottom',
                y=-0.35,
                xanchor='center',
                x=0.5,
                bgcolor='rgba(250,250,255,0.9)'
            ),
            font=dict(size=14, color='#222'),
            margin=dict(l=20, r=20, t=60, b=40),
            paper_bgcolor='rgba(250,250,255,1)',
            plot_bgcolor='rgba(250,250,255,1)',
            showlegend=True
        )
        st.plotly_chart(pie_fig, use_container_width=True)

        # --- Additional KPIs ---
        st.markdown("""
        <style>
        .kpi-sub-row {
            display: flex;
            gap: 20px;
            margin-top: 20px;
            margin-bottom: 10px;
            justify-content: center;
        }
        .kpi-sub-card {
            background: linear-gradient(90deg, #e3f2fd 60%, #fce4ec 100%);
            border-radius: 10px;
            padding: 15px 20px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.07);
            text-align: center;
            min-width: 200px;
            transition: transform 0.2s;
        }
        .kpi-sub-card:hover {
            transform: scale(1.02);
        }
        .kpi-sub-label {
            font-size: 1.1rem;
            color: #607d8b;
            margin-bottom: 5px;
        }
        .kpi-sub-value {
            font-size: 2.1rem;
            font-weight: 700;
            color: #ad1457;
        }
        </style>
        """, unsafe_allow_html=True)
        avg_price_per_transaction = filtered_summary['Avg price per transaction'].mean()
        avg_quantity_per_transaction = filtered_summary['Avg quantity per transaction'].mean()
        st.markdown(f"""
        <div class="kpi-sub-row">
            <div class="kpi-sub-card">
                <div class="kpi-sub-label">Avg Price per Transaction</div>
                <div class="kpi-sub-value">${avg_price_per_transaction:,.2f}</div>
            </div>
            <div class="kpi-sub-card">
                <div class="kpi-sub-label">Avg Quantity per Transaction</div>
                <div class="kpi-sub-value">{avg_quantity_per_transaction:,.2f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with right_col:
        st.markdown("### 📈 Sales Over Time by Segment")
        time_df = filtered_df.copy()
        time_df['Month'] = time_df['InvoiceDate'].dt.to_period('M').dt.to_timestamp()
        sales_over_time = time_df.groupby(['segment', 'Month'])['TotalPrice'].sum().reset_index()

        line_fig = go.Figure()
        for segment in selected_segments:
            seg_data = sales_over_time[sales_over_time['segment'] == segment]
            if seg_data.empty:
                continue
            peak_idx = seg_data['TotalPrice'].idxmax()
            peak_month = seg_data.loc[peak_idx, 'Month']
            peak_value = seg_data.loc[peak_idx, 'TotalPrice']

            line_fig.add_trace(go.Scatter(
                x=seg_data['Month'],
                y=seg_data['TotalPrice'],
                mode='lines+markers',
                name=f"Segment {segment}",
                line=dict(width=3),
                marker=dict(size=8),
            ))
            line_fig.add_trace(go.Scatter(
                x=[peak_month],
                y=[peak_value],
                mode='markers+text',
                marker=dict(size=12, color='crimson', symbol='diamond'),
                text=[f'${peak_value:,.0f}'],
                textposition="top center",
                name=f"Peak Segment {segment}",
                showlegend=False
            ))

        line_fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Total Sales",
            title="Monthly Sales Trend by Segment",
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
            template="plotly_white",
            height=600,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        line_fig.update_xaxes(
            dtick="M1",
            tickformat="%b\n%Y",
            range=[pd.Timestamp(start_date), pd.Timestamp(end_date)]
        )
        st.plotly_chart(line_fig, use_container_width=True)

    # --- Show filtered summary table ---
    st.dataframe(filtered_summary.style.set_properties(**{'text-align': 'center'}).set_table_styles(
        [{'selector': 'th', 'props': [('background-color', '#f8fafc'), ('color', '#1a237e'), ('font-size', '14px')]},
         {'selector': 'td', 'props': [('font-size', '12px'), ('padding', '8px')]}]
    ))

    # --- Bar Charts Section ---
    metric_options = [
        ('TotalPrice', 'Total Sales per Segment', 'Total Sales'),
        ('Avg price per item', 'Avg Price per Item', 'Avg Price per Item'),
        ('Avg quantity per transaction', 'Avg Quantity per Transaction', 'Avg Quantity per Transaction'),
        ('Frequency', 'Number of Transactions', 'Frequency')
    ]
    selected_metrics = st.multiselect(
        "Choose metrics to plot",
        options=[m[0] for m in metric_options],
        default=[m[0] for m in metric_options],
        key="metrics_multiselect_below"
    )
    bar_configs = [m for m in metric_options if m[0] in selected_metrics]

    with st.expander("Customize Plot Dimensions", expanded=False):
        fig_width = st.slider("Figure Width", min_value=800, max_value=1400, value=1200, key="fig_width_below")
        fig_height = st.slider("Figure Height", min_value=400, max_value=1200, value=600, key="fig_height_below")

    if bar_configs:
        n = len(bar_configs)
        ncols = 1 if n == 1 else 2
        nrows = (n + ncols - 1) // ncols

        fig = sp.make_subplots(rows=nrows, cols=ncols, subplot_titles=[m[1] for m in bar_configs], vertical_spacing=0.25)

        segment_indices = list(filtered_summary.index)
        color_map = {segment: better_colors[i % len(better_colors)] for i, segment in enumerate(segment_indices)}

        for idx, (col, title, ylabel) in enumerate(bar_configs):
            row = idx // ncols + 1
            col_pos = idx % ncols + 1
            sorted_summary = filtered_summary.sort_values(by=col, ascending=True)
            max_value = sorted_summary[col].max()
            normalized_values = sorted_summary[col] / max_value if max_value > 0 else sorted_summary[col]
            bar_colors = [color_map[segment] for segment in sorted_summary.index]
            if col == 'Avg price per item':
                text_vals = [f'${v:,.2f}' for v in sorted_summary[col]]
                hovertemplate = 'Segment: %{y}<br>Avg Price per Item: $%{x:,.2f}<extra></extra>'
                xaxis_tickprefix = '$'
            elif col == 'TotalPrice':
                text_vals = [f'${v:,.2f}' for v in sorted_summary[col]]
                hovertemplate = 'Segment: %{y}<br>Total Sales: $%{x:,.2f}<extra></extra>'
                xaxis_tickprefix = '$'
            else:
                text_vals = [f'{v:,.2f}' for v in sorted_summary[col]]
                hovertemplate = None
                xaxis_tickprefix = ''

            fig.add_trace(
                go.Bar(
                    y=sorted_summary.index.astype(str),
                    x=normalized_values,
                    marker_color=bar_colors,
                    text=text_vals,
                    textposition='auto',
                    textfont=dict(size=14),
                    name=title,
                    orientation='h',
                    marker=dict(line=dict(width=1, color='rgba(0,0,0,0.2)')),
                    hovertemplate=hovertemplate
                ),
                row=row, col=col_pos
            )
            fig.update_xaxes(title_text=ylabel, row=row, col=col_pos, tickprefix=xaxis_tickprefix, automargin=True)
            fig.update_yaxes(title_text="", row=row, col=col_pos, automargin=True)

        fig.update_layout(
            height=fig_height,
            width=fig_width,
            title_text="Segment Summary Insights",
            showlegend=False,
            template="plotly_white",
            barmode='group',
            bargap=0.2,
            margin=dict(l=100, r=100, t=100, b=50)
        )
        st.plotly_chart(fig, use_container_width=False)



  

