import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import plotly.subplots as sp
import plotly.express as px
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
from data_preprocessing import load_data
import warnings
warnings.filterwarnings("ignore")
import numpy as np
@st.cache_data
def load_and_prepare_data(file_path):
    """Load and preprocess the dataset."""
    df = load_data(file_path, with_cluster=True)
    df = df.rename(columns={'cluster': 'segment'})
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    return df

def apply_date_filter(df, start_date, end_date):
    """Filter data based on the provided date range."""
    if start_date > end_date:
        st.error("End date must be after start date.")
        return None
    mask = (df['InvoiceDate'].dt.date >= start_date) & (df['InvoiceDate'].dt.date <= end_date)
    return df[mask].copy()

def calculate_kpis(filtered_df, selected_segments):
    """Calculate key performance indicators (KPIs) and segment summary."""
    segment_summary = filtered_df.groupby('segment').agg({
        'InvoiceNo': 'count',
        'TotalPrice': 'sum',
        'Quantity': 'sum'
    }).rename(columns={'InvoiceNo': 'Frequency'}).sort_index()

    segment_summary['Avg price per transaction'] = segment_summary['TotalPrice'] / segment_summary['Frequency']
    segment_summary['Avg quantity per transaction'] = segment_summary['Quantity'] / segment_summary['Frequency']
    segment_summary['Avg price per item'] = segment_summary['TotalPrice'] / segment_summary['Quantity']

    filtered_summary = segment_summary.loc[selected_segments].fillna(0)
    
    total_sales = filtered_df['TotalPrice'].sum()
    total_transactions = filtered_df['InvoiceNo'].count()
    total_quantity = filtered_df['Quantity'].sum()
    avg_price_per_item = filtered_df['TotalPrice'].sum() / filtered_df['Quantity'].sum() if filtered_df['Quantity'].sum() > 0 else 0
    avg_quantity_per_transaction = filtered_df['Quantity'].sum() / filtered_df['InvoiceNo'].count() if filtered_df['InvoiceNo'].count() > 0 else 0
    
    return {
        'segment_summary': segment_summary,
        'filtered_summary': filtered_summary,
        'total_sales': total_sales,
        'total_transactions': total_transactions,
        'total_quantity': total_quantity,
        'avg_price_per_item': avg_price_per_item,
        'avg_quantity_per_transaction': avg_quantity_per_transaction
    }

def load_css(file_path="styles.css"):
    """Load CSS file for styling."""
    try:
        with open(file_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("CSS file not found. Using default Streamlit styling.")

def render_kpi_cards(kpis):
    """Render KPI cards for total sales, transactions, quantity, and avg price per item with custom styling."""
    st.markdown("""
        <style>
            .kpi-row {
                display: flex;
                flex-direction: row;
                justify-content: space-between;
                align-items: stretch;
                gap: 16px;
                width: 100%;
                margin-bottom: 1rem;
            }
            .kpi-card {
                background: linear-gradient(135deg, #ffffff 50%, #f1f5f9 100%);
                border-radius: 10px;
                padding: 1rem 0.5rem;
                box-shadow: 0 3px 12px rgba(0, 0, 0, 0.05);
                transition: transform 0.2s ease, box-shadow 0.2s ease;
                text-align: center;
                flex: 1 1 0;
                min-width: 0;
            }
            .kpi-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 16px rgba(0, 0, 0, 0.05);
            }
            .kpi-label {
                font-size: 1rem;
                font-weight: 500;
                color: #64748b;
                margin-bottom: 0.5rem;
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            }
            .kpi-value {
                font-size: 1.75rem;
                font-weight: 700;
                color: #1e293b;
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown(
        f'''
        <div class="kpi-row">
            <div class="kpi-card">
                <div class="kpi-label">Total Sales</div>
                <div class="kpi-value">${kpis["total_sales"]:,.2f}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Total Transactions</div>
                <div class="kpi-value">{kpis["total_transactions"]:,.0f}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Total Quantity</div>
                <div class="kpi-value">{kpis["total_quantity"]:,.0f}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Avg Price per Item</div>
                <div class="kpi-value">${kpis["avg_price_per_item"]:,.2f}</div>
            </div>
        </div>
        ''',
        unsafe_allow_html=True
    )

import plotly.express as px
import streamlit as st

def render_pie_chart(filtered_summary):
    """Render a stylish larger pie chart with dark-mode-friendly markdown heading, centered."""
    better_colors = px.colors.qualitative.Pastel + px.colors.qualitative.Set2

    # Centered markdown heading (works in dark mode)
    st.markdown("<h4 style='text-align:center;'>Revenue Contribution by Segment</h4>", unsafe_allow_html=True)

    pie_fig = px.pie(
        filtered_summary.reset_index(),
        names='segment',
        values='TotalPrice',
        color='segment',
        color_discrete_sequence=better_colors,
        hole=0  # Full pie, no donut
    )

    pie_fig.update_traces(
        textinfo='percent',  # Only show percentages
        textfont=dict(size=18, family='Arial', color='white'),
        marker=dict(
            line=dict(color='white', width=2)  # Clean separation between slices
        ),
        pull=[0.04]*len(filtered_summary),  # Slight pull for style
        hoverinfo='label+percent+value'
    )

    pie_fig.update_layout(
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.3,
            xanchor='center',
            x=0.5,
            font=dict(size=16)
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=30, b=60),
        showlegend=True,
        height=480,
        width=None
    )

    st.plotly_chart(pie_fig, use_container_width=True)


def render_sub_kpi_cards(filtered_summary):
    """Render sub-KPI cards for avg price and quantity per transaction with custom styling."""
    avg_price_per_transaction = filtered_summary['Avg price per transaction'].mean()
    avg_quantity_per_transaction = filtered_summary['Avg quantity per transaction'].mean()

    st.markdown("""
        <style>
            .sub-kpi-row {
                display: flex;
                flex-direction: row;
                justify-content: space-between;
                align-items: stretch;
                gap: 16px;
                width: 100%;
                margin-bottom: 1rem;
            }
            .sub-kpi-card {
                background: linear-gradient(135deg, #f8fafc 50%, #e0e7ef 100%);
                border-radius: 10px;
                padding: 0.75rem 0.5rem;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
                transition: transform 0.2s ease, box-shadow 0.2s ease;
                text-align: center;
                flex: 1 1 0;
                min-width: 0;
            }
            .sub-kpi-card:hover {
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
            }
            .sub-kpi-label {
                font-size: 0.95rem;
                font-weight: 500;
                color: #64748b;
                margin-bottom: 0.4rem;
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            }
            .sub-kpi-value {
                font-size: 1.4rem;
                font-weight: 700;
                color: #1e293b;
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown(
        f'''
        <div class="sub-kpi-row">
            <div class="sub-kpi-card">
                <div class="sub-kpi-label">Avg Price per Transaction</div>
                <div class="sub-kpi-value">${avg_price_per_transaction:,.2f}</div>
            </div>
            <div class="sub-kpi-card">
                <div class="sub-kpi-label">Avg Quantity per Transaction</div>
                <div class="sub-kpi-value">{avg_quantity_per_transaction:,.2f}</div>
            </div>
        </div>
        ''',
        unsafe_allow_html=True
    )

def render_line_chart(filtered_df, selected_segments, start_date, end_date):
    """Render line chart for sales over time by segment with shaded area matching segment colors."""
    # st.markdown("<h3 style='text-align:center; margin-bottom:0.7rem; margin-top:1.5rem;'>Monthly Sales by Segment</h3>", unsafe_allow_html=True)
    time_df = filtered_df.copy()
    time_df['Month'] = time_df['InvoiceDate'].dt.to_period('M').dt.to_timestamp()
    sales_over_time = time_df.groupby(['segment', 'Month'])['TotalPrice'].sum().reset_index()

    # Use the same color palette as pie/bar charts for consistency
    better_colors = px.colors.qualitative.Pastel + px.colors.qualitative.Set2
    segment_indices = sorted(filtered_df['segment'].unique())
    color_map = {segment: better_colors[i % len(better_colors)] for i, segment in enumerate(segment_indices)}

    line_fig = go.Figure()
    for segment in selected_segments:
        seg_data = sales_over_time[sales_over_time['segment'] == segment]
        if seg_data.empty:
            continue
        peak_idx = seg_data['TotalPrice'].idxmax()
        peak_month = seg_data.loc[peak_idx, 'Month']
        peak_value = seg_data.loc[peak_idx, 'TotalPrice']
        seg_color = color_map.get(segment, "#888")

        # Add shaded area (fill='tozeroy') with segment color and lower opacity
        line_fig.add_trace(go.Scatter(
            x=seg_data['Month'],
            y=seg_data['TotalPrice'],
            mode='lines',
            name=f"Segment {segment} Shade",
            line=dict(width=0, color=seg_color),
            fill='tozeroy',
            fillcolor=seg_color.replace('rgb', 'rgba').replace(')', ',0.18)') if seg_color.startswith('rgb') else seg_color + '33',
            opacity=1,
            showlegend=False
        ))
        # Add main line on top for legend and clarity
        line_fig.add_trace(go.Scatter(
            x=seg_data['Month'],
            y=seg_data['TotalPrice'],
            mode='lines+markers',
            name=f"Segment {segment}",
            line=dict(width=3, color=seg_color),
            marker=dict(size=8, color=seg_color),
        ))
        # Peak marker
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
        
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
        template="plotly_white",
        height=600,
        margin=dict(l=50, r=0, t=50, b=50)
    )
    line_fig.update_xaxes(
        dtick="M1",
        tickformat="%b\n%Y",
        range=[pd.Timestamp(start_date), pd.Timestamp(end_date)]
    )
    st.plotly_chart(line_fig, use_container_width=True)

def render_bar_charts(filtered_summary, selected_metrics):
    """Render bar charts for selected metrics with 100% width."""
    metric_options = [
        ('TotalPrice', 'Total Sales per Segment', 'Total Sales'),
        ('Avg price per item', 'Avg Price per Item', 'Avg Price per Item'),
        ('Avg quantity per transaction', 'Avg Quantity per Transaction', 'Avg Quantity per Transaction'),
        ('Frequency', 'Number of Transactions', 'Frequency')
    ]
    bar_configs = [m for m in metric_options if m[0] in selected_metrics]

    if bar_configs:
        n = len(bar_configs)
        ncols = 1 if n == 1 else 2
        nrows = (n + ncols - 1) // ncols

        fig = sp.make_subplots(rows=nrows, cols=ncols, subplot_titles=[m[1] for m in bar_configs], vertical_spacing=0.25)

        segment_indices = list(filtered_summary.index)
        better_colors = px.colors.qualitative.Pastel + px.colors.qualitative.Set2
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
                    name=title,
                    orientation='h',
                    marker=dict(line=dict(width=1)),
                    hovertemplate=hovertemplate,
                    textfont=dict(color='white', size=20)  # Set text color to white and bigger font
                ),
                row=row, col=col_pos
            )
            fig.update_xaxes(title_text=ylabel, row=row, col=col_pos, tickprefix=xaxis_tickprefix, automargin=True)
            fig.update_yaxes(title_text="", row=row, col=col_pos, automargin=True)

        # Set width to 100% by using use_container_width=True and removing explicit width
        fig.update_layout(
            height=600,
            title_text="Segment Summary Insights",
            showlegend=False,
            template="plotly_white",
            barmode='group',
            bargap=0.2,
            margin=dict(l=100, r=0, t=100, b=50)  # Remove right margin
        )
        st.plotly_chart(fig, use_container_width=True)

def render_summary_table(filtered_summary):
    """Render the filtered summary table with width fit to page."""
    st.subheader("Filtered Summary Table")
    st.dataframe(filtered_summary, use_container_width=True)

def show_clustering_page():
    """Main function to render the customer segmentation dashboard."""
    
    # load_css(os.path.join(os.path.dirname(__file__), "CSS", "clustering.css"))
    st.title("Customer Segmentation Dashboard")

    # Load and prepare data
    df = load_and_prepare_data("./Data/Online_Retail_Clustered.csv")

    # Place date selector in the sidebar for better UX and to save horizontal space
    with st.sidebar:
        st.header("Date Range Filter")
        start_date = st.date_input(
            "Start Date",
            value=pd.to_datetime("2024-12-01").date(),
            key="start_date"
        )
        end_date = st.date_input(
            "End Date",
            value=pd.to_datetime("2025-09-30").date(),
            key="end_date"
        )

    # Apply date filter
    filtered_df = apply_date_filter(df, start_date, end_date)
    if filtered_df is None:
        return

    # Segment selection in the sidebar (navbar)
    with st.sidebar:
        st.header("Segment Filter")
        all_segments = sorted(filtered_df['segment'].unique())
        selected_segments = st.multiselect(
            "Select Segment(s)",
            options=all_segments,
            default=all_segments,
            key="segment_multiselect"
        )

    filtered_df = filtered_df[filtered_df['segment'].isin(selected_segments)]
    kpis = calculate_kpis(filtered_df, selected_segments)
    render_kpi_cards(kpis)

    # Render charts
    left_col, right_col = st.columns([0.3, 0.7])
    with left_col:
        render_pie_chart(kpis['filtered_summary'])
        render_sub_kpi_cards(kpis['filtered_summary'])
    with right_col:
        render_line_chart(filtered_df, selected_segments, start_date, end_date)

    # Render metric selector above the bar charts, centered and styled
    st.markdown("<h4 style='text-align:center; margin-top:2rem;'>Segment Metrics Comparison</h4>", unsafe_allow_html=True)
    col_metrics, _ = st.columns([0.35, 0.65])
    with col_metrics:
        selected_metrics = st.multiselect(
            "Choose metrics to plot",
            options=['TotalPrice', 'Avg price per item', 'Avg quantity per transaction', 'Frequency'],
            default=['TotalPrice', 'Avg price per item', 'Avg quantity per transaction', 'Frequency'],
            key="metrics_multiselect_below",
            help="Select which metrics to display in the bar charts below."
        )
    render_bar_charts(kpis['filtered_summary'], selected_metrics)

    # Render summary table
    render_summary_table(kpis['filtered_summary'])

    # Download button for cluster data
    # --- Download button styling ---
    st.markdown("""
        <style>
        /* Download button */
        .stDownloadButton>button {
            background-color: rgb(255, 75, 75);
            color: white;
            border-radius: 8px;
            padding: 0.75rem 1.5rem;
            font-size: 1rem;
            font-weight: 500;
            border: none;
            transition: background-color 0.3s ease, transform 0.2s ease, box-shadow 0.2s ease;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .stDownloadButton>button::before {
            content: '↓'; /* Simple download icon */
            font-size: 1rem;
        }
        .stDownloadButton>button:hover {
            background-color: #b91c1c; /* dark red on hover */
            color: white !important;
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(185, 28, 28, 0.10); /* red shadow */
        }
        .stDownloadButton>button:focus {
            outline: none;
            box-shadow: 0 0 0 3px rgba(225, 29, 72, 0.3);
        }
        </style>
    """, unsafe_allow_html=True)

    csv = df.to_csv(index=True)
    st.download_button(
        label="Download The Segmented Data as CSV",
        data=csv,
        file_name="Segmented_data.csv",
        mime="text/csv",
        key="download_cluster_data"
    )