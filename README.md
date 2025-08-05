# **Online Retail Customer Segmentation & Sales Forecasting Dashboard**

## Overview

This project is an end-to-end analytics solution for an online retail business, combining customer segmentation and cluster-based sales forecasting into a single, interactive Streamlit dashboard.
It enables businesses to:

* Identify customer segments using clustering techniques.
* Visualize purchasing behaviors with interactive charts.
* Forecast sales for different customer segments using statistical models.
* Make informed decisions for marketing, inventory, and pricing strategies.

---

## Business Problem

Online retailers often lack insights into their diverse customer base and struggle to predict future sales.
This leads to:

* Untargeted marketing campaigns and lower ROI.
* Poor inventory planning, causing overstock or shortages.
* Inefficient pricing strategies that fail to capture demand patterns.

---

## Business Value

This solution:

* Segments customers into actionable groups for targeted engagement.
* Monitors key metrics and purchasing patterns across customer clusters.
* Forecasts sales for better demand planning and budget allocation.
* Improves marketing efficiency and operational decision-making.

---

## Dataset Information

The dataset used is the **[Online Retail Dataset from Kaggle](https://www.kaggle.com/datasets/vijayuv/onlineretail)**.
It contains transactional data from a UK-based online retail store between **01/12/2024 and 09/12/2025**.

### Variables Table

| Variable Name | Role    | Type        | Description                                                                                                               | Units    |
| ------------- | ------- | ----------- | ------------------------------------------------------------------------------------------------------------------------- | -------- |
| InvoiceNo     | ID      | Categorical | A 6-digit integral number uniquely assigned to each transaction. If the code starts with 'C', it indicates a cancellation |          |
| StockCode     | ID      | Categorical | A 5-digit integral number uniquely assigned to each distinct product                                                      |          |
| Description   | Feature | Categorical | Product name                                                                                                              |          |
| Quantity      | Feature | Integer     | The quantities of each product (item) per transaction                                                                     |          |
| InvoiceDate   | Feature | Date        | The date and time when each transaction was generated                                                                     |          |
| UnitPrice     | Feature | Continuous  | Product price per unit                                                                                                    | Sterling |
| CustomerID    | Feature | Categorical | A 5-digit integral number uniquely assigned to each customer                                                              |          |
| Country       | Feature | Categorical | The name of the country where each customer resides                                                                       |          |

---

## Features

### **Customer Segmentation Dashboard**

* Date range filtering.
* Filter by customer cluster (Budget Shoppers, Core Shoppers, Premium Shoppers, Regular Shoppers, Wholesale Buyers).
* KPI cards for:

  * Total Sales
  * Total Transactions
  * Total Quantity
  * Average Price per Item
* Revenue contribution pie chart by segment.
* Monthly sales trends with peak period annotation.
* Segment comparison bar charts for:

  * Total Sales
  * Average Price per Item
  * Average Quantity per Transaction
  * Transaction Frequency
* Summary table of segment metrics.
* Option to **download segmented data as CSV**.

### **Cluster-Based Sales Forecasting**

* Comparison of historical vs forecasted sales.
* Forecast models trained separately for each segment.
* Evaluation metrics including RMSE, MAPE, and R².
* Cluster-month heatmap for seasonal trends.
* Forecast horizon customization (1–10 months ahead).

---
## Strategy Assistant Chatbot

An interactive **Strategy Assistant chatbot** is integrated into the dashboard to help users interpret customer segments, retention patterns, and sales insights.

**Features:**
- Ask business questions about customer clusters, retention, and sales strategies.
- Get concise, actionable recommendations based on cluster summaries and key insights.
- References cluster characteristics (e.g., Core Shoppers, Premium Shoppers) and suggests tactics (e.g., retention, upselling, campaign timing).
- Maintains chat history for context-aware responses.

**Example questions:**
- "Which cluster should I focus retention on?"
- "How can I increase sales from Premium Shoppers?"
- "What are the main risks with Wholesale Buyers?"

The chatbot leverages project-specific segmentation and forecasting insights to provide tailored, business-oriented answers for decision support.
## Notebooks & Experimentation

The project includes Jupyter notebooks for:

* **EDA\_Clustering\_Results.ipynb** – Exploratory Data Analysis of clustering results.
* **Clustering\_Techniques\_Experimentation.ipynb** – Testing Kmeans clustering algorithm.
* **Forecasting\_Model\_Experimentation.ipynb** – Comparing forecasting methods.
* **training\_models\_for\_forecasting.py** – Script to retrain models.

---

## Configuration

The **`config.ini`** file contains project settings and parameters for clustering, data preprocessing, and forecasting:

| Section             | Variable           | Description                                                                                   |
|---------------------|-------------------|-----------------------------------------------------------------------------------------------|
| `[KmeansClustering]`| `n_clusters`      | Number of customer clusters to generate using KMeans.                                         |
|                     | `data_import_path`| Path to the raw input dataset.                                                                |
|                     | `data_export_path`| Path to save the clustered dataset after segmentation.                                        |
|                     | `cluster_labels`  | Names assigned to each customer segment (comma-separated).                                    |
| `[DataPreprocessing]`| `rolling_window` | Window size (in days) for rolling average calculations during preprocessing.                  |
|                     | `decomposition_period` | Period (in days) for time series decomposition (e.g., weekly seasonality).               |
| `[ArimaForecasting]`| `order`           | ARIMA model order parameters: (p, d, q).                                                      |
|                     | `seasonal_order`  | Seasonal ARIMA parameters: (P, D, Q, s), where s is the seasonality period (e.g., 7 for week).|

**Example `config.ini`:**
```ini
[KmeansClustering]
n_clusters=5
data_import_path=./Data/Online_Retail.csv
data_export_path=./Data/Online_Retail_Clustered.csv
cluster_labels = Budget Shoppers,Core Shoppers,Premium Shoppers,Regular Shoppers,WholeSale Buyers

[DataPreprocessing]
rolling_window=7
decomposition_period=7

[ArimaForecasting]
order=0,1,1
seasonal_order=2,0,[1,2],7
```

---

## Tech Stack

* Python
* Pandas / NumPy
* scikit-learn
* statsmodels
* Plotly
* Streamlit

---

## Installation & Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/Online-Retail.git
   cd Online-Retail 
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate   # Mac/Linux
   venv\Scripts\activate     # Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Place the dataset `Online_Retail.csv` in the `Data/` directory.



5. Train forecasting models:

   ```bash
   python training_models_for_forecasting.py
   ```

---

## Running the Streamlit App

```bash
streamlit run streamlit_App/app.py
```

Then open:

```
http://localhost:8501
```

---





