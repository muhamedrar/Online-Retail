from statsmodels.tsa.seasonal import seasonal_decompose
import pickle
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import warnings
warnings.filterwarnings("ignore")
import configparser
from sales_forcasting import SalesForecaster



config = configparser.ConfigParser()
config.read('./config.ini')
clusters = int(config['KmeansClustering']['n_clusters'])
cluster_labels = [label.strip() for label in config['KmeansClustering']['cluster_labels'].split(',')]
print(f"Cluster labels: {cluster_labels}")


# Train a single global model for all data
print("Training global model for all data...")
forecaster = SalesForecaster('./Data/Online_Retail.csv')
# Generate and print forecast
forecast_values = forecaster.forecast(future_steps=30)  # Using 30 days as a sample
print(f"Global Model Forecast (first 5 values):\n{forecast_values.head()}")
# Export global model
with open('./Models/sales_forecaster_global.pkl', 'wb') as f:
    pickle.dump(forecaster, f)
print("Global model trained and saved as ./Models/sales_forecaster_global.pkl")

# Train one model per cluster
for c in cluster_labels:
    print(f"Training model for cluster {c}")
    try:
        forecaster = SalesForecaster('./Data/Online_Retail_Clustered.csv', cluster=c)
        # Generate and print forecast
        forecast_values = forecaster.forecast(future_steps=30)  # Using 30 days as a sample
        print(f"Forecast for cluster {c} (first 5 values):\n{forecast_values.head()}")
        # Export cluster-specific model
        with open(f'./Models/sales_forecaster_cluster_{c.replace(' ', '_')}.pkl', 'wb') as f:
            pickle.dump(forecaster, f)
        print(f"Model for cluster {c} trained and saved as ./Models/sales_forecaster_cluster_{c.replace(' ', '_')}.pkl")
    except Exception as e:
        print(f"Error training model for cluster {c}: {str(e)}")
        continue