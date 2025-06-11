from statsmodels.tsa.seasonal import seasonal_decompose
import pickle
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import warnings
warnings.filterwarnings("ignore")
import configparser
from src.sales_forcasting import SalesForecaster



config = configparser.ConfigParser()
config.read('./config.ini')
clusters = int(config['KmeansClustering']['n_clusters'])
cluster_labels = [label.strip() for label in config['KmeansClustering']['cluster_labels'].split(',')]
print(cluster_labels)

# train model for each cluster 

for c in cluster_labels:
    print(f"Training model for cluster {c}")
    forecaster = SalesForecaster('./Data/Online_Retail_Clustered.csv', cluster=c)
    
    # export model
    with open(f'./Models/sales_forecaster_cluster_{c}.pkl', 'wb') as f:
        pickle.dump(forecaster, f)
    print(f"Model for cluster {c} trained and saved.")



# Train model for all clusters combined
forecaster = SalesForecaster('./Data/Online_Retail.csv')

# export model
with open('./Models/sales_forecaster.pkl', 'wb') as f:
    pickle.dump(forecaster, f)



## train model for each country (top 3 countries in the dataset)

countries = ['United Kingdom', 'Germany', 'France']
for country in countries:
    print(f"Training model for country {country}")
    forecaster = SalesForecaster('./Data/Online_Retail.csv', Country=country)
    
    # export model
    with open(f'./Models/sales_forecaster_{country.replace(" ", "_")}.pkl', 'wb') as f:
        pickle.dump(forecaster, f)


# training models for combinitions of clusters and countries

for cluster_label in cluster_labels:
    for country in countries:
        print(f"Training model for cluster {cluster_label} and country {country}")

        try:
            # Initialize and train the SalesForecaster
            forecaster = SalesForecaster('./Data/Online_Retail_Clustered.csv', cluster=cluster_label, Country=country)
            
            # Export model with sanitized filename
            filename = f'./Models/sales_forecaster_{cluster_label.replace(" ", "_")}_{country.replace(" ", "_")}.pkl'
            with open(filename, 'wb') as f:
                pickle.dump(forecaster, f)
            print(f"Model for cluster {cluster_label} and country {country} trained and saved as {filename}")
        except Exception as e:
            print(f"Error training model for cluster {cluster_label} and country {country}: {str(e)}")
            continue