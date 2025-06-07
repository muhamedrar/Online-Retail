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


# # train model for each cluster 

for c in range(clusters):
    print(f"Training model for cluster {c}")
    forecaster = SalesForecaster('./Data/Online_Retail_Clustered.csv', cluster=c)
    
    # export model
    with open(f'./Models/sales_forecaster_cluster_{c}.pkl', 'wb') as f:
        pickle.dump(forecaster, f)



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