from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import sys
import os
import configparser
# Add the parent directory of 'src' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import warnings
warnings.filterwarnings("ignore")
from src.data_preprocessing import remove_outliers , preprocess_data, load_data ,feature_engineering


config = configparser.ConfigParser()
config.read('./config.ini')

if 'KmeansClustering' in config and \
   all(k in config['KmeansClustering'] for k in ['n_clusters', 'data_import_path', 'data_export_path']):
    n_clusters = config['KmeansClustering'].getint('n_clusters')
    data_import_path = config['KmeansClustering']['data_import_path']
    data_export_path = config['KmeansClustering']['data_export_path']
else:
    # Provide default values or raise an error
    print("KmeansClustering section not found in config.ini or missing keys. Using default values.")
    n_clusters = 5
    data_import_path = './Data/Online_Retail.csv'
    data_export_path = './Data/Clustered_Online_Retail.csv'



def prep_data_for_clustering(df):
    
    df = preprocess_data(df)
    df = feature_engineering(df)
    df = remove_outliers(df, 'UnitPrice')
    df = remove_outliers(df, 'Quantity') 
    df = remove_outliers(df, 'UnitPrice')
    

    data = df[['Quantity','UnitPrice','Country']]
    preCluster = data.copy()
    preCluster = pd.get_dummies(data, columns=['Country'])

    # Normalize the data
    scaler = StandardScaler()
    
    preCluster[['Quantity','UnitPrice']] = scaler.fit_transform(preCluster[['Quantity','UnitPrice']])
    print(f"prep_data_for_clustering done with shape: {preCluster.shape}")
    return preCluster , data



def perform_clustering(df, n_clusters=n_clusters):
    print(f"Starting clustering with {n_clusters} clusters...")
    preCluster,data = prep_data_for_clustering(df)
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(preCluster)
    data = load_data('./Data/Online_Retail.csv', with_cluster=False)
    data = preprocess_data(data)
    data = feature_engineering(data)
    
    data = remove_outliers(data, 'Quantity')
    data = remove_outliers(data, 'UnitPrice')
    data = remove_outliers(data, 'UnitPrice')

    data['cluster'] = labels
    print(f"perform_clustering done with {n_clusters} clusters.")
    return data




df = load_data(data_import_path,with_cluster = False)
data = perform_clustering(df, n_clusters=n_clusters)
#map clusters names
Cluster_names = {
    0: 'Regular Shoppers',
    1: 'Premium Shoppers',
    2: 'WholeSale Buyers',
    3: 'Core Shoppers',
    4: 'Budget Shoppers',
}
data['cluster'] = data['cluster'].map(Cluster_names)
data.to_csv(data_export_path, index=False)












