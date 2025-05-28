from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import sys
import os

# Add the parent directory of 'src' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_preprocessing import remove_outliers , preprocess_data, load_data ,feature_engineering


def prep_data_for_clustering(df):
    
    df = preprocess_data(df)
    df = feature_engineering(df)
    df = remove_outliers(df, 'TotalPrice')
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



def perform_clustering(df, n_clusters=5):
    print(f"Starting clustering with {n_clusters} clusters...")
    preCluster,data = prep_data_for_clustering(df)
    
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(preCluster)
    data['cluster'] = labels
    print(f"perform_clustering done with {n_clusters} clusters.")
    return data




df = load_data('./Data/Online_Retail.csv',with_cluster = False)
data = perform_clustering(df, n_clusters=5)
data.to_csv('./Data/Online_Retail_Clustered.csv', index=False)









