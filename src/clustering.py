from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import sys
import os

# Add the parent directory of 'src' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_preprocessing import remove_outliers , preprocess_data, load_data


def prep_data_for_clustering(df):
    
    df = preprocess_data(df)
    df = remove_outliers(df, 'Quantity')
    df = remove_outliers(df, 'TotalPrice')
    df = remove_outliers(df, 'UnitPrice')
    

    preCluster = df[['Quantity','UnitPrice','Country']]
    
    preCluster = pd.get_dummies(preCluster, columns=['Country'])

    # Normalize the data
    scaler = StandardScaler()
    preCluster[['Quantity','UnitPrice']] = scaler.fit_transform(preCluster[['Quantity','UnitPrice']])
    print(f"prep_data_for_clustering done with shape: {preCluster.shape}")
    return preCluster



def perform_clustering(df, n_clusters=5):
    print(f"Starting clustering with {n_clusters} clusters...")
    preCluster = prep_data_for_clustering(df)
    
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(preCluster)
    print(f"perform_clustering done with {n_clusters} clusters.")
    return labels




df = load_data('./Data/Online_Retail.csv',with_cluster = False)

labels = perform_clustering(df, n_clusters=5)
df['cluster'] = labels
df.to_csv('../Data/Online_Retail_Clustered', index=False)








