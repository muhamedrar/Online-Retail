from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
from src.data_preprocessing import remove_outliers , preprocess_data


def prep_data_for_clustering(df):
    
    df = preprocess_data(df)
    df = remove_outliers(df, 'Quantity')
    df = remove_outliers(df, 'UnitPrice')

    preCluster = df[['Quantity','UnitPrice','Country']]
    
    preCluster = pd.get_dummies(preCluster, columns=['Country'])
 
    # Normalize the data
    scaler = StandardScaler()
    preCluster[['Quantity','UnitPrice']] = scaler.fit_transform(preCluster[['Quantity','UnitPrice']])
    
    return preCluster



def perform_clustering(df, n_clusters=5):
    preCluster = prep_data_for_clustering(df)
    
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(preCluster)
    
    return labels



###  add labels to the original dataframe (figure out how to do this in the project later)





