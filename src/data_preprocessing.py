import pandas as pd

def load_data(file_path, with_cluster = False):

    try:
        if with_cluster == False:
            types = {'InvoiceNo': str, 'StockCode': str, 'Description': str, 'Quantity': int, 'InvoiceDate': str, 'UnitPrice': float, 'CustomerID': float, 'Country': str}
            data = pd.read_csv(file_path, dtype=types)
            
            return data
        else:
            types = {'InvoiceNo': str, 'StockCode': str, 'Description': str, 'Quantity': int, 'InvoiceDate': str, 'UnitPrice': float, 'CustomerID': float, 'Country': str, 'cluster': int}
            data = pd.read_csv(file_path, dtype=types)
            return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    

def preprocess_data(df):
    if df is None:
        print("Error: DataFrame is None. Please check the data loading process.")
        return None
    df = df[~df['InvoiceNo'].str.startswith('C')] # Remove cancelled orders
    df.dropna(inplace=True) # Remove rows with NaN values
    df.drop_duplicates(inplace=True) # Remove duplicate rows
    
    
    
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df = feature_engineering(df)
    print(f"preprocess_data done with shape: {df.shape}")

    return df


def save_preprocessed_data(df, output_path):
    
    try:
        df.to_csv(output_path, index=False)
        print(f"Preprocessed data saved to {output_path}")
    except Exception as e:
        print(f"Error saving preprocessed data: {e}")


def feature_engineering(df):

    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    df['TotalPrice'] = df['TotalPrice'].astype(float)
    df['Year'] = pd.to_datetime(df['InvoiceDate']).dt.year
    df['Month'] = pd.to_datetime(df['InvoiceDate']).dt.month
    df['Day'] = pd.to_datetime(df['InvoiceDate']).dt.day
    df['Hour'] = pd.to_datetime(df['InvoiceDate']).dt.hour

    return df


def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Count the number of records to be removed
    initial_count = df.shape[0]
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    removed_count = initial_count - filtered_df.shape[0]
    
    # Print the count of removed records
    print(f"Removed {removed_count} outliers from column '{column}'")
    
    return filtered_df



