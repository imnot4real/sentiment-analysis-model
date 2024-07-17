import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def split_data(df, test_size=0.2, random_state=42):
    X = df['review']
    y = df['sentiment']
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
