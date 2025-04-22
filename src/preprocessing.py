import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    le = LabelEncoder()
    df["location_type"] = le.fit_transform(df["location_type"])
    return df
