from src.preprocessing import load_and_preprocess_data
from src.reduce_data import apply_pca
from src.train_model import train_and_evaluate

if __name__ == "__main__":
    df = load_and_preprocess_data("data/traffic_data.csv")
    reduced_df, _ = apply_pca(df.drop("data_usage_mb", axis=1))
    train_and_evaluate(reduced_df, df["data_usage_mb"])
