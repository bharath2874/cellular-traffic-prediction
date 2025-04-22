import streamlit as st
import pandas as pd
from src.preprocessing import load_and_preprocess_data
from src.reduce_data import apply_pca
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

st.title("📶 Cellular Traffic Prediction using PCA + Random Forest")

uploaded_file = st.file_uploader("Upload your traffic CSV file", type=["csv"])

if uploaded_file is not None:
    df = load_and_preprocess_data(uploaded_file)
    st.subheader("📊 Raw Data Preview")
    st.write(df.head())

    X = df.drop("data_usage_mb", axis=1)
    y = df["data_usage_mb"]
    
    st.subheader("📉 Applying PCA (3 components)")
    reduced_data, _ = apply_pca(X)
    st.write(pd.DataFrame(reduced_data, columns=["PC1", "PC2", "PC3"]).head())

    st.subheader("🤖 Training Random Forest Model...")
    X_train, X_test, y_train, y_test = train_test_split(reduced_data, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    st.success(f"✅ Mean Squared Error: {mse:.2f}")

    st.subheader("📈 Actual vs Predicted Data Usage")
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.values[:50], label='Actual')
    plt.plot(predictions[:50], label='Predicted')
    plt.legend()
    st.pyplot(plt)
else:
    st.info("Please upload a dataset to get started.")
