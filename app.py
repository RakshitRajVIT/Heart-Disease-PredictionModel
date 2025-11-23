import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler

# Load Dataset
df = pd.read_csv("heart.csv")

# Load Best Model (CatBoost or any selected model)
model = joblib.load("heart_disease_model.pkl")

st.title("❤️ Heart Disease Prediction Dashboard")
st.write("This application uses multiple Machine Learning models to analyze heart disease data and make predictions.")

# Sidebar Navigation
menu = st.sidebar.selectbox("Select Section", 
                            ["📊 Dataset Overview", 
                             "📈 Visualizations", 
                             "🤖 Model Comparison", 
                             "❤️ Heart Disease Prediction"])

sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")

# -------------------------------------------------------------
# 📊 1. Dataset Overview
# -------------------------------------------------------------
if menu == "📊 Dataset Overview":
    st.header("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Summary Statistics")
    st.write(df.describe())

    st.subheader("Target Distribution")
    fig, ax = plt.subplots()
    df.target.value_counts().plot(kind="bar", color=["salmon", "lightblue"], ax=ax)
    ax.set_title("Heart Disease Distribution")
    st.pyplot(fig)

    st.write("165 people have heart disease, 138 do not.")

# -------------------------------------------------------------
# 📈 2. Visualizations
# -------------------------------------------------------------
elif menu == "📈 Visualizations":
    st.header("Data Visualizations")

    # --- Categorical Columns ---
    categorical_val = []
    continuous_val = []

    for column in df.columns:
        if len(df[column].unique()) <= 10:
            categorical_val.append(column)
        else:
            continuous_val.append(column)

    st.subheader("Categorical Feature Plots")
    fig = plt.figure(figsize=(15, 15))
    for i, column in enumerate(categorical_val, 1):
        plt.subplot(3, 3, i)
        df[df["target"] == 0][column].hist(color='blue', alpha=0.6, label="No Disease")
        df[df["target"] == 1][column].hist(color='red', alpha=0.6, label="Disease")
        plt.legend()
        plt.xlabel(column)
    st.pyplot(fig)

    # --- Continuous Columns ---
    st.subheader("Continuous Feature Plots")
    fig = plt.figure(figsize=(15, 15))
    for i, column in enumerate(continuous_val, 1):
        plt.subplot(3, 2, i)
        df[df["target"] == 0][column].hist(color='blue', alpha=0.6)
        df[df["target"] == 1][column].hist(color='red', alpha=0.6)
        plt.xlabel(column)
    st.pyplot(fig)

    # --- Scatter Plot ---
    st.subheader("Age vs Max Heart Rate")
    fig = plt.figure(figsize=(10, 8))
    plt.scatter(df.age[df.target==1], df.thalach[df.target==1], c="salmon")
    plt.scatter(df.age[df.target==0], df.thalach[df.target==0], c="lightblue")
    plt.xlabel("Age")
    plt.ylabel("Max Heart Rate")
    plt.legend(["Disease", "No Disease"])
    st.pyplot(fig)

    # --- Correlation Heatmap ---
    st.subheader("Correlation Matrix")
    corr_matrix = df.corr()
    fig, ax = plt.subplots(figsize=(15, 15))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
    st.pyplot(fig)

# -------------------------------------------------------------
# 🤖 3. Model Comparison
# -------------------------------------------------------------
elif menu == "🤖 Model Comparison":
    st.header("Model Accuracy Comparison")

    # Read pre-saved comparison CSV (recommended)
    results = pd.read_csv("model_comparison.csv")  # You must save this from your model section

    st.dataframe(results)

    fig = plt.figure(figsize=(10, 6))
    bar_width = 0.35
    indices = range(len(results))
    plt.bar(indices, results['Training Accuracy'], width=bar_width, label='Train Accuracy')
    plt.bar([i + bar_width for i in indices], results['Testing Accuracy'], width=bar_width, label='Test Accuracy')
    plt.xticks([i + bar_width/2 for i in indices], results['Model'])
    plt.title("Training vs Testing Accuracy")
    plt.legend()
    st.pyplot(fig)

# -------------------------------------------------------------
# ❤️ 4. Heart Disease Prediction
# -------------------------------------------------------------
elif menu == "❤️ Heart Disease Prediction":
    st.header("Heart Disease Prediction")

    st.write("Enter patient information to predict heart disease:")

    # User Inputs
    age = st.number_input("Age", 1, 120)
    sex = st.selectbox("Sex (1=Male, 0=Female)", [0,1])
    cp = st.selectbox("Chest Pain Type (0-3)", [0,1,2,3])
    trestbps = st.number_input("Resting Blood Pressure")
    chol = st.number_input("Cholesterol Level")
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0,1])
    restecg = st.selectbox("Rest ECG (0-2)", [0,1,2])
    thalach = st.number_input("Max Heart Rate Achieved")
    exang = st.selectbox("Exercise Induced Angina", [0,1])
    oldpeak = st.number_input("Oldpeak Value", 0.0, 10.0)
    slope = st.selectbox("Slope (0-2)", [0,1,2])
    ca = st.selectbox("Number of Vessels", [0,1,2,3])
    thal = st.selectbox("Thal (0-3)", [0,1,2,3])

    # Prepare input
    input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg,
                                thalach, exang, oldpeak, slope, ca, thal]],
                              columns=df.drop("target", axis=1).columns)

    # Scaling required columns
    scaler = StandardScaler()
    col_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    input_data[col_to_scale] = scaler.fit_transform(input_data[col_to_scale])

    # Predict
    if st.button("Predict"):
        prediction = model.predict(input_data)[0]

        if prediction == 1:
            st.error("🚨 High chance of heart disease!")
        else:
            st.success("💚 No signs of heart disease detected.")

