# -----------------------------------------
# HEART DISEASE PREDICTION - COMPLETE MODEL
# -----------------------------------------

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

import joblib  # For saving the model

sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")


# Step 2: Read Data
df = pd.read_csv("heart.csv")
print(df.head())


# Step 3: Data Summary
pd.set_option("display.float", "{:.2f}".format)
print(df.describe())

print(df.isna().sum())  # No missing values


# Step 4: Visualize Target Distribution
df.target.value_counts().plot(kind="bar", color=["salmon", "lightblue"])
plt.title("Heart Disease Count")
plt.show()


# Step 5: Categorize Columns
categorical_val = []
continous_val = []

for column in df.columns:
    if len(df[column].unique()) <= 10:
        categorical_val.append(column)
    else:
        continous_val.append(column)

categorical_val.remove("target")


# Step 6: Categorical Plots
plt.figure(figsize=(15, 15))
for i, column in enumerate(categorical_val, 1):
    plt.subplot(3, 3, i)
    df[df.target == 0][column].hist(bins=35, color='blue', alpha=0.6, label='No Disease')
    df[df.target == 1][column].hist(bins=35, color='red', alpha=0.6, label='Disease')
    plt.title(column)
    plt.legend()
plt.show()


# Step 7: Continuous Plots
plt.figure(figsize=(15, 15))
for i, column in enumerate(continous_val, 1):
    plt.subplot(3, 2, i)
    df[df.target == 0][column].hist(bins=35, color='blue', alpha=0.6, label='No Disease')
    df[df.target == 1][column].hist(bins=35, color='red', alpha=0.6, label='Disease')
    plt.title(column)
    plt.legend()
plt.show()


# Step 8: Scatter Plot Age vs Thalach
plt.figure(figsize=(10, 8))
plt.scatter(df.age[df.target == 1], df.thalach[df.target == 1], c="salmon")
plt.scatter(df.age[df.target == 0], df.thalach[df.target == 0], c="lightblue")
plt.title("Heart Disease vs Age & Max Heart Rate")
plt.xlabel("Age")
plt.ylabel("Thalach")
plt.legend(["Disease", "No Disease"])
plt.show()


# Step 9: Correlation Matrix
plt.figure(figsize=(15, 15))
sns.heatmap(df.corr(), annot=True, cmap="YlGnBu")
plt.show()

df.drop("target", axis=1).corrwith(df.target).plot(kind="bar", figsize=(12, 8), title="Correlation with Target")
plt.show()


# STEP 10: DATA PROCESSING
dataset = pd.get_dummies(df, columns=categorical_val)

scaler = StandardScaler()
cols_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
dataset[cols_to_scale] = scaler.fit_transform(dataset[cols_to_scale])


# Step 11: Model Training Setup
X = dataset.drop("target", axis=1)
y = dataset["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print(f"Training Accuracy: {accuracy_score(y_train, y_train_pred) * 100:.2f}%")
    print(f"Testing Accuracy: {accuracy_score(y_test, y_test_pred) * 100:.2f}%")
    print("Classification Report:\n", classification_report(y_test, y_test_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))

    return accuracy_score(y_train, y_train_pred), accuracy_score(y_test, y_test_pred)


# Models
print("\n--- Random Forest ---")
rf = RandomForestClassifier(random_state=42)
rf_train, rf_test = evaluate_model(rf, X_train, y_train, X_test, y_test)

print("\n--- Gradient Boosting ---")
gb = GradientBoostingClassifier(random_state=42)
gb_train, gb_test = evaluate_model(gb, X_train, y_train, X_test, y_test)

print("\n--- XGBoost ---")
xgb_model = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
xgb_train, xgb_test = evaluate_model(xgb_model, X_train, y_train, X_test, y_test)

print("\n--- LightGBM ---")
lgb_model = lgb.LGBMClassifier(random_state=42)
lgb_train, lgb_test = evaluate_model(lgb_model, X_train, y_train, X_test, y_test)

print("\n--- CatBoost ---")
cb_model = CatBoostClassifier(verbose=0, random_state=42)
cb_train, cb_test = evaluate_model(cb_model, X_train, y_train, X_test, y_test)


# Compare all models
results = pd.DataFrame({
    "Model": ["Random Forest", "Gradient Boosting", "XGBoost", "LightGBM", "CatBoost"],
    "Train Accuracy": [rf_train*100, gb_train*100, xgb_train*100, lgb_train*100, cb_train*100],
    "Test Accuracy": [rf_test*100, gb_test*100, xgb_test*100, lgb_test*100, cb_test*100],
})

print("\nMODEL COMPARISON:\n")
print(results)


# Visualization of model performance
plt.figure(figsize=(10, 6))
x = range(len(results))
bar_width = 0.35

plt.bar(x, results["Train Accuracy"], width=bar_width, label="Train Accuracy")
plt.bar([i + bar_width for i in x], results["Test Accuracy"], width=bar_width, label="Test Accuracy")

plt.xticks([i + bar_width/2 for i in x], results["Model"], rotation=45)
plt.ylabel("Accuracy (%)")
plt.title("Model Accuracy Comparison")
plt.legend()
plt.show()


# Save best model (CatBoost)
joblib.dump(cb_model, "heart_disease_model.pkl")
print("\n➡ Model saved as heart_disease_model.pkl")
