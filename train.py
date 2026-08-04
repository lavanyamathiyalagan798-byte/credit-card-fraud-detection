import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Dataset
df = pd.read_csv("creditcard.csv")

# Features and Target
X = df.drop("Class", axis=1)
y = df["Class"]

# Standardize Time and Amount
scaler = StandardScaler()
X["Time"] = scaler.fit_transform(X[["Time"]])
X["Amount"] = scaler.fit_transform(X[["Amount"]])

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train Model
print("Training started...")

model = RandomForestClassifier(
    n_estimators=50,
    random_state=42,
    n_jobs=1
)

model.fit(X_train, y_train)

print("Training completed!")

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save Model
joblib.dump(model, "fraud_model.pkl")

print("\nModel trained successfully!")
print("Model saved as fraud_model.pkl")