import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
from tabulate import tabulate  # for clean table display

# Step 1: Load cleaned dataset
df = pd.read_csv("data/text_data_clean.csv")
print("✅ Dataset loaded successfully:", df.shape)

# Optional: Train on smaller sample for faster results (you can remove later)
df = df.sample(n=20000, random_state=42)

# Step 2: Split features and labels
X = df["text"]
y = df["label"]

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 4: TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=2000, stop_words="english")
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)
print("✅ Text vectorization completed!")

# Step 5: Train Multiple Models
results = []

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
print("\n⚙️ Training Logistic Regression...")
lr_model.fit(X_train_vect, y_train)
lr_pred = lr_model.predict(X_test_vect)
lr_acc = accuracy_score(y_test, lr_pred)
results.append(["Logistic Regression", "Linear Model", f"{lr_acc * 100:.2f}%"])

# Random Forest
rf_model = RandomForestClassifier(n_estimators=10, max_depth=10, n_jobs=-1, random_state=42)
print("\n⚙️ Training Random Forest Classifier...")
rf_model.fit(X_train_vect, y_train)
rf_pred = rf_model.predict(X_test_vect)
rf_acc = accuracy_score(y_test, rf_pred)
results.append(["Random Forest", "Ensemble Model", f"{rf_acc * 100:.2f}%"])

# Step 6: Display Accuracy Table
print("\n📊 Model Accuracy Comparison:\n")
table = tabulate(
    results,
    headers=["Algorithm", "Type", "Accuracy (%)"],
    tablefmt="fancy_grid"
)
print(table)

# Step 7: Save Best Model (based on accuracy)
best_model = rf_model if rf_acc > lr_acc else lr_model
joblib.dump(best_model, "models/cyber_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("\n✅ Best model and vectorizer saved successfully in 'models/' folder!")
