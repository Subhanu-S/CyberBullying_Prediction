import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression  # or use RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Step 1: Load dataset
df = pd.read_csv("data/text_data_clean.csv")
print("Dataset loaded successfully:", df.shape)

X = df["text"]
y = df["label"]

# Step 2: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 3: Convert text into TF-IDF features
print("Vectorizing text data...")
vectorizer = TfidfVectorizer(max_features=3000, stop_words="english")
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)
print("✅ Vectorization complete!")

# Step 4: Train model (you can use LogisticRegression or RandomForest)
model = LogisticRegression(max_iter=1000)
print("Training model... Please wait ⏳")
model.fit(X_train_vect, y_train)

# Step 5: Evaluate
y_pred = model.predict(X_test_vect)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model trained successfully! Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 6: Save model and vectorizer
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/cyber_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")
print("\n✅ Model and vectorizer saved successfully in 'models/' folder.")
