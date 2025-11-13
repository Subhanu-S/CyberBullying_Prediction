import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib

# Step 1: Load dataset
path = r"C:\Users\hp\OneDrive\Desktop\CyberbullyingDetection\data\text_data_clean.csv"
df = pd.read_csv(path)
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())

# Step 2: Choose correct columns (adjust if different)
X = df['text']
y = df['label']

# Step 3: Convert text to TF-IDF vectors
vectorizer = TfidfVectorizer(max_features=5000)
X_vect = vectorizer.fit_transform(X)
print("Text vectorization completed!")

# Step 4: Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_vect, y)
print("✅ Model training completed successfully!")

# Step 5: Save model & vectorizer
joblib.dump(model, "cyber_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("💾 Model and vectorizer saved!")

# Step 6: Test on sample comments
new_comments = [
    "You are such a loser!",
    "I appreciate your help!",
    "You are awesome!",
    "Nobody likes you!"
]
new_vect = vectorizer.transform(new_comments)
predictions = model.predict(new_vect)

print("\n🔍 Test Predictions:")
for comment, pred in zip(new_comments, predictions):
    label = "Bullying 😡" if pred == 1 else "Not Bullying 😊"
    print(f"{comment} --> {label}")
