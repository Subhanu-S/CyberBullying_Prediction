import joblib

# Load the saved model and vectorizer
print("Loading model...")
model = joblib.load("models/cyber_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")
print("✅ Model loaded successfully!\n")

# Test comments
new_comments = [
    "I hate you, you are so dumb!",
    "Have a nice day!",
    "You’re such an idiot!"
]

# Transform the comments
new_vect = vectorizer.transform(new_comments)
predictions = model.predict(new_vect)

# Print results in a friendly way
for comment, pred in zip(new_comments, predictions):
    result = "Cyberbullying 😡" if pred == 1 else "Not Bullying 😊"
    print(f"Comment: {comment}\nPrediction: {result}\n")
