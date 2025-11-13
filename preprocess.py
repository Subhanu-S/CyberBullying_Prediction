import pandas as pd
import re

# Step 1: Load merged dataset
df = pd.read_csv("data/text_data.csv", encoding='utf-8')
print("Loaded merged dataset:", df.shape)
print("Columns:", df.columns.tolist())

# Step 2: Detect proper text and label columns automatically
text_col = None
label_col = None

for col in df.columns:
    if any(k in col.lower() for k in ['text', 'tweet', 'comment', 'content', 'body']):
        text_col = col
    if any(k in col.lower() for k in ['label', 'class', 'toxic', 'target']):
        label_col = col

print("Detected text column:", text_col)
print("Detected label column:", label_col)

# Step 3: Keep only those columns
if text_col and label_col:
    df = df[[text_col, label_col]]
    df.columns = ['text', 'label']
else:
    raise Exception("⚠️ Could not find text/label columns — check your CSV structure!")

# Step 4: Clean the text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df['text'] = df['text'].apply(clean_text)

# Step 5: Drop blanks or short rows
df = df.dropna()
df = df[df['text'].str.len() > 1]

print("After cleaning:", df.shape)

# Step 6: Save cleaned dataset
df.to_csv("data/text_data_clean.csv", index=False)
print("✅ Cleaned dataset saved as data/text_data_clean.csv")
