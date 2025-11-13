import pandas as pd
import glob
import os

# Step 1: Find all CSV files except already merged ones
files = [f for f in glob.glob("data/*.csv") if not f.endswith(("text_data.csv", "text_data_clean.csv"))]

print("Found files:")
for f in files:
    print(" -", os.path.basename(f))

dfs = []

for file in files:
    try:
        df = pd.read_csv(file, encoding='utf-8', low_memory=False)
        print(f"\nLoaded {os.path.basename(file)} with {df.shape[0]} rows and {df.shape[1]} columns.")

        text_col = None
        label_col = None

        # Priority order for text columns
        possible_text_cols = ['Text', 'text', 'tweet', 'comment', 'CONTENT', 'content', 'body', 'comment_text']
        for col in possible_text_cols:
            if col in df.columns:
                text_col = col
                break

        # Priority order for label columns
        possible_label_cols = ['oh_label', 'label', 'class', 'toxic', 'target', 'annotation']
        for col in possible_label_cols:
            if col in df.columns:
                label_col = col
                break

        if text_col and label_col:
            df = df[[text_col, label_col]]
            df.columns = ['text', 'label']
            dfs.append(df)
            print(f"✅ Selected columns from {os.path.basename(file)} → text: {text_col}, label: {label_col}")
        else:
            print(f"⚠️ Skipped {os.path.basename(file)} (no valid text/label columns found)")

    except Exception as e:
        print(f"❌ Error reading {os.path.basename(file)}: {e}")

# Merge all valid datasets
if not dfs:
    raise Exception("No valid data found to merge!")

merged = pd.concat(dfs, ignore_index=True)
print("\n✅ Total combined rows before cleaning:", merged.shape)

# Remove rows where text is missing or numeric
merged['text'] = merged['text'].astype(str)
merged = merged[~merged['text'].str.isnumeric()]  # remove numbers-only
merged = merged[merged['text'].str.len() > 2]
merged = merged.dropna(subset=['text', 'label'])

# Save final merged dataset
merged.to_csv("data/text_data.csv", index=False)
print("✅ Final merged dataset saved as data/text_data.csv with shape:", merged.shape)
