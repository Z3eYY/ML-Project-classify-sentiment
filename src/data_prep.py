import pandas as pd
import re
from sklearn.model_selection import train_test_split
columns = ['sentiment', 'id', 'date', 'query', 'user', 'text']
file_path = 'data/training.1600000.processed.noemoticon.csv'
data = pd.read_csv(file_path, encoding='latin-1', names=columns)

# Keep only the relevant columns
data = data[['sentiment', 'text']]

# Convert sentiment labels  Simplify it to 0 (negative) and 1 (positive) for consistency.
data['sentiment'] = data['sentiment'].replace({4: 1})

def clean_text(text):
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+", "", text)    # Remove mentions
    text = re.sub(r"#", "", text)       # Remove hashtags symbol
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove punctuation and numbers
    text = re.sub(r"\s+", " ", text).strip() # Remove extra whitespace
    return text

data['text'] = data['text'].apply(clean_text)

# Drop rows with missing text
data.dropna(subset=['text'], inplace=True)

# Drop duplicate rows
data.drop_duplicates(inplace=True)

#conver text to lower
data['text'] = data['text'].str.lower()
data = data.sample(frac=1, random_state=28)
reduced_data = data.sample(frac=0.1, random_state=42)
cleaned_file_path = 'data/cleaned_data.csv'
reduced_data.to_csv(cleaned_file_path, index=True, index_label="index")



train_data, temp_data = train_test_split(reduced_data, test_size=0.3, random_state=38)

val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=38)


# Recheck and drop any NaN values after splitting
train_data.dropna(subset=['text'], inplace=True)
val_data.dropna(subset=['text'], inplace=True)
test_data.dropna(subset=['text'], inplace=True)

# Ensure all text values are strings (if necessary)
train_data['text'] = train_data['text'].astype(str)
val_data['text'] = val_data['text'].astype(str)
test_data['text'] = test_data['text'].astype(str)



# Remove rows with empty strings or only whitespace
train_data = train_data[train_data['text'].str.strip() != '']
val_data = val_data[val_data['text'].str.strip() != '']
test_data = test_data[test_data['text'].str.strip() != '']




train_data.to_csv('data/train_data.csv',index_label="index")
val_data.to_csv('data/val_data.csv',index_label="index")
test_data.to_csv('data/test_data.csv',index_label="index")


print("Data successfully split and saved:")
print(f"Training data: {len(train_data)} samples")
print(f"Validation data: {len(val_data)} samples")
print(f"Test data: {len(test_data)} samples")






