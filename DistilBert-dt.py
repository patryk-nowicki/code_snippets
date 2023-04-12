import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, TFDistilBertModel
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# Load data
df = pd.read_csv('data.csv')

# Split into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Initialize tokenizer and encode text data
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
train_encodings = tokenizer(train_df['text'].tolist(), truncation=True, padding=True)
test_encodings = tokenizer(test_df['text'].tolist(), truncation=True, padding=True)

# Convert labels to integers
label2id = {'negative': 0, 'positive': 1}
train_labels = [label2id[label] for label in train_df['label'].tolist()]
test_labels = [label2id[label] for label in test_df['label'].tolist()]

# Load pre-trained DistilBERT model
distilbert = TFDistilBertModel.from_pretrained('distilbert-base-uncased')

# Extract DistilBERT embeddings
train_embeddings = distilbert.predict(train_encodings)['last_hidden_state'][:, 0, :]
test_embeddings = distilbert.predict(test_encodings)['last_hidden_state'][:, 0, :]

# Train decision tree on embeddings
clf = DecisionTreeClassifier()
clf.fit(train_embeddings, train_labels)

# Evaluate decision tree on test set
y_pred = clf.predict(test_embeddings)
print(classification_report(test_labels, y_pred))
