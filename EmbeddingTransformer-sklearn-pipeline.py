import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from transformers import DistilBertTokenizer, TFDistilBertModel

# Define custom transformer to extract DistilBERT embeddings
class EmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.distilbert = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        encodings = self.tokenizer(X['text'].tolist(), truncation=True, padding=True)
        embeddings = self.distilbert.predict(encodings)['last_hidden_state'][:, 0, :]
        return embeddings

# Load data
df = pd.read_csv('data.csv')

# Define pipeline
pipeline = Pipeline([
    ('embeddings', EmbeddingTransformer()),
    ('clf', DecisionTreeClassifier())
])

# Train pipeline
pipeline.fit(df[['text']], df['label'])

# Predict labels for new data
y_pred = pipeline.predict(new_data[['text']])
