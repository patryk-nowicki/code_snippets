import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from transformers import DistilBertTokenizer, TFDistilBertModel

# Define custom transformers for preprocessing
class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X = X.str.lower()  # Convert text to lowercase
        return X
    
# Load data
df = pd.read_csv('data.csv')

# Define column transformer
text_transformer = Pipeline(steps=[
    ('preprocess', TextPreprocessor()),
    ('tokenizer', DistilBertTokenizer.from_pretrained('distilbert-base-uncased')),
    ('embedding', TFDistilBertModel.from_pretrained('distilbert-base-uncased'))
])

numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('text', text_transformer, 'text_column'),
        ('num', numerical_transformer, ['numerical_column_1', 'numerical_column_2']),
        ('cat', categorical_transformer, 'categorical_column')
    ])

# Define pipeline
pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('clf', DecisionTreeClassifier())
])

# Train pipeline
X = df[['text_column', 'numerical_column_1', 'numerical_column_2', 'categorical_column']]
y = df['label']
pipeline.fit(X, y)

# Predict labels for new data
new_data = pd.DataFrame({'text_column': ['new text', 'more text'], 
                         'numerical_column_1': [1, 2],
                         'numerical_column_2': [3, 4],
                         'categorical_column': ['cat1', 'cat2']})
y_pred = pipeline.predict(new_data)
