from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from category_encoders import TopKEncoder
from sklearn.preprocessing import OneHotEncoder

# Define the numerical, categorical, and text columns
numerical_cols = ['age', 'income']
categorical_cols = ['gender', 'education']
text_cols = ['description']

# Create transformers for each column type
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('topk', TopKEncoder(k=10)),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
])

text_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='')),
    ('tfidf_vectorizer', TfidfVectorizer())
])

# Create a ColumnTransformer to apply the appropriate transformer to each column
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols),
        ('text', text_transformer, text_cols)
    ])

# Create the final pipeline by chaining the preprocessor with the classifier
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    # Add any additional steps, such as a classifier
])
