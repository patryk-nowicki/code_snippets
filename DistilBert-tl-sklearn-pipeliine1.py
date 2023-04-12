from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

# Define column types
numeric_cols = ['num1', 'num2', 'num3']
categorical_cols = ['cat1', 'cat2', 'cat3']
text_cols = ['text1', 'text2']

# Define transformers
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

text_transformer = Pipeline(steps=[
    ('tfidf', TfidfVectorizer())
])

# Combine transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols),
        ('text', text_transformer, text_cols)
    ])

# Define model pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('clf', DecisionTreeClassifier())
])

# Fit the pipeline on training data and labels
pipeline.fit(X_train, y_train)

# Predict labels for test data
y_pred = pipeline.predict(X_test)
