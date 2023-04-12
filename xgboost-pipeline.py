import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the data into a Pandas DataFrame
data = pd.read_csv('your_data.csv')

# Define the categorical and numerical features
cat_features = ['cat_feature1', 'cat_feature2', 'cat_feature3']
num_features = ['num_feature1', 'num_feature2', 'num_feature3']

# Define the preprocessing steps for categorical and numerical features
cat_pipeline = Pipeline([
    ('encoder', LabelEncoder())
])

num_pipeline = Pipeline([
    ('scaler', StandardScaler())
])

# Combine the preprocessing steps into a ColumnTransformer
preprocessor = ColumnTransformer([
    ('cat', cat_pipeline, cat_features),
    ('num', num_pipeline, num_features)
])

# Define the XGBoost model
model = xgb.XGBClassifier()

# Combine the preprocessing and model steps into a Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', model)
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop('target', axis=1), 
                                                    data['target'], 
                                                    test_size=0.2, 
                                                    random_state=42)

# Define the hyperparameter search space
param_grid = {
    'model__n_estimators': [100, 500, 1000],
    'model__max_depth': [3, 5, 7],
    'model__learning_rate': [0.01, 0.1, 0.5]
}

# Define the GridSearchCV object
grid_search = GridSearchCV(pipeline, 
                           param_grid=param_grid, 
                           cv=5, 
                           n_jobs=-1)

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train, y_train)

# Make predictions on the testing set using the best model found by GridSearchCV
y_pred = grid_search.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Print the best hyperparameters found by GridSearchCV
print(grid_search.best_params_)
