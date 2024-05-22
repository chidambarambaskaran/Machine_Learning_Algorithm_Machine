import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib

def preprocess_data(df, task='classification', test_size=0.2, random_state=42):
    X = df.iloc[:, :-1]  
    y = df.iloc[:, -1]
    
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler() if task == 'regression' else MinMaxScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    if task == 'classification' and y_train.dtype == 'object':
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)
    
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)
    
    return X_train, X_test, y_train, y_test, preprocessor

def train_and_evaluate_models(X_train, X_test, y_train, y_test, task='classification', algorithm='Logistic Regression'):
    if task == 'classification':
        if algorithm == 'Logistic Regression':
            model = LogisticRegression(max_iter=1000, random_state=42)
        elif algorithm == 'Random Forest Classifier':
            model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=2, random_state=42)
        elif algorithm == 'Support Vector Classifier':
            model = SVC(C=1.0, kernel='rbf', random_state=42)
        elif algorithm == 'Gradient Boosting Classifier':
            model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        elif algorithm == 'Decision Tree Classifier':
            model = DecisionTreeClassifier(max_depth=5, random_state=42)
        else:
            return None, None
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return model, accuracy

    elif task == 'regression':
        if algorithm == 'Linear Regression':
            model = LinearRegression()
        elif algorithm == 'Random Forest Regressor':
            model = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=2, random_state=42)
        elif algorithm == 'Support Vector Regressor':
            model = SVR(C=1.0, kernel='rbf')
        elif algorithm == 'Gradient Boosting Regressor':
            model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        elif algorithm == 'Decision Tree Regressor':
            model = DecisionTreeRegressor(max_depth=5, random_state=42)
        else:
            return None, None
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        threshold = 0.1 
        
        correct_predictions = np.abs(y_test - y_pred) <= threshold * np.abs(y_test)
        accuracy = np.mean(correct_predictions)
        
        return model, accuracy

def save_model(model, preprocessor, filename='model.pkl'):
    joblib.dump({'model': model, 'preprocessor': preprocessor}, filename)
    print(f'Model saved to {filename}')

def load_model(filename='model.pkl'):
    data = joblib.load(filename)
    return data['model'], data['preprocessor']