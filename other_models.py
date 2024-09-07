import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import log_loss, accuracy_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import numpy as np
from scipy.stats import randint, uniform

class CrimeClassifier:
    def __init__(self, train):
        self.train = train
        self.train_labels = self.train['Category']
        self.train = self.train.drop(['Descript', 'Resolution', 'Category'], axis=1)
        
        self.models = {
            "XGB": XGBClassifier(),
            "RF": RandomForestClassifier(random_state=42),
            "NB": GaussianNB(),
            "MLP": MLPClassifier(random_state=42)
        }
        
        self.param_distributions = {
            "XGB": {
                'max_depth': randint(3, 10),
                'learning_rate': uniform(0.01, 0.3),
                'n_estimators': randint(50, 200)
            },
            "RF": {
                'n_estimators': randint(50, 200),
                'max_features': ['auto', 'sqrt', 'log2'],
                'max_depth': [None, 10, 20, 30]
            },
            "MLP": {
                'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                'activation': ['tanh', 'relu'],
                'solver': ['sgd', 'adam'],
                'alpha': uniform(0.0001, 0.05),
                'learning_rate': ['constant', 'adaptive']
            }
        }
        
        self.results = {}
        
    def preprocessing(self, data):
        scaler = StandardScaler()
        data[["X", "Y"]] = scaler.fit_transform(data[["X", "Y"]])
        data['Dates'] = pd.to_datetime(data['Dates'])
        data['Year'] = data['Dates'].dt.year
        data['Month'] = data['Dates'].dt.month
        data['Day'] = data['Dates'].dt.day
        data['Hour'] = data['Dates'].dt.hour
        data['Minute'] = data['Dates'].dt.minute
        data = data.drop(['Dates', 'Address'], axis=1)
        data['PdDistrict'] = data['PdDistrict'].astype('category').cat.codes
        data['DayOfWeek'] = data['DayOfWeek'].astype('category').cat.codes
        return data
    
    def train_or_fine_tune_model(self, model_name, n_iter=0, n_jobs=-1):
        print(f'\nFine-tuning {model_name}...\n')
        model = self.models[model_name]
        param_distribution = self.param_distributions.get(model_name, {})
        
        label_encoded_y = LabelEncoder().fit_transform(self.train_labels)
        X_train, X_val, y_train, y_val = train_test_split(self.train, label_encoded_y, test_size=0.15, random_state=42)
        
        if n_iter > 0 and param_distribution is not None:
            random_search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_distribution,
                n_iter=n_iter,
                cv=3,
                scoring='neg_log_loss',
                verbose=1, n_jobs=n_jobs,
                random_state=42)
            random_search.fit(X_train, y_train)
            best_model = random_search.best_estimator_
            self.models[model_name] = best_model  # Update the model with the best found parameters
            print(f'Best parameters for {model_name}: {random_search.best_params_}')
        else:
            model.fit(X_train, y_train)
            best_model = model
        
        predicted_labels = best_model.predict(X_val)
        accuracy = accuracy_score(y_val, predicted_labels) * 100
        predicted_probs = best_model.predict_proba(X_val)
        log_loss_score = log_loss(y_val, predicted_probs)
        
        self.results[model_name] = {
            'accuracy': accuracy,
            'log_loss': log_loss_score
        }
        
        print(f'{model_name} Fine-Tuned Accuracy: {accuracy:.2f}%')
        print(f'{model_name} Fine-Tuned Log Loss: {log_loss_score:.4f}')
        print('\n' + '#' * 100 + '\n')
    
    def compare_models(self):
        print("\nModel Comparison:\n")
        for model_name, result in self.results.items():
            print(f"Model: {model_name}")
            print(f"Accuracy: {result['accuracy']:.2f}%")
            print(f"Log Loss: {result['log_loss']:.4f}")
            print('-' * 30)

    def run(self, n_iter=0, n_jobs=-1):
        self.train = self.preprocessing(self.train)
        for model_name in self.models:
            self.train_or_fine_tune_model(model_name, n_iter, n_jobs)
        self.compare_models()
