#!/usr/bin/env python3
"""
Enhanced training script that saves comprehensive metrics for all models
"""

import sys
import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from code.train import DiseaseModelTrainer

class EnhancedModelTrainer(DiseaseModelTrainer):
    """Enhanced trainer that saves comprehensive metrics"""
    
    def calculate_comprehensive_metrics(self, y_true, y_pred, model_name):
        """Calculate and save comprehensive metrics"""
        try:
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            metrics = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'model_type': model_name
            }
            
            print(f"{model_name} Comprehensive Metrics:")
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1 Score:  {f1:.4f}")
            
            return metrics
        except Exception as e:
            print(f"Error calculating metrics for {model_name}: {e}")
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'model_type': model_name
            }
    
    def train_diabetes_model_enhanced(self):
        """Train diabetes model with comprehensive metrics"""
        print("Training Enhanced Diabetes Prediction Model...")
        
        data = self.create_synthetic_diabetes_data()
        X = data.drop('Outcome', axis=1)
        y = data['Outcome']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        metrics = self.calculate_comprehensive_metrics(y_test, preds, "Diabetes")
        
        # Save model and metrics
        joblib.dump(model, f'{self.models_dir}/diabetes_model.sav')
        with open(f'{self.models_dir}/diabetes_metrics.json', 'w') as f:
            json.dump(metrics, f)
        
        return metrics
    
    def train_heart_model_enhanced(self):
        """Train heart disease model with comprehensive metrics"""
        print("Training Enhanced Heart Disease Prediction Model...")
        
        data = self.create_synthetic_heart_data()
        X = data.drop('target', axis=1)
        y = data['target']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        metrics = self.calculate_comprehensive_metrics(y_test, preds, "Heart Disease")
        
        joblib.dump(model, f'{self.models_dir}/heart_disease_model.sav')
        with open(f'{self.models_dir}/heart_metrics.json', 'w') as f:
            json.dump(metrics, f)
        
        return metrics
    
    def train_parkinsons_model_enhanced(self):
        """Train Parkinson's model with comprehensive metrics"""
        print("Training Enhanced Parkinson's Disease Prediction Model...")
        
        data = self.create_synthetic_parkinsons_data()
        X = data.drop('status', axis=1)
        y = data['status']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = SVC(kernel='rbf', random_state=42, probability=True)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        metrics = self.calculate_comprehensive_metrics(y_test, preds, "Parkinson's Disease")
        
        joblib.dump(model, f'{self.models_dir}/parkinsons_model.sav')
        with open(f'{self.models_dir}/parkinsons_metrics.json', 'w') as f:
            json.dump(metrics, f)
        
        return metrics
    
    def train_liver_model_enhanced(self):
        """Train liver disease model with comprehensive metrics"""
        print("Training Enhanced Liver Disease Prediction Model...")
        
        data = self.create_synthetic_liver_data()
        X = data.drop('Dataset', axis=1)
        y = data['Dataset']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        metrics = self.calculate_comprehensive_metrics(y_test, preds, "Liver Disease")
        
        joblib.dump(model, f'{self.models_dir}/liver_model.sav')
        with open(f'{self.models_dir}/liver_metrics.json', 'w') as f:
            json.dump(metrics, f)
        
        return metrics
    
    def train_lung_cancer_model_enhanced(self):
        """Train lung cancer model with comprehensive metrics"""
        print("Training Enhanced Lung Cancer Prediction Model...")
        
        data = self.create_synthetic_lung_cancer_data()
        X = data.drop('LUNGCANCER', axis=1)
        y = data['LUNGCANCER']
        
        # Encode categorical variables
        from sklearn.preprocessing import LabelEncoder
        le_gender = LabelEncoder()
        X['GENDER'] = le_gender.fit_transform(X['GENDER'])
        
        # Encode target variable
        le_target = LabelEncoder()
        y_encoded = le_target.fit_transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        metrics = self.calculate_comprehensive_metrics(y_test, preds, "Lung Cancer")
        
        joblib.dump(model, f'{self.models_dir}/lung_cancer_model.sav')
        joblib.dump(le_target, f'{self.models_dir}/lung_cancer_encoder.pkl')
        with open(f'{self.models_dir}/lung_cancer_metrics.json', 'w') as f:
            json.dump(metrics, f)
        
        return metrics
    
    def train_hepatitis_model_enhanced(self):
        """Train hepatitis model with comprehensive metrics"""
        print("Training Enhanced Hepatitis Prediction Model...")
        
        data = self.create_synthetic_hepatitis_data()
        X = data.drop('Category', axis=1)
        y = data['Category']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        metrics = self.calculate_comprehensive_metrics(y_test, preds, "Hepatitis")
        
        joblib.dump(model, f'{self.models_dir}/hepititisc_model.sav')
        with open(f'{self.models_dir}/hepatitis_metrics.json', 'w') as f:
            json.dump(metrics, f)
        
        return metrics
    
    def train_chronic_kidney_model_enhanced(self):
        """Train chronic kidney disease model with comprehensive metrics"""
        print("Training Enhanced Chronic Kidney Disease Prediction Model...")
        
        data = self.create_synthetic_chronic_kidney_data()
        X = data.drop('classification', axis=1)
        y = data['classification']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        metrics = self.calculate_comprehensive_metrics(y_test, preds, "Chronic Kidney Disease")
        
        joblib.dump(model, f'{self.models_dir}/chronic_model.sav')
        with open(f'{self.models_dir}/chronic_metrics.json', 'w') as f:
            json.dump(metrics, f)
        
        return metrics
    
    def train_breast_cancer_model_enhanced(self):
        """Train breast cancer model with comprehensive metrics"""
        print("Training Enhanced Breast Cancer Prediction Model...")
        
        data = self.create_synthetic_breast_cancer_data()
        X = data.drop('diagnosis', axis=1)
        y = data['diagnosis']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = SVC(kernel='rbf', random_state=42, probability=True)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        metrics = self.calculate_comprehensive_metrics(y_test, preds, "Breast Cancer")
        
        joblib.dump(model, f'{self.models_dir}/breast_cancer.sav')
        with open(f'{self.models_dir}/breast_cancer_metrics.json', 'w') as f:
            json.dump(metrics, f)
        
        return metrics
    
    def train_all_models_enhanced(self):
        """Train all models with comprehensive metrics"""
        print("\n===== TRAINING ALL MODELS WITH COMPREHENSIVE METRICS =====\n")
        
        # Train the main symptom-based disease model
        symptom_accuracy = self.train_symptom_based_disease_model()
        
        # Train individual disease models with metrics
        diabetes_metrics = self.train_diabetes_model_enhanced()
        heart_metrics = self.train_heart_model_enhanced()
        parkinsons_metrics = self.train_parkinsons_model_enhanced()
        liver_metrics = self.train_liver_model_enhanced()
        lung_cancer_metrics = self.train_lung_cancer_model_enhanced()
        hepatitis_metrics = self.train_hepatitis_model_enhanced()
        chronic_kidney_metrics = self.train_chronic_kidney_model_enhanced()
        breast_cancer_metrics = self.train_breast_cancer_model_enhanced()
        
        # Save comprehensive summary
        all_metrics = {
            'symptom_based': {'accuracy': symptom_accuracy, 'model_type': 'XGBoost'},
            'diabetes': diabetes_metrics,
            'heart_disease': heart_metrics,
            'parkinsons': parkinsons_metrics,
            'liver_disease': liver_metrics,
            'lung_cancer': lung_cancer_metrics,
            'hepatitis': hepatitis_metrics,
            'chronic_kidney': chronic_kidney_metrics,
            'breast_cancer': breast_cancer_metrics
        }
        
        with open(f'{self.models_dir}/all_metrics_summary.json', 'w') as f:
            json.dump(all_metrics, f, indent=2)
        
        print("\n===== COMPREHENSIVE METRICS SUMMARY =====")
        for disease, metrics in all_metrics.items():
            if isinstance(metrics, dict) and 'accuracy' in metrics:
                print(f"{disease.replace('_', ' ').title()}:")
                print(f"  Accuracy:  {metrics.get('accuracy', 0):.4f}")
                if 'precision' in metrics:
                    print(f"  Precision: {metrics.get('precision', 0):.4f}")
                    print(f"  Recall:    {metrics.get('recall', 0):.4f}")
                    print(f"  F1 Score:  {metrics.get('f1_score', 0):.4f}")
                print()
        
        print("All models trained with comprehensive metrics saved!")
        return all_metrics

def main():
    """Main function to train all models with metrics"""
    print("Starting enhanced model training with comprehensive metrics...")
    
    trainer = EnhancedModelTrainer()
    trainer.train_all_models_enhanced()
    
    print("\nEnhanced model training completed!")
    print("All models now include comprehensive metrics (accuracy, precision, recall, F1-score)")

if __name__ == "__main__":
    main()
