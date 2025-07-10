#!/usr/bin/env python3
"""
Create a balanced chronic kidney disease model
"""

# Step 1: Create synthetic data
import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Create 1000 samples with 30% positive cases
n_samples = 1000
n_positive = 300
n_negative = 700

print("Creating synthetic chronic kidney disease dataset...")

# Initialize feature arrays
features = {}

# Create negative cases (healthy patients)
print("Generating healthy patient data...")
features['age'] = list(np.random.randint(20, 60, n_negative)) + list(np.random.randint(50, 85, n_positive))
features['bp'] = list(np.random.randint(90, 140, n_negative)) + list(np.random.randint(140, 200, n_positive))
features['sg'] = list(np.random.uniform(1.015, 1.025, n_negative)) + list(np.random.uniform(1.005, 1.015, n_positive))
features['al'] = list(np.random.choice([0, 1], n_negative, p=[0.9, 0.1])) + list(np.random.choice([0, 1, 2, 3, 4], n_positive, p=[0.2, 0.3, 0.3, 0.15, 0.05]))
features['su'] = list(np.random.choice([0, 1], n_negative, p=[0.95, 0.05])) + list(np.random.choice([0, 1, 2, 3, 4], n_positive, p=[0.4, 0.3, 0.2, 0.08, 0.02]))
features['rbc'] = list(np.random.choice([0, 1], n_negative, p=[0.1, 0.9])) + list(np.random.choice([0, 1], n_positive, p=[0.6, 0.4]))
features['pc'] = list(np.random.choice([0, 1], n_negative, p=[0.9, 0.1])) + list(np.random.choice([0, 1], n_positive, p=[0.3, 0.7]))
features['pcc'] = list(np.random.choice([0, 1], n_negative, p=[0.95, 0.05])) + list(np.random.choice([0, 1], n_positive, p=[0.5, 0.5]))
features['ba'] = list(np.random.choice([0, 1], n_negative, p=[0.9, 0.1])) + list(np.random.choice([0, 1], n_positive, p=[0.4, 0.6]))
features['bgr'] = list(np.random.randint(70, 120, n_negative)) + list(np.random.randint(80, 300, n_positive))
features['bu'] = list(np.random.randint(10, 25, n_negative)) + list(np.random.randint(25, 150, n_positive))
features['sc'] = list(np.random.uniform(0.5, 1.2, n_negative)) + list(np.random.uniform(1.5, 15.0, n_positive))
features['sod'] = list(np.random.randint(135, 145, n_negative)) + list(np.random.randint(120, 150, n_positive))
features['pot'] = list(np.random.uniform(3.5, 5.0, n_negative)) + list(np.random.uniform(3.0, 7.0, n_positive))
features['hemo'] = list(np.random.uniform(12.0, 16.0, n_negative)) + list(np.random.uniform(6.0, 12.0, n_positive))
features['pcv'] = list(np.random.randint(35, 50, n_negative)) + list(np.random.randint(15, 40, n_positive))
features['wc'] = list(np.random.randint(4000, 11000, n_negative)) + list(np.random.randint(3000, 15000, n_positive))
features['rc'] = list(np.random.uniform(4.0, 6.0, n_negative)) + list(np.random.uniform(2.5, 5.0, n_positive))
features['htn'] = list(np.random.choice([0, 1], n_negative, p=[0.8, 0.2])) + list(np.random.choice([0, 1], n_positive, p=[0.3, 0.7]))
features['dm'] = list(np.random.choice([0, 1], n_negative, p=[0.9, 0.1])) + list(np.random.choice([0, 1], n_positive, p=[0.5, 0.5]))
features['cad'] = list(np.random.choice([0, 1], n_negative, p=[0.95, 0.05])) + list(np.random.choice([0, 1], n_positive, p=[0.7, 0.3]))
features['appet'] = list(np.random.choice([0, 1], n_negative, p=[0.1, 0.9])) + list(np.random.choice([0, 1], n_positive, p=[0.6, 0.4]))
features['pe'] = list(np.random.choice([0, 1], n_negative, p=[0.95, 0.05])) + list(np.random.choice([0, 1], n_positive, p=[0.4, 0.6]))
features['ane'] = list(np.random.choice([0, 1], n_negative, p=[0.9, 0.1])) + list(np.random.choice([0, 1], n_positive, p=[0.3, 0.7]))

# Create labels
labels = [0] * n_negative + [1] * n_positive

# Create DataFrame
data = pd.DataFrame(features)
data['classification'] = labels

# Shuffle the data
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Dataset created with {len(data)} samples")
print(f"Class distribution: {data['classification'].value_counts().to_dict()}")

# Step 2: Train the model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import json
import os

print("Training Random Forest model...")

X = data.drop('classification', axis=1)
y = data['classification']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train model with balanced class weights
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced',
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2
)

model.fit(X_train, y_train)

# Make predictions
preds = model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, preds)
precision = precision_score(y_test, preds, average='weighted')
recall = recall_score(y_test, preds, average='weighted')
f1 = f1_score(y_test, preds, average='weighted')

print(f"Model Performance:")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall: {recall:.4f}")
print(f"  F1 Score: {f1:.4f}")

# Test with some sample predictions
print("\nTesting model predictions:")
test_cases = [
    # Healthy case
    [25, 120, 1.020, 0, 0, 1, 0, 0, 0, 90, 15, 0.8, 140, 4.0, 14.0, 42, 7000, 4.5, 0, 0, 0, 1, 0, 0],
    # Disease case  
    [65, 160, 1.010, 2, 1, 0, 1, 1, 1, 180, 80, 5.0, 125, 6.0, 8.0, 25, 12000, 3.0, 1, 1, 1, 0, 1, 1]
]

for i, case in enumerate(test_cases):
    pred = model.predict([case])[0]
    prob = model.predict_proba([case])[0]
    print(f"Test case {i+1}: Prediction = {pred}, Probabilities = {prob}")

# Step 3: Save model and metrics
os.makedirs('models', exist_ok=True)

# Save model
joblib.dump(model, 'models/chronic_model.sav')
print("Model saved to models/chronic_model.sav")

# Save metrics
metrics = {
    'accuracy': float(accuracy),
    'precision': float(precision),
    'recall': float(recall),
    'f1_score': float(f1),
    'model_type': 'Random Forest',
    'features': list(X.columns),
    'training_samples': len(X_train),
    'test_samples': len(X_test)
}

with open('models/chronic_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print("Metrics saved to models/chronic_metrics.json")

print("\n✅ Chronic kidney disease model created successfully!")
print("The model should now provide balanced predictions.")
