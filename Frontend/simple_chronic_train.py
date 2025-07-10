import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import joblib
import json
import os

def create_chronic_kidney_data(n_samples=1000):
    """Create balanced synthetic chronic kidney disease dataset"""
    np.random.seed(42)
    
    # Create balanced dataset - 70% healthy, 30% with kidney disease
    n_healthy = int(n_samples * 0.7)
    n_diseased = n_samples - n_healthy
    
    # Initialize arrays
    age = np.zeros(n_samples)
    bp = np.zeros(n_samples)
    sg = np.zeros(n_samples)
    al = np.zeros(n_samples)
    su = np.zeros(n_samples)
    rbc = np.zeros(n_samples)
    pc = np.zeros(n_samples)
    pcc = np.zeros(n_samples)
    ba = np.zeros(n_samples)
    bgr = np.zeros(n_samples)
    bu = np.zeros(n_samples)
    sc = np.zeros(n_samples)
    sod = np.zeros(n_samples)
    pot = np.zeros(n_samples)
    hemo = np.zeros(n_samples)
    pcv = np.zeros(n_samples)
    wc = np.zeros(n_samples)
    rc = np.zeros(n_samples)
    htn = np.zeros(n_samples)
    dm = np.zeros(n_samples)
    cad = np.zeros(n_samples)
    appet = np.zeros(n_samples)
    pe = np.zeros(n_samples)
    ane = np.zeros(n_samples)
    
    # Generate healthy samples (first n_healthy samples)
    age[:n_healthy] = np.random.randint(20, 60, n_healthy)
    bp[:n_healthy] = np.random.randint(90, 140, n_healthy)
    sg[:n_healthy] = np.random.uniform(1.010, 1.025, n_healthy)
    al[:n_healthy] = np.random.choice([0, 1], n_healthy, p=[0.9, 0.1])
    su[:n_healthy] = np.random.choice([0, 1], n_healthy, p=[0.95, 0.05])
    rbc[:n_healthy] = np.random.choice([0, 1], n_healthy, p=[0.1, 0.9])
    pc[:n_healthy] = np.random.choice([0, 1], n_healthy, p=[0.9, 0.1])
    pcc[:n_healthy] = np.random.choice([0, 1], n_healthy, p=[0.95, 0.05])
    ba[:n_healthy] = np.random.choice([0, 1], n_healthy, p=[0.9, 0.1])
    bgr[:n_healthy] = np.random.randint(70, 120, n_healthy)
    bu[:n_healthy] = np.random.randint(10, 25, n_healthy)
    sc[:n_healthy] = np.random.uniform(0.5, 1.2, n_healthy)
    sod[:n_healthy] = np.random.randint(135, 145, n_healthy)
    pot[:n_healthy] = np.random.uniform(3.5, 5.0, n_healthy)
    hemo[:n_healthy] = np.random.uniform(12.0, 16.0, n_healthy)
    pcv[:n_healthy] = np.random.randint(35, 50, n_healthy)
    wc[:n_healthy] = np.random.randint(4000, 11000, n_healthy)
    rc[:n_healthy] = np.random.uniform(4.0, 6.0, n_healthy)
    htn[:n_healthy] = np.random.choice([0, 1], n_healthy, p=[0.8, 0.2])
    dm[:n_healthy] = np.random.choice([0, 1], n_healthy, p=[0.9, 0.1])
    cad[:n_healthy] = np.random.choice([0, 1], n_healthy, p=[0.95, 0.05])
    appet[:n_healthy] = np.random.choice([0, 1], n_healthy, p=[0.1, 0.9])
    pe[:n_healthy] = np.random.choice([0, 1], n_healthy, p=[0.95, 0.05])
    ane[:n_healthy] = np.random.choice([0, 1], n_healthy, p=[0.9, 0.1])
    
    # Generate diseased samples (last n_diseased samples)
    age[n_healthy:] = np.random.randint(45, 85, n_diseased)
    bp[n_healthy:] = np.random.randint(140, 200, n_diseased)
    sg[n_healthy:] = np.random.uniform(1.005, 1.015, n_diseased)
    al[n_healthy:] = np.random.choice([0, 1, 2, 3, 4], n_diseased, p=[0.2, 0.3, 0.3, 0.15, 0.05])
    su[n_healthy:] = np.random.choice([0, 1, 2, 3, 4], n_diseased, p=[0.4, 0.3, 0.2, 0.08, 0.02])
    rbc[n_healthy:] = np.random.choice([0, 1], n_diseased, p=[0.6, 0.4])
    pc[n_healthy:] = np.random.choice([0, 1], n_diseased, p=[0.3, 0.7])
    pcc[n_healthy:] = np.random.choice([0, 1], n_diseased, p=[0.5, 0.5])
    ba[n_healthy:] = np.random.choice([0, 1], n_diseased, p=[0.4, 0.6])
    bgr[n_healthy:] = np.random.randint(80, 300, n_diseased)
    bu[n_healthy:] = np.random.randint(25, 150, n_diseased)
    sc[n_healthy:] = np.random.uniform(1.5, 15.0, n_diseased)
    sod[n_healthy:] = np.random.randint(120, 150, n_diseased)
    pot[n_healthy:] = np.random.uniform(3.0, 7.0, n_diseased)
    hemo[n_healthy:] = np.random.uniform(6.0, 12.0, n_diseased)
    pcv[n_healthy:] = np.random.randint(15, 40, n_diseased)
    wc[n_healthy:] = np.random.randint(3000, 15000, n_diseased)
    rc[n_healthy:] = np.random.uniform(2.5, 5.0, n_diseased)
    htn[n_healthy:] = np.random.choice([0, 1], n_diseased, p=[0.3, 0.7])
    dm[n_healthy:] = np.random.choice([0, 1], n_diseased, p=[0.5, 0.5])
    cad[n_healthy:] = np.random.choice([0, 1], n_diseased, p=[0.7, 0.3])
    appet[n_healthy:] = np.random.choice([0, 1], n_diseased, p=[0.6, 0.4])
    pe[n_healthy:] = np.random.choice([0, 1], n_diseased, p=[0.4, 0.6])
    ane[n_healthy:] = np.random.choice([0, 1], n_diseased, p=[0.3, 0.7])
    
    # Create classification labels
    classification = np.zeros(n_samples)
    classification[n_healthy:] = 1
    
    # Shuffle the data
    indices = np.random.permutation(n_samples)
    
    data = pd.DataFrame({
        'age': age[indices], 'bp': bp[indices], 'sg': sg[indices], 'al': al[indices], 'su': su[indices],
        'rbc': rbc[indices], 'pc': pc[indices], 'pcc': pcc[indices], 'ba': ba[indices], 'bgr': bgr[indices],
        'bu': bu[indices], 'sc': sc[indices], 'sod': sod[indices], 'pot': pot[indices], 'hemo': hemo[indices],
        'pcv': pcv[indices], 'wc': wc[indices], 'rc': rc[indices], 'htn': htn[indices], 'dm': dm[indices],
        'cad': cad[indices], 'appet': appet[indices], 'pe': pe[indices], 'ane': ane[indices],
        'classification': classification[indices]
    })
    
    return data

def train_model():
    print("Creating synthetic chronic kidney disease dataset...")
    data = create_chronic_kidney_data(2000)
    print(f"Dataset created with {len(data)} samples")
    print(f"Class distribution: {data['classification'].value_counts().to_dict()}")
    
    X = data.drop('classification', axis=1)
    y = data['classification']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training Random Forest model...")
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
    
    # Create models directory if it doesn't exist
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
    
    return accuracy

if __name__ == "__main__":
    train_model()
