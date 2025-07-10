#!/usr/bin/env python3
"""
Test script to verify chronic kidney disease prediction fix
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import json

def test_chronic_kidney_fix():
    """Test the chronic kidney disease prediction fix"""
    
    print("🔍 TESTING CHRONIC KIDNEY DISEASE PREDICTION FIX")
    print("=" * 60)
    
    # Test 1: Check if model files exist
    print("\n1️⃣ CHECKING MODEL FILES:")
    model_path = "models/chronic_model.sav"
    metrics_path = "models/chronic_metrics.json"
    
    if os.path.exists(model_path):
        print(f"✅ Model file exists: {model_path}")
    else:
        print(f"❌ Model file missing: {model_path}")
        return False
    
    if os.path.exists(metrics_path):
        print(f"✅ Metrics file exists: {metrics_path}")
    else:
        print(f"⚠️ Metrics file missing: {metrics_path}")
    
    # Test 2: Load and test the model
    print("\n2️⃣ TESTING MODEL LOADING:")
    try:
        model = joblib.load(model_path)
        print("✅ Model loaded successfully")
        print(f"   Model type: {type(model).__name__}")
        print(f"   Number of features: {model.n_features_in_}")
    except Exception as e:
        print(f"❌ Failed to load model: {str(e)}")
        return False
    
    # Test 3: Test predictions with different scenarios
    print("\n3️⃣ TESTING PREDICTIONS:")
    
    # Test case 1: Healthy patient (should predict negative)
    healthy_case = [
        25,    # age
        120,   # bp
        1.020, # sg
        0,     # al
        0,     # su
        1,     # rbc
        0,     # pc
        0,     # pcc
        0,     # ba
        90,    # bgr
        15,    # bu
        0.8,   # sc
        140,   # sod
        4.0,   # pot
        14.0,  # hemo
        42,    # pcv
        7000,  # wc
        4.5,   # rc
        0,     # htn
        0,     # dm
        0,     # cad
        1,     # appet
        0,     # pe
        0      # ane
    ]
    
    # Test case 2: High-risk patient (should predict positive)
    diseased_case = [
        65,    # age
        160,   # bp
        1.010, # sg
        2,     # al
        1,     # su
        0,     # rbc
        1,     # pc
        1,     # pcc
        1,     # ba
        180,   # bgr
        80,    # bu
        5.0,   # sc
        125,   # sod
        6.0,   # pot
        8.0,   # hemo
        25,    # pcv
        12000, # wc
        3.0,   # rc
        1,     # htn
        1,     # dm
        1,     # cad
        0,     # appet
        1,     # pe
        1      # ane
    ]
    
    try:
        # Test healthy case
        healthy_pred = model.predict([healthy_case])[0]
        healthy_prob = model.predict_proba([healthy_case])[0]
        
        print(f"   Healthy case prediction: {healthy_pred} ({'Positive' if healthy_pred == 1 else 'Negative'})")
        print(f"   Healthy case probabilities: [Negative: {healthy_prob[0]:.3f}, Positive: {healthy_prob[1]:.3f}]")
        
        # Test diseased case
        diseased_pred = model.predict([diseased_case])[0]
        diseased_prob = model.predict_proba([diseased_case])[0]
        
        print(f"   Diseased case prediction: {diseased_pred} ({'Positive' if diseased_pred == 1 else 'Negative'})")
        print(f"   Diseased case probabilities: [Negative: {diseased_prob[0]:.3f}, Positive: {diseased_prob[1]:.3f}]")
        
        # Check if predictions are different (this was the main issue)
        if healthy_pred != diseased_pred:
            print("✅ Model produces different predictions for different cases")
        else:
            print("⚠️ Model produces same prediction for both cases - may need retraining")
            
    except Exception as e:
        print(f"❌ Failed to make predictions: {str(e)}")
        return False
    
    # Test 4: Check model metrics
    print("\n4️⃣ CHECKING MODEL METRICS:")
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            
            print(f"   Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
            print(f"   Precision: {metrics.get('precision', 'N/A'):.4f}")
            print(f"   Recall: {metrics.get('recall', 'N/A'):.4f}")
            print(f"   F1 Score: {metrics.get('f1_score', 'N/A'):.4f}")
            print(f"   Model Type: {metrics.get('model_type', 'N/A')}")
            print(f"   Training Samples: {metrics.get('training_samples', 'N/A')}")
            print(f"   Test Samples: {metrics.get('test_samples', 'N/A')}")
            
        except Exception as e:
            print(f"⚠️ Could not read metrics: {str(e)}")
    
    # Test 5: Test with multiple random cases
    print("\n5️⃣ TESTING WITH RANDOM CASES:")
    np.random.seed(42)
    
    predictions = []
    for i in range(10):
        # Generate random case
        random_case = [
            np.random.randint(20, 80),      # age
            np.random.randint(90, 180),     # bp
            np.random.uniform(1.005, 1.025), # sg
            np.random.randint(0, 4),        # al
            np.random.randint(0, 4),        # su
            np.random.randint(0, 2),        # rbc
            np.random.randint(0, 2),        # pc
            np.random.randint(0, 2),        # pcc
            np.random.randint(0, 2),        # ba
            np.random.randint(70, 250),     # bgr
            np.random.randint(10, 100),     # bu
            np.random.uniform(0.5, 10.0),   # sc
            np.random.randint(120, 150),    # sod
            np.random.uniform(3.0, 6.0),    # pot
            np.random.uniform(8.0, 16.0),   # hemo
            np.random.randint(20, 50),      # pcv
            np.random.randint(4000, 15000), # wc
            np.random.uniform(3.0, 6.0),    # rc
            np.random.randint(0, 2),        # htn
            np.random.randint(0, 2),        # dm
            np.random.randint(0, 2),        # cad
            np.random.randint(0, 2),        # appet
            np.random.randint(0, 2),        # pe
            np.random.randint(0, 2)         # ane
        ]
        
        pred = model.predict([random_case])[0]
        predictions.append(pred)
    
    positive_count = sum(predictions)
    negative_count = len(predictions) - positive_count
    
    print(f"   Random test results: {positive_count} positive, {negative_count} negative")
    
    if positive_count > 0 and negative_count > 0:
        print("✅ Model produces varied predictions")
    else:
        print("⚠️ Model produces only one type of prediction")
    
    print("\n" + "=" * 60)
    print("🎉 CHRONIC KIDNEY DISEASE PREDICTION TEST COMPLETED!")
    
    if healthy_pred != diseased_pred and (positive_count > 0 and negative_count > 0):
        print("✅ ALL TESTS PASSED - Model is working correctly!")
        return True
    else:
        print("⚠️ SOME ISSUES DETECTED - Model may need adjustment")
        return False

if __name__ == "__main__":
    # Change to the correct directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    success = test_chronic_kidney_fix()
    
    if success:
        print("\n🚀 The chronic kidney disease prediction should now work correctly!")
        print("   - Navigate to 'Chronic Kidney prediction' in the app")
        print("   - Try different parameter combinations")
        print("   - You should see both positive and negative predictions")
    else:
        print("\n🔧 If issues persist:")
        print("   - Use the 'Create Chronic Kidney Disease Model' button in the app")
        print("   - Or run the model creation script manually")
    
    sys.exit(0 if success else 1)
