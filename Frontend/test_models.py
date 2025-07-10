#!/usr/bin/env python3
"""
Test script to verify all trained models are working correctly
"""

import joblib
import os
import numpy as np
import pandas as pd

def test_models():
    """Test all trained models"""
    print("🔍 Testing All Trained Models")
    print("=" * 50)
    
    models_to_test = {
        'diabetes_model.sav': 'Diabetes Model',
        'heart_disease_model.sav': 'Heart Disease Model',
        'parkinsons_model.sav': 'Parkinsons Model',
        'lung_cancer_model.sav': 'Lung Cancer Model',
        'breast_cancer.sav': 'Breast Cancer Model',
        'chronic_model.sav': 'Chronic Kidney Disease Model',
        'hepititisc_model.sav': 'Hepatitis Model',
        'liver_model.sav': 'Liver Disease Model'
    }
    
    results = {}
    
    for model_file, model_name in models_to_test.items():
        model_path = f"models/{model_file}"
        
        print(f"\n📊 Testing {model_name}...")
        
        # Check if model file exists
        if not os.path.exists(model_path):
            print(f"❌ {model_name}: Model file not found!")
            results[model_name] = "File Missing"
            continue
        
        try:
            # Load the model
            model = joblib.load(model_path)
            print(f"✅ {model_name}: Model loaded successfully")
            
            # Test prediction with dummy data
            if 'diabetes' in model_file:
                # Test diabetes model (8 features)
                test_data = np.array([[1, 120, 80, 20, 100, 25.0, 0.5, 30]])
                prediction = model.predict(test_data)
                print(f"   Test prediction: {prediction[0]}")
                
            elif 'heart' in model_file:
                # Test heart model (13 features)
                test_data = np.array([[50, 1, 0, 120, 200, 0, 0, 150, 0, 1.0, 0, 0, 0]])
                prediction = model.predict(test_data)
                print(f"   Test prediction: {prediction[0]}")
                
            elif 'parkinsons' in model_file:
                # Test parkinsons model (22 features)
                test_data = np.array([[150, 200, 120, 0.005, 0.00001, 0.003, 0.003, 0.01, 0.03, 0.3, 0.02, 0.03, 0.04, 0.06, 0.02, 20, 0.5, 0.7, -5, 0.2, 2.5, 0.2]])
                prediction = model.predict(test_data)
                print(f"   Test prediction: {prediction[0]}")
                
            elif 'lung_cancer' in model_file:
                # Test lung cancer model (15 features)
                test_data = np.array([[0, 50, 2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1]])
                prediction = model.predict(test_data)
                print(f"   Test prediction: {prediction[0]}")
                
            elif 'breast_cancer' in model_file:
                # Test breast cancer model (30 features)
                test_data = np.random.rand(1, 30)
                prediction = model.predict(test_data)
                print(f"   Test prediction: {prediction[0]}")
                
            elif 'chronic' in model_file:
                # Test chronic kidney model (24 features)
                test_data = np.array([[50, 120, 1.02, 0, 0, 1, 1, 0, 0, 120, 30, 1.2, 140, 4.0, 12, 40, 8000, 4.5, 0, 0, 0, 1, 0, 0]])
                prediction = model.predict(test_data)
                print(f"   Test prediction: {prediction[0]}")
                
            elif 'hepatitis' in model_file:
                # Test hepatitis model (12 features)
                test_data = np.array([[50, 1, 40, 100, 30, 30, 1.0, 8, 4, 80, 30, 70]])
                prediction = model.predict(test_data)
                print(f"   Test prediction: {prediction[0]}")
                
            elif 'liver' in model_file:
                # Test liver model (10 features)
                test_data = np.array([[0, 50, 1.0, 0.3, 100, 30, 30, 7.0, 4.0, 1.2]])
                prediction = model.predict(test_data)
                print(f"   Test prediction: {prediction[0]}")
            
            results[model_name] = "✅ Working"
            
        except Exception as e:
            print(f"❌ {model_name}: Error - {str(e)}")
            results[model_name] = f"Error: {str(e)}"
    
    # Print summary
    print("\n" + "=" * 50)
    print("📋 MODEL TEST SUMMARY")
    print("=" * 50)
    
    working_count = 0
    for model_name, status in results.items():
        print(f"{model_name}: {status}")
        if "Working" in status:
            working_count += 1
    
    print(f"\n🎯 {working_count}/{len(results)} models are working correctly!")
    
    # Test XGBoost model
    print("\n🔍 Testing XGBoost Symptom-based Model...")
    xgb_model_path = "model/xgboost_model.json"
    if os.path.exists(xgb_model_path):
        print("✅ XGBoost model file found")
    else:
        print("❌ XGBoost model file not found")
    
    return results

if __name__ == "__main__":
    test_models()
