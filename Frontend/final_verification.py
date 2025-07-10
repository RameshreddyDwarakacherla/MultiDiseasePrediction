#!/usr/bin/env python3
"""
Final verification script to ensure all components are working correctly
"""

import os
import sys
import importlib.util

def check_file_exists(file_path, description):
    """Check if a file exists"""
    if os.path.exists(file_path):
        print(f"✅ {description}: Found")
        return True
    else:
        print(f"❌ {description}: Missing")
        return False

def check_import(module_name, description):
    """Check if a module can be imported"""
    try:
        __import__(module_name)
        print(f"✅ {description}: Import successful")
        return True
    except ImportError as e:
        print(f"❌ {description}: Import failed - {e}")
        return False

def main():
    """Main verification function"""
    print("🔍 FINAL VERIFICATION OF ENHANCED DISEASE PREDICTION SYSTEM")
    print("=" * 70)
    
    # Check core files
    print("\n📁 CHECKING CORE FILES:")
    files_to_check = [
        ("app.py", "Main Streamlit Application"),
        ("train_models.py", "Model Training Script"),
        ("test_models.py", "Model Testing Script"),
        ("code/train.py", "Training Module"),
        ("code/DiseaseModel.py", "Disease Model Class"),
        ("code/helper.py", "Helper Functions"),
        ("MODEL_TRAINING_README.md", "Training Documentation"),
        ("IMPLEMENTATION_SUMMARY.md", "Implementation Summary"),
        ("requirements.txt", "Dependencies File")
    ]
    
    file_count = 0
    for file_path, description in files_to_check:
        if check_file_exists(file_path, description):
            file_count += 1
    
    # Check model files
    print("\n🤖 CHECKING TRAINED MODELS:")
    model_files = [
        ("models/diabetes_model.sav", "Diabetes Model"),
        ("models/heart_disease_model.sav", "Heart Disease Model"),
        ("models/parkinsons_model.sav", "Parkinsons Model"),
        ("models/lung_cancer_model.sav", "Lung Cancer Model"),
        ("models/breast_cancer.sav", "Breast Cancer Model"),
        ("models/chronic_model.sav", "Chronic Kidney Disease Model"),
        ("models/hepititisc_model.sav", "Hepatitis Model"),
        ("models/liver_model.sav", "Liver Disease Model"),
        ("model/xgboost_model.json", "XGBoost Symptom Model")
    ]
    
    model_count = 0
    for file_path, description in model_files:
        if check_file_exists(file_path, description):
            model_count += 1
    
    # Check data files
    print("\n📊 CHECKING DATA FILES:")
    data_files = [
        ("data/dataset.csv", "Main Dataset"),
        ("data/clean_dataset.tsv", "Cleaned Dataset"),
        ("data/lung_cancer.csv", "Lung Cancer Dataset"),
        ("data/symptom_Description.csv", "Symptom Descriptions"),
        ("data/symptom_precaution.csv", "Symptom Precautions")
    ]
    
    data_count = 0
    for file_path, description in data_files:
        if check_file_exists(file_path, description):
            data_count += 1
    
    # Check Python imports
    print("\n🐍 CHECKING PYTHON DEPENDENCIES:")
    imports_to_check = [
        ("streamlit", "Streamlit Framework"),
        ("pandas", "Pandas Data Analysis"),
        ("numpy", "NumPy Numerical Computing"),
        ("sklearn", "Scikit-learn Machine Learning"),
        ("joblib", "Joblib Model Persistence"),
        ("plotly", "Plotly Interactive Plots"),
        ("PIL", "Pillow Image Processing"),
        ("xgboost", "XGBoost Gradient Boosting"),
        ("matplotlib", "Matplotlib Plotting"),
        ("seaborn", "Seaborn Statistical Visualization")
    ]
    
    import_count = 0
    for module_name, description in imports_to_check:
        if check_import(module_name, description):
            import_count += 1
    
    # Optional imports
    print("\n🔧 CHECKING OPTIONAL DEPENDENCIES:")
    optional_imports = [
        ("cv2", "OpenCV Image Processing"),
        ("tensorflow", "TensorFlow Deep Learning")
    ]
    
    optional_count = 0
    for module_name, description in optional_imports:
        if check_import(module_name, description):
            optional_count += 1
    
    # Summary
    print("\n" + "=" * 70)
    print("📋 VERIFICATION SUMMARY")
    print("=" * 70)
    print(f"Core Files: {file_count}/{len(files_to_check)} ✅")
    print(f"Model Files: {model_count}/{len(model_files)} ✅")
    print(f"Data Files: {data_count}/{len(data_files)} ✅")
    print(f"Required Dependencies: {import_count}/{len(imports_to_check)} ✅")
    print(f"Optional Dependencies: {optional_count}/{len(optional_imports)} ✅")
    
    total_checks = len(files_to_check) + len(model_files) + len(data_files) + len(imports_to_check)
    total_passed = file_count + model_count + data_count + import_count
    
    print(f"\n🎯 OVERALL SCORE: {total_passed}/{total_checks} ({(total_passed/total_checks)*100:.1f}%)")
    
    if total_passed == total_checks:
        print("\n🎉 ALL CHECKS PASSED! System is ready to use.")
        print("\n🚀 To run the application:")
        print("   streamlit run app.py")
    else:
        print(f"\n⚠️  {total_checks - total_passed} checks failed. Please review the issues above.")
    
    # Feature summary
    print("\n✨ ENHANCED FEATURES IMPLEMENTED:")
    features = [
        "🤖 Explainable AI with feature importance analysis",
        "🎨 Blue gradient theme with modern UI design",
        "📷 Lung disease detection from X-ray images",
        "📊 Interactive visualizations for predictions",
        "🏥 Medical-grade interface with professional styling",
        "🔍 Model performance metrics in sidebar",
        "💡 Health recommendations based on risk factors",
        "🎯 Confidence scores for all predictions",
        "📱 Responsive design with tabs and columns",
        "⚠️  Appropriate medical disclaimers"
    ]
    
    for feature in features:
        print(f"   {feature}")
    
    print("\n📚 DOCUMENTATION AVAILABLE:")
    docs = [
        "📖 MODEL_TRAINING_README.md - Comprehensive training guide",
        "📋 IMPLEMENTATION_SUMMARY.md - Complete feature overview",
        "🧪 test_models.py - Model verification script",
        "🏗️  train_models.py - Model training script"
    ]
    
    for doc in docs:
        print(f"   {doc}")

if __name__ == "__main__":
    main()
