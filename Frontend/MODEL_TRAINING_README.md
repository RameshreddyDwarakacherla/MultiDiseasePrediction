# Disease Prediction Models - Training Guide

## Overview
This project has been modified to train your own machine learning models instead of using pre-trained models. The training system creates models for multiple diseases using synthetic datasets that simulate real medical data patterns.

## Trained Models

The following models are now trained from scratch:

1. **Symptom-based Disease Prediction** (XGBoost) - Accuracy: 100.00%
2. **Diabetes Prediction** (Random Forest) - Accuracy: 85.50%
3. **Heart Disease Prediction** (Random Forest) - Accuracy: 77.50%
4. **Parkinson's Disease Prediction** (SVM) - Accuracy: 66.50%
5. **Liver Disease Prediction** (Random Forest) - Accuracy: 99.50%
6. **Lung Cancer Prediction** (Random Forest) - Accuracy: 86.00%
7. **Hepatitis Prediction** (Random Forest) - Accuracy: 95.00%
8. **Chronic Kidney Disease Prediction** (Random Forest) - Accuracy: 86.00%
9. **Breast Cancer Prediction** (SVM) - Accuracy: 79.50%

## How to Train Models

### Method 1: Using the Training Script
```bash
cd Multiple-Disease-Prediction-Webapp/Frontend
python train_models.py
```

### Method 2: Using the Training Module Directly
```bash
cd Multiple-Disease-Prediction-Webapp/Frontend
python -c "from code.train import DiseaseModelTrainer; trainer = DiseaseModelTrainer(); trainer.train_all_models()"
```

## Training Process

### Data Generation
Since real medical datasets often have privacy restrictions, the training system generates synthetic datasets that:
- Follow realistic medical parameter distributions
- Include appropriate correlations between symptoms and diseases
- Maintain statistical properties similar to real medical data
- Ensure reproducible results with fixed random seeds

### Model Selection
Different algorithms are used for different diseases based on their characteristics:
- **XGBoost**: For multi-class symptom-based disease prediction
- **Random Forest**: For most binary classification tasks (robust and interpretable)
- **SVM**: For complex pattern recognition (Parkinson's and Breast Cancer)

### Training Features

#### Diabetes Model Features:
- Pregnancies, Glucose, Blood Pressure, Skin Thickness
- Insulin, BMI, Diabetes Pedigree Function, Age

#### Heart Disease Model Features:
- Age, Sex, Chest Pain Type, Resting Blood Pressure
- Cholesterol, Fasting Blood Sugar, Resting ECG
- Max Heart Rate, Exercise Induced Angina, ST Depression
- Slope, Number of Major Vessels, Thalassemia

#### Parkinson's Disease Model Features:
- Voice measurements (MDVP, Jitter, Shimmer, etc.)
- Harmonic-to-noise ratio, RPDE, DFA, Spread, PPE

#### Liver Disease Model Features:
- Age, Gender, Bilirubin levels, Enzyme levels
- Protein levels, Albumin ratios

#### Lung Cancer Model Features:
- Demographics, Smoking history, Symptoms
- Lifestyle factors, Physical symptoms

#### Hepatitis Model Features:
- Age, Sex, Liver function tests
- Enzyme levels, Protein markers

#### Chronic Kidney Disease Model Features:
- Age, Blood pressure, Specific gravity
- Albumin, Sugar, Blood cells, Bacteria
- Blood glucose, Urea, Creatinine, Electrolytes
- Hemoglobin, Cell counts, Comorbidities

#### Breast Cancer Model Features:
- Radius, Texture, Perimeter, Area
- Smoothness, Compactness, Concavity
- Concave points, Symmetry, Fractal dimension
- (Mean, SE, and Worst values for each)

## File Structure

```
Frontend/
├── models/                     # Trained model files (.sav)
├── model/                      # XGBoost model files (.json)
├── data/                       # Training datasets
├── code/
│   ├── train.py               # Main training module
│   ├── DiseaseModel.py        # Disease prediction class
│   └── helper.py              # Helper functions
├── train_models.py            # Training script
├── app.py                     # Main Streamlit application
└── MODEL_TRAINING_README.md   # This file
```

## Running the Application

After training the models, run the Streamlit application:

```bash
cd Multiple-Disease-Prediction-Webapp/Frontend
streamlit run app.py
```

## Customization

### Adding New Diseases
1. Create a synthetic data generation method in `code/train.py`
2. Add a training method for the new disease
3. Update the `train_all_models()` method
4. Add the prediction interface in `app.py`

### Modifying Existing Models
1. Edit the corresponding data generation method
2. Adjust feature distributions or risk calculations
3. Retrain the models using the training script

### Using Real Data
To use real medical datasets:
1. Replace synthetic data generation with real data loading
2. Ensure proper data preprocessing and feature engineering
3. Handle missing values and categorical encoding
4. Maintain the same feature names expected by the application

## Important Notes

- **Synthetic Data**: Current models use synthetic data for demonstration
- **Medical Disclaimer**: These models are for educational purposes only
- **Real Implementation**: For production use, replace with validated medical datasets
- **Privacy**: Ensure compliance with medical data privacy regulations
- **Validation**: Thoroughly validate models with real medical professionals

## Model Performance

The training system provides accuracy metrics for each model. These are based on synthetic data and should be validated with real medical data for production use.

## Troubleshooting

### Common Issues:
1. **Import Errors**: Ensure all required packages are installed
2. **Path Issues**: Run scripts from the Frontend directory
3. **Memory Issues**: Reduce dataset size if needed
4. **Model Loading**: Check that model files exist in the models/ directory

### Dependencies:
- pandas
- numpy
- scikit-learn
- xgboost
- joblib
- streamlit
- plotly
- matplotlib
- seaborn
