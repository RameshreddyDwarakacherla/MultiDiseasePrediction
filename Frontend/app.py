import streamlit as st
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
from streamlit_option_menu import option_menu
import pickle
from PIL import Image
import numpy as np
import plotly.figure_factory as ff
import streamlit as st
from code.DiseaseModel import DiseaseModel
from code.helper import prepare_symptoms_array
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import cv2
import json
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

# Set page config with blue theme
st.set_page_config(
    page_title="Multiple Disease Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for blue background and styling
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    .stSelectbox > div > div {
        background-color: rgba(255, 255, 255, 0.1);
        color: white;
    }
    .stNumberInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.1);
        color: white;
    }
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.1);
        color: white;
    }
    .stMultiSelect > div > div {
        background-color: rgba(255, 255, 255, 0.1);
    }
    h1, h2, h3 {
        color: white !important;
    }
    .stButton > button {
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
        color: white;
        border: none;
        border-radius: 20px;
        font-weight: bold;
    }
    .stSuccess {
        background-color: rgba(0, 255, 0, 0.1);
        border: 1px solid #00ff00;
        border-radius: 10px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Load model metrics
def load_model_metrics(disease_name):
    """Load comprehensive metrics for a specific disease model"""
    try:
        metrics_file = f"models/{disease_name}_metrics.json"
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                return json.load(f)
        return None
    except Exception as e:
        st.error(f"Error loading metrics for {disease_name}: {e}")
        return None

def display_model_metrics(metrics, disease_name):
    """Display comprehensive model metrics in a beautiful format"""
    if metrics:
        st.markdown(f"### 📊 {disease_name} Model Performance")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="🎯 Accuracy",
                value=f"{metrics.get('accuracy', 0):.3f}",
                delta=f"{metrics.get('accuracy', 0)*100:.1f}%"
            )

        with col2:
            st.metric(
                label="🔍 Precision",
                value=f"{metrics.get('precision', 0):.3f}",
                delta=f"{metrics.get('precision', 0)*100:.1f}%"
            )

        with col3:
            st.metric(
                label="📈 Recall",
                value=f"{metrics.get('recall', 0):.3f}",
                delta=f"{metrics.get('recall', 0)*100:.1f}%"
            )

        with col4:
            st.metric(
                label="⚖️ F1 Score",
                value=f"{metrics.get('f1_score', 0):.3f}",
                delta=f"{metrics.get('f1_score', 0)*100:.1f}%"
            )

        # Create a beautiful metrics chart
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            'Score': [
                metrics.get('accuracy', 0),
                metrics.get('precision', 0),
                metrics.get('recall', 0),
                metrics.get('f1_score', 0)
            ]
        })

        fig = px.bar(
            metrics_df,
            x='Metric',
            y='Score',
            title=f"{disease_name} Model Performance Metrics",
            color='Score',
            color_continuous_scale='viridis',
            text='Score'
        )
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            title_font_color='white',
            showlegend=False
        )
        fig.update_layout(yaxis=dict(range=[0, 1]))

        st.plotly_chart(fig, use_container_width=True)

# Enhanced Explainable AI Functions
def explain_prediction_advanced(model, X, feature_names, input_values, prediction_type="classification"):
    """Generate advanced feature importance explanation for predictions"""
    try:
        if hasattr(model, 'feature_importances_'):
            # For tree-based models
            importances = model.feature_importances_

            # Calculate feature contributions based on input values and importance
            contributions = []
            for i, (feature, importance, value) in enumerate(zip(feature_names, importances, input_values[0])):
                # Normalize contribution based on value and importance
                contribution = importance * abs(value) if value != 0 else importance * 0.1
                contributions.append(contribution)

            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importances,
                'input_value': input_values[0],
                'contribution': contributions,
                'risk_level': ['High' if c > np.mean(contributions) else 'Medium' if c > np.mean(contributions)*0.5 else 'Low' for c in contributions]
            }).sort_values('contribution', ascending=False)

            return feature_importance.head(10)
        else:
            # For other models, use permutation importance
            from sklearn.inspection import permutation_importance
            perm_importance = permutation_importance(model, X, np.zeros(X.shape[0]), n_repeats=3, random_state=42)

            contributions = []
            for i, (importance, value) in enumerate(zip(perm_importance.importances_mean, input_values[0])):
                contribution = importance * abs(value) if value != 0 else importance * 0.1
                contributions.append(contribution)

            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': perm_importance.importances_mean,
                'input_value': input_values[0],
                'contribution': contributions,
                'risk_level': ['High' if c > np.mean(contributions) else 'Medium' if c > np.mean(contributions)*0.5 else 'Low' for c in contributions]
            }).sort_values('contribution', ascending=False)

            return feature_importance.head(10)
    except Exception as e:
        st.error(f"Error generating explanation: {str(e)}")
        return None

def plot_feature_importance_advanced(feature_importance_df, title="Feature Importance Analysis"):
    """Plot advanced feature importance with risk levels using plotly"""
    if feature_importance_df is not None:
        # Create color mapping for risk levels
        color_map = {'High': '#ff4444', 'Medium': '#ffaa00', 'Low': '#44ff44'}
        feature_importance_df['color'] = feature_importance_df['risk_level'].map(color_map)

        fig = px.bar(
            feature_importance_df.head(10),
            x='contribution',
            y='feature',
            orientation='h',
            title=title,
            color='risk_level',
            color_discrete_map=color_map,
            hover_data=['importance', 'input_value', 'contribution'],
            text='contribution'
        )

        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            title_font_color='white',
            xaxis_title="Contribution to Prediction",
            yaxis_title="Features",
            legend_title="Risk Level"
        )
        return fig
    return None

def display_risk_factors_analysis(feature_importance_df, disease_name):
    """Display detailed risk factors analysis"""
    if feature_importance_df is not None:
        st.markdown(f"### 🔍 {disease_name} Risk Factors Analysis")

        # Top risk factors
        high_risk_factors = feature_importance_df[feature_importance_df['risk_level'] == 'High']
        medium_risk_factors = feature_importance_df[feature_importance_df['risk_level'] == 'Medium']
        low_risk_factors = feature_importance_df[feature_importance_df['risk_level'] == 'Low']

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### 🔴 High Risk Factors")
            if not high_risk_factors.empty:
                for _, factor in high_risk_factors.iterrows():
                    st.markdown(f"**{factor['feature']}**")
                    st.markdown(f"Value: {factor['input_value']:.2f}")
                    st.markdown(f"Contribution: {factor['contribution']:.3f}")
                    st.markdown("---")
            else:
                st.info("No high-risk factors identified")

        with col2:
            st.markdown("#### 🟡 Medium Risk Factors")
            if not medium_risk_factors.empty:
                for _, factor in medium_risk_factors.iterrows():
                    st.markdown(f"**{factor['feature']}**")
                    st.markdown(f"Value: {factor['input_value']:.2f}")
                    st.markdown(f"Contribution: {factor['contribution']:.3f}")
                    st.markdown("---")
            else:
                st.info("No medium-risk factors identified")

        with col3:
            st.markdown("#### 🟢 Low Risk Factors")
            if not low_risk_factors.empty:
                for _, factor in low_risk_factors.iterrows():
                    st.markdown(f"**{factor['feature']}**")
                    st.markdown(f"Value: {factor['input_value']:.2f}")
                    st.markdown(f"Contribution: {factor['contribution']:.3f}")
                    st.markdown("---")
            else:
                st.info("No low-risk factors identified")



def create_chronic_kidney_model():
    """Create a balanced chronic kidney disease model"""
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    import json
    import os

    # Set random seed for reproducibility
    np.random.seed(42)

    # Create 2000 samples with 30% positive cases
    n_samples = 2000
    n_positive = 600
    n_negative = 1400

    # Initialize feature arrays
    features = {}

    # Create negative cases (healthy patients)
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

    # Save model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/chronic_model.sav')

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

    return model, accuracy


# loading the models
diabetes_model = joblib.load("models/diabetes_model.sav")
heart_model = joblib.load("models/heart_disease_model.sav")
parkinson_model = joblib.load("models/parkinsons_model.sav")

# Load chronic kidney model with error handling
try:
    chronic_disease_model = joblib.load('models/chronic_model.sav')
except FileNotFoundError:
    # Use diabetes model as temporary fallback
    chronic_disease_model = joblib.load("models/diabetes_model.sav")
    # We'll show a message in the chronic kidney section to create the proper model

hepatitis_model = joblib.load('models/hepititisc_model.sav')
liver_model = joblib.load('models/liver_model.sav')


# sidebar
with st.sidebar:
    st.markdown("### 🏥 Multiple Disease Prediction System")
    st.markdown("---")

    selected = option_menu('Disease Prediction Menu', [
        'Disease Prediction',
        'Diabetes Prediction',
        'Heart disease Prediction',
        'Parkison Prediction',
        'Liver prediction',
        'Hepatitis prediction',
        'Chronic Kidney prediction',
    ],
        icons=['🔍','🩺', '❤️', '🧠','🫁','🦠','🫘'],
        default_index=0,
        styles={
            "container": {"padding": "5!important", "background-color": "rgba(255,255,255,0.1)"},
            "icon": {"color": "white", "font-size": "18px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "color": "white"},
            "nav-link-selected": {"background-color": "#4facfe"},
        })

    st.markdown("---")

    # Explainable AI Section
    st.markdown("### 🤖 Explainable AI Features")
    show_explanation = st.checkbox("Show Feature Importance", value=True)
    explanation_type = st.selectbox(
        "Explanation Type",
        ["Feature Importance", "Top Contributing Factors"],
        help="Choose how to explain the model's predictions"
    )

    st.markdown("---")

    # Model Information
    st.markdown("### 📊 Model Information")
    st.info("""
    **All models are trained from scratch using:**
    - Random Forest (Most diseases)
    - XGBoost (Symptom-based)
    - SVM (Parkinson's)
    """)

    st.markdown("### 🎯 Model Performance Metrics")

    # Load comprehensive metrics
    try:
        with open('models/all_metrics_summary.json', 'r') as f:
            all_metrics = json.load(f)

        # Display metrics in expandable sections
        for disease_key, metrics in all_metrics.items():
            if isinstance(metrics, dict) and 'accuracy' in metrics:
                disease_name = disease_key.replace('_', ' ').title()

                with st.expander(f"📊 {disease_name} Metrics"):
                    if 'precision' in metrics:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
                            st.metric("Precision", f"{metrics['precision']:.3f}")
                        with col2:
                            st.metric("Recall", f"{metrics['recall']:.3f}")
                            st.metric("F1 Score", f"{metrics['f1_score']:.3f}")
                    else:
                        st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
                        st.info("XGBoost model - Comprehensive metrics available in main interface")

    except FileNotFoundError:
        # Fallback to basic accuracies
        st.markdown("#### Basic Accuracies")
        accuracies = {
            "Symptom-based": "100.0%",
            "Diabetes": "85.5%",
            "Heart Disease": "77.5%",
            "Parkinson's": "66.5%",
            "Liver Disease": "99.5%",
            "Hepatitis": "95.0%",
            "Chronic Kidney": "86.0%"
        }

        for disease, accuracy in accuracies.items():
            st.metric(disease, accuracy)




# multiple disease prediction
if selected == 'Disease Prediction': 
    # Create disease class and load ML model
    disease_model = DiseaseModel()
    disease_model.load_xgboost('model/xgboost_model.json')

    # Title
    st.write('# Disease Prediction using Machine Learning')

    symptoms = st.multiselect('What are your symptoms?', options=disease_model.all_symptoms)

    X = prepare_symptoms_array(symptoms)

    # Trigger XGBoost model
    if st.button('Predict'): 
        # Run the model with the python script
        
        prediction, prob = disease_model.predict(X)
        st.write(f'## Disease: {prediction} with {prob*100:.2f}% probability')


        tab1, tab2= st.tabs(["Description", "Precautions"])

        with tab1:
            st.write(disease_model.describe_predicted_disease())

        with tab2:
            precautions = disease_model.predicted_disease_precautions()
            for i in range(4):
                st.write(f'{i+1}. {precautions[i]}')




# Diabetes prediction page
if selected == 'Diabetes Prediction':  # pagetitle
    st.title("Diabetes Disease Prediction")

    # Create two columns for layout
    main_col, image_col = st.columns([2, 1])

    with image_col:
        image = Image.open('d3.jpg')
        st.image(image, caption='Diabetes Disease Prediction')

    with main_col:
        st.markdown("### Enter Patient Information")
        name = st.text_input("Patient Name:")

    # Input parameters
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, help="Number of times pregnant")
    with col2:
        Glucose = st.number_input("Glucose Level", min_value=0, max_value=200, help="Plasma glucose concentration (mg/dL)")
    with col3:
        BloodPressure = st.number_input("Blood Pressure", min_value=0, max_value=150, help="Diastolic blood pressure (mm Hg)")

    with col1:
        SkinThickness = st.number_input("Skin Thickness", min_value=0, max_value=100, help="Triceps skin fold thickness (mm)")
    with col2:
        Insulin = st.number_input("Insulin Level", min_value=0, max_value=900, help="2-Hour serum insulin (mu U/ml)")
    with col3:
        BMI = st.number_input("BMI Value", min_value=0.0, max_value=70.0, help="Body mass index (weight in kg/(height in m)^2)")

    with col1:
        DiabetesPedigreefunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, help="Diabetes pedigree function (genetic influence)")
    with col2:
        Age = st.number_input("Age", min_value=0, max_value=120, help="Age in years")

    # code for prediction
    diabetes_dig = ''

    # button
    if st.button("Predict Diabetes Risk"):
        # Create input array
        input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreefunction, Age]])

        # Make prediction
        diabetes_prediction = diabetes_model.predict(input_data)

        # Get probability if available
        try:
            diabetes_prob = diabetes_model.predict_proba(input_data)[0][1]
            probability_text = f" (Confidence: {diabetes_prob*100:.2f}%)"
        except:
            probability_text = ""

        # Display result
        if diabetes_prediction[0] == 1:
            diabetes_dig = f"We are sorry to inform you that you may have Diabetes{probability_text}."
            image = Image.open('positive.jpg')
            st.image(image, caption='')
        else:
            diabetes_dig = f"Good news! You likely don't have Diabetes{probability_text}."
            image = Image.open('negative.jpg')
            st.image(image, caption='')

        st.success(f"{name}, {diabetes_dig}")

        # Display comprehensive model metrics
        diabetes_metrics = load_model_metrics("diabetes")
        if diabetes_metrics:
            display_model_metrics(diabetes_metrics, "Diabetes")

        # Show explainable AI if enabled
        if show_explanation:
            st.markdown("### 🤖 AI Explanation: Understanding Your Results")

            # Feature names for diabetes model
            feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

            # Get advanced feature importance
            feature_importance = explain_prediction_advanced(diabetes_model, input_data, feature_names, input_data)

            # Plot advanced feature importance
            if feature_importance is not None:
                st.markdown("#### 📊 AI Analysis: Which Parameters Cause High Risk")
                fig = plot_feature_importance_advanced(feature_importance, "Diabetes Risk Factors Analysis")
                st.plotly_chart(fig, use_container_width=True)

                # Display detailed risk factors analysis
                display_risk_factors_analysis(feature_importance, "Diabetes")

                # Display top contributing factors with detailed explanation
                st.markdown("#### 🎯 Top 5 Contributing Factors:")
                top_factors = feature_importance.head(5)
                for i, (_, row) in enumerate(top_factors.iterrows()):
                    risk_emoji = "🔴" if row['risk_level'] == 'High' else "🟡" if row['risk_level'] == 'Medium' else "🟢"
                    st.markdown(f"{risk_emoji} **{i+1}. {row['feature']}** ({row['risk_level']} Risk)")
                    st.markdown(f"   - Your Value: {row['input_value']:.2f}")
                    st.markdown(f"   - Contribution Score: {row['contribution']:.4f}")
                    st.markdown(f"   - Model Importance: {row['importance']:.4f}")

                # Enhanced health recommendations based on risk analysis
                st.markdown("### 💡 Personalized Health Recommendations:")
                high_risk_features = feature_importance[feature_importance['risk_level'] == 'High']['feature'].tolist()

                if 'Glucose' in high_risk_features:
                    st.error("🩸 **CRITICAL: High Glucose Level** - Immediate medical consultation recommended for blood sugar management")
                elif 'Glucose' in top_factors['feature'].values:
                    st.warning("📊 **Monitor your glucose levels** - Consider regular blood sugar testing and dietary adjustments")

                if 'BMI' in high_risk_features:
                    st.error("⚖️ **CRITICAL: High BMI** - Urgent lifestyle changes needed - consult a nutritionist")
                elif 'BMI' in top_factors['feature'].values:
                    st.info("⚖️ **Maintain a healthy weight** - Focus on balanced nutrition and regular exercise")

                if 'Age' in high_risk_features:
                    st.warning("🕰️ **Age-related risk** - More frequent health screenings recommended")
                elif 'Age' in top_factors['feature'].values:
                    st.info("🕰️ **Age is a factor** - Regular check-ups become more important as you age")

                if 'DiabetesPedigreeFunction' in high_risk_features:
                    st.warning("👪 **Strong family history** - Genetic predisposition requires careful monitoring")
                elif 'DiabetesPedigreeFunction' in top_factors['feature'].values:
                    st.info("👪 **Family history matters** - Inform your doctor about your family's diabetes history")

                if 'Insulin' in high_risk_features:
                    st.error("💉 **CRITICAL: Insulin resistance** - Endocrinologist consultation recommended")

                # Overall risk assessment
                high_risk_count = len(high_risk_features)
                if high_risk_count >= 3:
                    st.error("⚠️ **HIGH OVERALL RISK** - Multiple critical factors identified. Immediate medical attention recommended.")
                elif high_risk_count >= 1:
                    st.warning("⚠️ **MODERATE RISK** - Some concerning factors identified. Schedule a medical check-up.")
                else:
                    st.success("✅ **LOW RISK** - Most factors are within acceptable ranges. Continue healthy lifestyle.")
        
        



# Heart prediction page
if selected == 'Heart disease Prediction':
    st.title("Heart Disease Prediction")

    # Create two columns for layout
    main_col, image_col = st.columns([2, 1])

    with image_col:
        image = Image.open('heart2.jpg')
        st.image(image, caption='Heart Disease Analysis')

    with main_col:
        st.markdown("### Enter Cardiac Assessment Data")
        name = st.text_input("Patient Name:")

    # Create tabs for better organization
    input_tab, info_tab = st.tabs(["Patient Data Input", "Parameter Information"])

    with input_tab:
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input("Age", min_value=20, max_value=100, value=45)
        with col2:
            sex = 0
            gender = st.radio("Gender", ["Female", "Male"])
            if gender == "Male":
                sex = 1
            else:
                sex = 0
        with col3:
            cp = 0
            chest_pain_types = {
                "Typical Angina": 0,
                "Atypical Angina": 1,
                "Non-anginal Pain": 2,
                "Asymptomatic": 3
            }
            cp_selection = st.selectbox("Chest Pain Type", list(chest_pain_types.keys()))
            cp = chest_pain_types[cp_selection]

        with col1:
            trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=90, max_value=200, value=120)
        with col2:
            chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
        with col3:
            fbs = 0
            fbs_selection = st.radio("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
            if fbs_selection == "Yes":
                fbs = 1

        with col1:
            restecg_options = {
                "Normal": 0,
                "ST-T Wave Abnormality": 1,
                "Left Ventricular Hypertrophy": 2
            }
            restecg_selection = st.selectbox("Resting ECG", list(restecg_options.keys()))
            restecg = restecg_options[restecg_selection]

        with col2:
            thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
        with col3:
            exang = 0
            exang_selection = st.radio("Exercise Induced Angina", ["No", "Yes"])
            if exang_selection == "Yes":
                exang = 1

        with col1:
            oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
        with col2:
            slope_options = {
                "Upsloping": 0,
                "Flat": 1,
                "Downsloping": 2
            }
            slope_selection = st.selectbox("Peak Exercise ST Segment", list(slope_options.keys()))
            slope = slope_options[slope_selection]

        with col3:
            ca = st.number_input("Number of Major Vessels (0-3)", min_value=0, max_value=3, value=0)

        with col1:
            thal_options = {
                "Normal": 0,
                "Fixed Defect": 1,
                "Reversible Defect": 2
            }
            thal_selection = st.selectbox("Thalassemia", list(thal_options.keys()))
            thal = thal_options[thal_selection]

    with info_tab:
        st.markdown("### Parameter Information")
        st.markdown("""
        - **Age**: Patient's age in years
        - **Gender**: Male or Female
        - **Chest Pain Type**:
            - Typical Angina: Chest pain related to decreased blood supply to the heart
            - Atypical Angina: Chest pain not related to heart
            - Non-anginal Pain: Typically esophageal spasms
            - Asymptomatic: No symptoms
        - **Resting Blood Pressure**: mm Hg on admission to the hospital
        - **Serum Cholesterol**: mg/dl
        - **Fasting Blood Sugar**: > 120 mg/dl
        - **Resting ECG**: Results of electrocardiogram while at rest
        - **Max Heart Rate**: Maximum heart rate achieved during exercise
        - **Exercise Induced Angina**: Angina induced by exercise
        - **ST Depression**: ST depression induced by exercise relative to rest
        - **Peak Exercise ST Segment**: The slope of the peak exercise ST segment
        - **Number of Major Vessels**: Number of major vessels colored by fluoroscopy (0-3)
        - **Thalassemia**: A blood disorder
        """)

    # code for prediction
    heart_dig = ''

    # button
    if st.button("Predict Heart Disease Risk"):
        # Create input array
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

        # Make prediction
        heart_prediction = heart_model.predict(input_data)

        # Get probability if available
        try:
            heart_prob = heart_model.predict_proba(input_data)[0][1]
            probability_text = f" (Confidence: {heart_prob*100:.2f}%)"
        except:
            probability_text = ""

        # Display result
        if heart_prediction[0] == 1:
            heart_dig = f"We are sorry to inform you that you may have Heart Disease{probability_text}."
            image = Image.open('positive.jpg')
            st.image(image, caption='')
        else:
            heart_dig = f"Good news! You likely don't have Heart Disease{probability_text}."
            image = Image.open('negative.jpg')
            st.image(image, caption='')

        st.success(f"{name}, {heart_dig}")

        # Display comprehensive model metrics
        heart_metrics = load_model_metrics("heart")
        if heart_metrics:
            display_model_metrics(heart_metrics, "Heart Disease")

        # Show explainable AI if enabled
        if show_explanation:
            st.markdown("### 🤖 AI Explanation: Understanding Your Heart Disease Risk")

            # Feature names for heart model
            feature_names = ['Age', 'Sex', 'Chest Pain Type', 'Resting BP', 'Cholesterol',
                            'Fasting Blood Sugar', 'Resting ECG', 'Max Heart Rate',
                            'Exercise Angina', 'ST Depression', 'ST Slope',
                            'Major Vessels', 'Thalassemia']

            # Get advanced feature importance
            feature_importance = explain_prediction_advanced(heart_model, input_data, feature_names, input_data)

            # Plot advanced feature importance
            if feature_importance is not None:
                st.markdown("#### 📊 AI Analysis: Which Parameters Cause High Heart Disease Risk")
                fig = plot_feature_importance_advanced(feature_importance, "Heart Disease Risk Factors Analysis")
                st.plotly_chart(fig, use_container_width=True)

                # Display detailed risk factors analysis
                display_risk_factors_analysis(feature_importance, "Heart Disease")

                # Display top contributing factors with detailed explanation
                st.markdown("#### 🎯 Top 5 Contributing Factors:")
                top_factors = feature_importance.head(5)
                for i, (_, row) in enumerate(top_factors.iterrows()):
                    risk_emoji = "🔴" if row['risk_level'] == 'High' else "🟡" if row['risk_level'] == 'Medium' else "🟢"
                    st.markdown(f"{risk_emoji} **{i+1}. {row['feature']}** ({row['risk_level']} Risk)")
                    st.markdown(f"   - Your Value: {row['input_value']:.2f}")
                    st.markdown(f"   - Contribution Score: {row['contribution']:.4f}")
                    st.markdown(f"   - Model Importance: {row['importance']:.4f}")

                # Enhanced health recommendations based on risk analysis
                st.markdown("### 💡 Personalized Cardiac Health Recommendations:")
                high_risk_features = feature_importance[feature_importance['risk_level'] == 'High']['feature'].tolist()

                if 'Cholesterol' in high_risk_features:
                    st.error("🚨 **CRITICAL: High Cholesterol** - Immediate dietary changes and possible medication needed")
                elif 'Cholesterol' in top_factors['feature'].values:
                    st.warning("🍎 **Monitor cholesterol levels** - Consider heart-healthy diet and regular testing")

                if 'Chest Pain Type' in high_risk_features:
                    st.error("💔 **CRITICAL: Significant Chest Pain** - Immediate cardiology consultation required")
                elif 'Chest Pain Type' in top_factors['feature'].values:
                    st.warning("⚠️ **Chest pain detected** - Discuss symptoms with your doctor promptly")

                if 'ST Depression' in high_risk_features:
                    st.error("📈 **CRITICAL: Abnormal ST Depression** - Advanced cardiac testing recommended")
                elif 'ST Depression' in top_factors['feature'].values:
                    st.info("📊 **Monitor ST changes** - Regular ECG monitoring may be beneficial")

                if 'Max Heart Rate' in high_risk_features:
                    st.error("💓 **CRITICAL: Poor Exercise Capacity** - Cardiac rehabilitation program recommended")
                elif 'Max Heart Rate' in top_factors['feature'].values:
                    st.info("❤️ **Improve cardiovascular fitness** - Regular moderate exercise under medical guidance")

                if 'Major Vessels' in high_risk_features:
                    st.error("🩸 **CRITICAL: Vessel Blockage** - Immediate angiography/intervention may be needed")
                elif 'Major Vessels' in top_factors['feature'].values:
                    st.warning("🔍 **Vessel concerns** - Advanced cardiac imaging recommended")

                if 'Age' in high_risk_features:
                    st.warning("🕰️ **Age-related cardiac risk** - More frequent cardiac screenings recommended")
                elif 'Age' in top_factors['feature'].values:
                    st.info("🕰️ **Age factor** - Regular cardiac check-ups important as you age")

                if 'Thalassemia' in high_risk_features:
                    st.error("🩸 **CRITICAL: Blood Disorder Impact** - Hematology and cardiology coordination needed")

                # Overall cardiac risk assessment
                high_risk_count = len(high_risk_features)
                if high_risk_count >= 3:
                    st.error("🚨 **VERY HIGH CARDIAC RISK** - Multiple critical factors. Emergency cardiology consultation recommended.")
                elif high_risk_count >= 1:
                    st.warning("⚠️ **ELEVATED CARDIAC RISK** - Concerning factors identified. Schedule cardiology appointment.")
                else:
                    st.success("✅ **LOW CARDIAC RISK** - Most factors within acceptable ranges. Continue heart-healthy lifestyle.")









if selected == 'Parkison Prediction':
    st.title("Parkison prediction")
    image = Image.open('p1.jpg')
    st.image(image, caption='parkinsons disease')
  # parameters
#    name	MDVP:Fo(Hz)	MDVP:Fhi(Hz)	MDVP:Flo(Hz)	MDVP:Jitter(%)	MDVP:Jitter(Abs)	MDVP:RAP	MDVP:PPQ	Jitter:DDP	MDVP:Shimmer	MDVP:Shimmer(dB)	Shimmer:APQ3	Shimmer:APQ5	MDVP:APQ	Shimmer:DDA	NHR	HNR	status	RPDE	DFA	spread1	spread2	D2	PPE
   # change the variables according to the dataset used in the model
    name = st.text_input("Name:")
    col1, col2, col3 = st.columns(3)
    with col1:
        MDVP = st.number_input("MDVP:Fo(Hz)")
    with col2:
        MDVPFIZ = st.number_input("MDVP:Fhi(Hz)")
    with col3:
        MDVPFLO = st.number_input("MDVP:Flo(Hz)")
    with col1:
        MDVPJITTER = st.number_input("MDVP:Jitter(%)")
    with col2:
        MDVPJitterAbs = st.number_input("MDVP:Jitter(Abs)")
    with col3:
        MDVPRAP = st.number_input("MDVP:RAP")

    with col2:

        MDVPPPQ = st.number_input("MDVP:PPQ ")
    with col3:
        JitterDDP = st.number_input("Jitter:DDP")
    with col1:
        MDVPShimmer = st.number_input("MDVP:Shimmer")
    with col2:
        MDVPShimmer_dB = st.number_input("MDVP:Shimmer(dB)")
    with col3:
        Shimmer_APQ3 = st.number_input("Shimmer:APQ3")
    with col1:
        ShimmerAPQ5 = st.number_input("Shimmer:APQ5")
    with col2:
        MDVP_APQ = st.number_input("MDVP:APQ")
    with col3:
        ShimmerDDA = st.number_input("Shimmer:DDA")
    with col1:
        NHR = st.number_input("NHR")
    with col2:
        HNR = st.number_input("HNR")
  
    with col2:
        RPDE = st.number_input("RPDE")
    with col3:
        DFA = st.number_input("DFA")
    with col1:
        spread1 = st.number_input("spread1")
    with col1:
        spread2 = st.number_input("spread2")
    with col3:
        D2 = st.number_input("D2")
    with col1:
        PPE = st.number_input("PPE")

    # code for prediction
    parkinson_dig = ''
    
    # button
    if st.button("Parkinson test result"):
        parkinson_prediction=[[]]
        # change the parameters according to the model
        parkinson_prediction = parkinson_model.predict([[MDVP, MDVPFIZ, MDVPFLO, MDVPJITTER, MDVPJitterAbs, MDVPRAP, MDVPPPQ, JitterDDP, MDVPShimmer,MDVPShimmer_dB, Shimmer_APQ3, ShimmerAPQ5, MDVP_APQ, ShimmerDDA, NHR, HNR,  RPDE, DFA, spread1, spread2, D2, PPE]])

        # Get probability if available
        try:
            parkinson_prob = parkinson_model.predict_proba([[MDVP, MDVPFIZ, MDVPFLO, MDVPJITTER, MDVPJitterAbs, MDVPRAP, MDVPPPQ, JitterDDP, MDVPShimmer,MDVPShimmer_dB, Shimmer_APQ3, ShimmerAPQ5, MDVP_APQ, ShimmerDDA, NHR, HNR,  RPDE, DFA, spread1, spread2, D2, PPE]])[0][1]
            probability_text = f" (Confidence: {parkinson_prob*100:.2f}%)"
        except:
            probability_text = ""

        if parkinson_prediction[0] == 1:
            parkinson_dig = f'We are sorry to inform you that you may have Parkinson\'s disease{probability_text}.'
            image = Image.open('positive.jpg')
            st.image(image, caption='')
        else:
            parkinson_dig = f"Good news! You likely don't have Parkinson's disease{probability_text}."
            image = Image.open('negative.jpg')
            st.image(image, caption='')

        st.success(f"{name}, {parkinson_dig}")

        # Display comprehensive model metrics
        parkinsons_metrics = load_model_metrics("parkinsons")
        if parkinsons_metrics:
            display_model_metrics(parkinsons_metrics, "Parkinson's Disease")

        # Show explainable AI if enabled
        if show_explanation:
            st.markdown("### 🤖 AI Explanation: Understanding Your Parkinson's Disease Risk")

            # Feature names for Parkinson's model
            feature_names = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
                            'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
                            'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
                            'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA',
                            'spread1', 'spread2', 'D2', 'PPE']

            # Create input array for explanation
            input_data = np.array([[MDVP, MDVPFIZ, MDVPFLO, MDVPJITTER, MDVPJitterAbs, MDVPRAP, MDVPPPQ, JitterDDP, MDVPShimmer,MDVPShimmer_dB, Shimmer_APQ3, ShimmerAPQ5, MDVP_APQ, ShimmerDDA, NHR, HNR,  RPDE, DFA, spread1, spread2, D2, PPE]])

            # Get advanced feature importance
            feature_importance = explain_prediction_advanced(parkinson_model, input_data, feature_names, input_data)

            # Plot advanced feature importance
            if feature_importance is not None:
                st.markdown("#### 📊 AI Analysis: Which Voice Parameters Cause High Parkinson's Risk")
                fig = plot_feature_importance_advanced(feature_importance, "Parkinson's Disease Voice Analysis")
                st.plotly_chart(fig, use_container_width=True)

                # Display detailed risk factors analysis
                display_risk_factors_analysis(feature_importance, "Parkinson's Disease")

                # Display top contributing factors with detailed explanation
                st.markdown("#### 🎯 Top 5 Voice Pattern Contributors:")
                top_factors = feature_importance.head(5)
                for i, (_, row) in enumerate(top_factors.iterrows()):
                    risk_emoji = "🔴" if row['risk_level'] == 'High' else "🟡" if row['risk_level'] == 'Medium' else "🟢"
                    st.markdown(f"{risk_emoji} **{i+1}. {row['feature']}** ({row['risk_level']} Risk)")
                    st.markdown(f"   - Your Value: {row['input_value']:.4f}")
                    st.markdown(f"   - Contribution Score: {row['contribution']:.4f}")
                    st.markdown(f"   - Model Importance: {row['importance']:.4f}")

                # Enhanced health recommendations based on voice analysis
                st.markdown("### 💡 Personalized Neurological Health Recommendations:")
                high_risk_features = feature_importance[feature_importance['risk_level'] == 'High']['feature'].tolist()

                voice_quality_features = ['MDVP:Jitter(%)', 'MDVP:Shimmer', 'HNR', 'NHR']
                frequency_features = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)']
                complexity_features = ['RPDE', 'DFA', 'D2', 'PPE']

                # Check for voice quality issues
                voice_issues = [f for f in high_risk_features if any(vf in f for vf in voice_quality_features)]
                if voice_issues:
                    st.error("🎤 **CRITICAL: Voice Quality Abnormalities** - Speech therapy evaluation recommended")
                    st.markdown("**Affected voice parameters:**")
                    for issue in voice_issues:
                        st.markdown(f"   • {issue}")

                # Check for frequency issues
                freq_issues = [f for f in high_risk_features if any(ff in f for ff in frequency_features)]
                if freq_issues:
                    st.warning("📢 **Voice Frequency Irregularities** - Vocal cord examination recommended")

                # Check for complexity issues
                complex_issues = [f for f in high_risk_features if any(cf in f for cf in complexity_features)]
                if complex_issues:
                    st.warning("🧠 **Voice Pattern Complexity Changes** - Neurological assessment recommended")

                # Overall neurological risk assessment
                high_risk_count = len(high_risk_features)
                if high_risk_count >= 5:
                    st.error("🚨 **VERY HIGH NEUROLOGICAL RISK** - Multiple voice abnormalities detected. Immediate neurologist consultation recommended.")
                elif high_risk_count >= 2:
                    st.warning("⚠️ **ELEVATED NEUROLOGICAL RISK** - Several voice pattern changes detected. Schedule neurological evaluation.")
                else:
                    st.success("✅ **LOW NEUROLOGICAL RISK** - Voice patterns mostly within normal ranges. Continue monitoring.")

                # Specific recommendations
                st.markdown("#### 🎯 Specific Recommendations:")
                st.info("🎤 **Voice Exercises**: Practice vocal exercises to maintain voice quality")
                st.info("🧠 **Cognitive Activities**: Engage in activities that challenge your brain")
                st.info("🏃 **Physical Exercise**: Regular exercise may help maintain neurological health")
                st.info("👨‍⚕️ **Regular Monitoring**: Consider periodic voice analysis and neurological check-ups")











# Liver prediction page
if selected == 'Liver prediction':  # pagetitle
    st.title("Liver disease prediction")
    image = Image.open('liver.jpg')
    st.image(image, caption='Liver disease prediction.')
    # columns
    # no inputs from the user
# st.write(info.astype(int).info())
    name = st.text_input("Name:")
    col1, col2, col3 = st.columns(3)

    with col1:
        Sex=0
        display = ("male", "female")
        options = list(range(len(display)))
        value = st.selectbox("Gender", options, format_func=lambda x: display[x])
        if value == "male":
            Sex = 0
        elif value == "female":
            Sex = 1
    with col2:
        age = st.number_input("Entre your age") # 2 
    with col3:
        Total_Bilirubin = st.number_input("Entre your Total_Bilirubin") # 3
    with col1:
        Direct_Bilirubin = st.number_input("Entre your Direct_Bilirubin")# 4

    with col2:
        Alkaline_Phosphotase = st.number_input("Entre your Alkaline_Phosphotase") # 5
    with col3:
        Alamine_Aminotransferase = st.number_input("Entre your Alamine_Aminotransferase") # 6
    with col1:
        Aspartate_Aminotransferase = st.number_input("Entre your Aspartate_Aminotransferase") # 7
    with col2:
        Total_Protiens = st.number_input("Entre your Total_Protiens")# 8
    with col3:
        Albumin = st.number_input("Entre your Albumin") # 9
    with col1:
        Albumin_and_Globulin_Ratio = st.number_input("Entre your Albumin_and_Globulin_Ratio") # 10 
    # code for prediction
    liver_dig = ''

    # button
    if st.button("Liver test result"):
        liver_prediction=[[]]
        liver_prediction = liver_model.predict([[Sex,age,Total_Bilirubin,Direct_Bilirubin,Alkaline_Phosphotase,Alamine_Aminotransferase,Aspartate_Aminotransferase,Total_Protiens,Albumin,Albumin_and_Globulin_Ratio]])

        # Get probability if available
        try:
            liver_prob = liver_model.predict_proba([[Sex,age,Total_Bilirubin,Direct_Bilirubin,Alkaline_Phosphotase,Alamine_Aminotransferase,Aspartate_Aminotransferase,Total_Protiens,Albumin,Albumin_and_Globulin_Ratio]])[0][1]
            probability_text = f" (Confidence: {liver_prob*100:.2f}%)"
        except:
            probability_text = ""

        # Display prediction result
        if liver_prediction[0] == 1:
            image = Image.open('positive.jpg')
            st.image(image, caption='')
            liver_dig = f"We are sorry to inform you that you may have liver disease{probability_text}."
        else:
            image = Image.open('negative.jpg')
            st.image(image, caption='')
            liver_dig = f"Good news! You likely don't have liver disease{probability_text}."

        st.success(f"{name}, {liver_dig}")

        # Display comprehensive model metrics
        liver_metrics = load_model_metrics("liver")
        if liver_metrics:
            display_model_metrics(liver_metrics, "Liver Disease")

        # Show explainable AI if enabled
        if show_explanation:
            st.markdown("### 🤖 AI Explanation: Understanding Your Liver Disease Risk")

            # Feature names for liver model
            feature_names = ['Gender', 'Age', 'Total Bilirubin', 'Direct Bilirubin',
                            'Alkaline Phosphatase', 'Alamine Aminotransferase',
                            'Aspartate Aminotransferase', 'Total Proteins', 'Albumin',
                            'Albumin and Globulin Ratio']

            # Create input array for explanation
            input_data = np.array([[Sex,age,Total_Bilirubin,Direct_Bilirubin,Alkaline_Phosphotase,Alamine_Aminotransferase,Aspartate_Aminotransferase,Total_Protiens,Albumin,Albumin_and_Globulin_Ratio]])

            # Get advanced feature importance
            feature_importance = explain_prediction_advanced(liver_model, input_data, feature_names, input_data)

            # Plot advanced feature importance
            if feature_importance is not None:
                st.markdown("#### 📊 AI Analysis: Which Lab Values Cause High Liver Disease Risk")
                fig = plot_feature_importance_advanced(feature_importance, "Liver Disease Risk Factors Analysis")
                st.plotly_chart(fig, use_container_width=True)

                # Display detailed risk factors analysis
                display_risk_factors_analysis(feature_importance, "Liver Disease")

                # Display top contributing factors with detailed explanation
                st.markdown("#### 🎯 Top 5 Contributing Lab Values:")
                top_factors = feature_importance.head(5)
                for i, (_, row) in enumerate(top_factors.iterrows()):
                    risk_emoji = "🔴" if row['risk_level'] == 'High' else "🟡" if row['risk_level'] == 'Medium' else "🟢"
                    st.markdown(f"{risk_emoji} **{i+1}. {row['feature']}** ({row['risk_level']} Risk)")
                    st.markdown(f"   - Your Value: {row['input_value']:.2f}")
                    st.markdown(f"   - Contribution Score: {row['contribution']:.4f}")
                    st.markdown(f"   - Model Importance: {row['importance']:.4f}")

                # Enhanced health recommendations based on liver function analysis
                st.markdown("### 💡 Personalized Liver Health Recommendations:")
                high_risk_features = feature_importance[feature_importance['risk_level'] == 'High']['feature'].tolist()

                if 'Total Bilirubin' in high_risk_features or 'Direct Bilirubin' in high_risk_features:
                    st.error("🟡 **CRITICAL: Elevated Bilirubin** - Immediate hepatologist consultation for jaundice evaluation")
                elif any(bil in top_factors['feature'].values for bil in ['Total Bilirubin', 'Direct Bilirubin']):
                    st.warning("🟡 **Bilirubin elevation** - Monitor liver function and consider hepatology referral")

                if 'Alamine Aminotransferase' in high_risk_features or 'Aspartate Aminotransferase' in high_risk_features:
                    st.error("🧪 **CRITICAL: Elevated Liver Enzymes** - Urgent liver function evaluation needed")
                elif any(enzyme in top_factors['feature'].values for enzyme in ['Alamine Aminotransferase', 'Aspartate Aminotransferase']):
                    st.warning("🧪 **Liver enzyme elevation** - Repeat liver function tests and avoid hepatotoxic substances")

                if 'Alkaline Phosphatase' in high_risk_features:
                    st.error("📈 **CRITICAL: High Alkaline Phosphatase** - Biliary obstruction or liver disease evaluation needed")
                elif 'Alkaline Phosphatase' in top_factors['feature'].values:
                    st.warning("📈 **Alkaline phosphatase elevation** - Consider imaging studies and hepatology consultation")

                if 'Albumin' in high_risk_features or 'Total Proteins' in high_risk_features:
                    st.error("🥩 **CRITICAL: Low Protein/Albumin** - Liver synthetic function impairment - immediate medical attention")
                elif any(protein in top_factors['feature'].values for protein in ['Albumin', 'Total Proteins']):
                    st.warning("🥩 **Protein levels concerning** - Nutritional assessment and liver function monitoring")

                if 'Albumin and Globulin Ratio' in high_risk_features:
                    st.warning("⚖️ **Abnormal A/G Ratio** - Liver function and immune system evaluation recommended")

                if 'Age' in high_risk_features:
                    st.warning("🕰️ **Age-related liver risk** - Regular liver function monitoring recommended")

                # Overall liver disease risk assessment
                high_risk_count = len(high_risk_features)
                if high_risk_count >= 3:
                    st.error("🚨 **VERY HIGH LIVER DISEASE RISK** - Multiple critical lab abnormalities. Immediate hepatologist consultation required.")
                elif high_risk_count >= 1:
                    st.warning("⚠️ **ELEVATED LIVER DISEASE RISK** - Concerning lab values identified. Schedule hepatology evaluation.")
                else:
                    st.success("✅ **LOW LIVER DISEASE RISK** - Most lab values within acceptable ranges. Continue liver-healthy lifestyle.")

                # Specific liver health recommendations
                st.markdown("#### 🎯 Liver Health Recommendations:")
                st.info("🚫 **Avoid Alcohol**: Limit or eliminate alcohol consumption to protect liver health")
                st.info("💊 **Medication Review**: Review all medications and supplements with your doctor")
                st.info("🥗 **Healthy Diet**: Follow a balanced diet low in processed foods and high in antioxidants")
                st.info("💉 **Vaccination**: Consider hepatitis A and B vaccination if not immune")
                st.info("🔬 **Regular Monitoring**: Periodic liver function tests to track health status")






# Hepatitis prediction page
if selected == 'Hepatitis prediction':
    st.title("Hepatitis Prediction")
    image = Image.open('h.png')
    st.image(image, caption='Hepatitis Prediction')

    # Columns
    # No inputs from the user
    name = st.text_input("Name:")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Enter your age")  # 2
    with col2:
        sex = st.selectbox("Gender", ["Male", "Female"])
        sex = 1 if sex == "Male" else 2
    with col3:
        total_bilirubin = st.number_input("Enter your Total Bilirubin")  # 3

    with col1:
        direct_bilirubin = st.number_input("Enter your Direct Bilirubin")  # 4
    with col2:
        alkaline_phosphatase = st.number_input("Enter your Alkaline Phosphatase")  # 5
    with col3:
        alamine_aminotransferase = st.number_input("Enter your Alamine Aminotransferase")  # 6

    with col1:
        aspartate_aminotransferase = st.number_input("Enter your Aspartate Aminotransferase")  # 7
    with col2:
        total_proteins = st.number_input("Enter your Total Proteins")  # 8
    with col3:
        albumin = st.number_input("Enter your Albumin")  # 9

    with col1:
        albumin_and_globulin_ratio = st.number_input("Enter your Albumin and Globulin Ratio")  # 10

    with col2:
        your_ggt_value = st.number_input("Enter your GGT value")  # Add this line
    with col3:
        your_prot_value = st.number_input("Enter your PROT value")  # Add this line

    # Code for prediction
    hepatitis_result = ''

    # Button
    if st.button("Predict Hepatitis"):
        # Create a DataFrame with user inputs
        user_data = pd.DataFrame({
            'Age': [age],
            'Sex': [sex],
            'ALB': [total_bilirubin],  # Correct the feature name
            'ALP': [direct_bilirubin],  # Correct the feature name
            'ALT': [alkaline_phosphatase],  # Correct the feature name
            'AST': [alamine_aminotransferase],
            'BIL': [aspartate_aminotransferase],  # Correct the feature name
            'CHE': [total_proteins],  # Correct the feature name
            'CHOL': [albumin],  # Correct the feature name
            'CREA': [albumin_and_globulin_ratio],  # Correct the feature name
            'GGT': [your_ggt_value],  # Replace 'your_ggt_value' with the actual value
            'PROT': [your_prot_value]  # Replace 'your_prot_value' with the actual value
        })

        # Perform prediction
        hepatitis_prediction = hepatitis_model.predict(user_data)

        # Get probability if available
        try:
            hepatitis_prob = hepatitis_model.predict_proba(user_data)[0][1]
            probability_text = f" (Confidence: {hepatitis_prob*100:.2f}%)"
        except:
            probability_text = ""

        # Display result
        if hepatitis_prediction[0] == 1:
            hepatitis_result = f"We are sorry to inform you that you may have Hepatitis{probability_text}."
            image = Image.open('positive.jpg')
            st.image(image, caption='')
        else:
            hepatitis_result = f'Good news! You likely do not have Hepatitis{probability_text}.'
            image = Image.open('negative.jpg')
            st.image(image, caption='')

        st.success(f"{name}, {hepatitis_result}")

        # Display comprehensive model metrics
        hepatitis_metrics = load_model_metrics("hepatitis")
        if hepatitis_metrics:
            display_model_metrics(hepatitis_metrics, "Hepatitis")

        # Show explainable AI if enabled
        if show_explanation:
            st.markdown("### 🤖 AI Explanation: Understanding Your Hepatitis Risk")

            # Feature names for hepatitis model
            feature_names = ['Age', 'Sex', 'ALB', 'ALP', 'ALT', 'AST', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']

            # Convert user data to numpy array for explanation
            input_data = user_data.values

            # Get advanced feature importance
            feature_importance = explain_prediction_advanced(hepatitis_model, input_data, feature_names, input_data)

            # Plot advanced feature importance
            if feature_importance is not None:
                st.markdown("#### 📊 AI Analysis: Which Lab Values Cause High Hepatitis Risk")
                fig = plot_feature_importance_advanced(feature_importance, "Hepatitis Risk Factors Analysis")
                st.plotly_chart(fig, use_container_width=True)

                # Display detailed risk factors analysis
                display_risk_factors_analysis(feature_importance, "Hepatitis")

                # Display top contributing factors with detailed explanation
                st.markdown("#### 🎯 Top 5 Contributing Lab Values:")
                top_factors = feature_importance.head(5)
                for i, (_, row) in enumerate(top_factors.iterrows()):
                    risk_emoji = "🔴" if row['risk_level'] == 'High' else "🟡" if row['risk_level'] == 'Medium' else "🟢"
                    st.markdown(f"{risk_emoji} **{i+1}. {row['feature']}** ({row['risk_level']} Risk)")
                    st.markdown(f"   - Your Value: {row['input_value']:.2f}")
                    st.markdown(f"   - Contribution Score: {row['contribution']:.4f}")
                    st.markdown(f"   - Model Importance: {row['importance']:.4f}")

                # Enhanced health recommendations based on hepatitis analysis
                st.markdown("### 💡 Personalized Hepatitis Health Recommendations:")
                high_risk_features = feature_importance[feature_importance['risk_level'] == 'High']['feature'].tolist()

                if 'ALT' in high_risk_features or 'AST' in high_risk_features:
                    st.error("🧪 **CRITICAL: Elevated Liver Enzymes** - Immediate hepatologist consultation for liver inflammation")
                elif any(enzyme in top_factors['feature'].values for enzyme in ['ALT', 'AST']):
                    st.warning("🧪 **Liver enzyme elevation** - Monitor liver function and avoid hepatotoxic substances")

                if 'BIL' in high_risk_features:
                    st.error("🟡 **CRITICAL: High Bilirubin** - Urgent evaluation for liver dysfunction and jaundice")
                elif 'BIL' in top_factors['feature'].values:
                    st.warning("🟡 **Bilirubin elevation** - Monitor for signs of liver impairment")

                if 'ALP' in high_risk_features:
                    st.error("📈 **CRITICAL: High Alkaline Phosphatase** - Biliary obstruction or liver disease evaluation needed")
                elif 'ALP' in top_factors['feature'].values:
                    st.warning("📈 **ALP elevation** - Consider imaging studies for biliary system")

                if 'ALB' in high_risk_features or 'PROT' in high_risk_features:
                    st.error("🥩 **CRITICAL: Protein Abnormalities** - Liver synthetic function assessment needed")
                elif any(protein in top_factors['feature'].values for protein in ['ALB', 'PROT']):
                    st.warning("🥩 **Protein levels concerning** - Nutritional and liver function evaluation")

                if 'GGT' in high_risk_features:
                    st.error("🍺 **CRITICAL: High GGT** - Alcohol-related liver damage or bile duct issues")
                elif 'GGT' in top_factors['feature'].values:
                    st.warning("🍺 **GGT elevation** - Consider alcohol cessation and liver protection")

                if 'CHOL' in high_risk_features:
                    st.warning("💊 **Cholesterol abnormalities** - Liver metabolism evaluation recommended")

                if 'CREA' in high_risk_features:
                    st.warning("🫘 **Kidney function concerns** - Hepatorenal syndrome evaluation may be needed")

                if 'Age' in high_risk_features:
                    st.warning("🕰️ **Age-related hepatitis risk** - Regular liver function monitoring recommended")

                # Overall hepatitis risk assessment
                high_risk_count = len(high_risk_features)
                if high_risk_count >= 4:
                    st.error("🚨 **VERY HIGH HEPATITIS RISK** - Multiple critical lab abnormalities. Immediate hepatologist and infectious disease consultation required.")
                elif high_risk_count >= 2:
                    st.warning("⚠️ **ELEVATED HEPATITIS RISK** - Several concerning lab values. Schedule comprehensive liver evaluation.")
                else:
                    st.success("✅ **LOW HEPATITIS RISK** - Most lab values within acceptable ranges. Continue liver-protective lifestyle.")

                # Specific hepatitis health recommendations
                st.markdown("#### 🎯 Hepatitis Prevention & Management:")
                st.info("💉 **Vaccination**: Ensure hepatitis A and B vaccination if not immune")
                st.info("🚫 **Avoid Alcohol**: Complete alcohol cessation to prevent further liver damage")
                st.info("💊 **Medication Safety**: Review all medications for hepatotoxicity with your doctor")
                st.info("🧼 **Hygiene**: Practice good hygiene to prevent hepatitis transmission")
                st.info("🔬 **Regular Monitoring**: Periodic liver function tests and viral load monitoring")
                st.info("👨‍⚕️ **Specialist Care**: Regular follow-up with hepatologist or gastroenterologist")











# jaundice prediction page
if selected == 'Jaundice prediction':  # pagetitle
    st.title("Jaundice disease prediction")
    image = Image.open('j.jpg')
    st.image(image, caption='Jaundice disease prediction')
    # columns
    # no inputs from the user
# st.write(info.astype(int).info())
    name = st.text_input("Name:")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Entre your age   ") # 2 
    with col2:
        Sex=0
        display = ("male", "female")
        options = list(range(len(display)))
        value = st.selectbox("Gender", options, format_func=lambda x: display[x])
        if value == "male":
            Sex = 0
        elif value == "female":
            Sex = 1
    with col3:
        Total_Bilirubin = st.number_input("Entre your Total_Bilirubin") # 3
    with col1:
        Direct_Bilirubin = st.number_input("Entre your Direct_Bilirubin")# 4

    with col2:
        Alkaline_Phosphotase = st.number_input("Entre your Alkaline_Phosphotase") # 5
    with col3:
        Alamine_Aminotransferase = st.number_input("Entre your Alamine_Aminotransferase") # 6
    with col1:
        Total_Protiens = st.number_input("Entre your Total_Protiens")# 8
    with col2:
        Albumin = st.number_input("Entre your Albumin") # 9 
    # code for prediction
    jaundice_dig = ''

    # button
    if st.button("Jaundice test result"):
        jaundice_prediction=[[]]
        jaundice_prediction = jaundice_model.predict([[age,Sex,Total_Bilirubin,Direct_Bilirubin,Alkaline_Phosphotase,Alamine_Aminotransferase,Total_Protiens,Albumin]])

        # after the prediction is done if the value in the list at index is 0 is 1 then the person is diabetic
        if jaundice_prediction[0] == 1:
            image = Image.open('positive.jpg')
            st.image(image, caption='')
            jaundice_dig = "we are really sorry to say but it seems like you have Jaundice."
        else:
            image = Image.open('negative.jpg')
            st.image(image, caption='')
            jaundice_dig = "Congratulation , You don't have Jaundice."
        st.success(name+' , ' + jaundice_dig)












from sklearn.preprocessing import LabelEncoder
import joblib


# Chronic Kidney Disease Prediction Page
if selected == 'Chronic Kidney prediction':
    st.title("Chronic Kidney Disease Prediction")

    # Check if we have the proper chronic kidney model
    import os
    if not os.path.exists('models/chronic_model.sav'):
        st.warning("⚠️ **Chronic Kidney Disease model not found!** Using temporary fallback model. Please create the proper model for accurate predictions.")

        # Add model creation option
        with st.expander("🔧 Create Chronic Kidney Disease Model", expanded=True):
            st.info("Click the button below to create a proper chronic kidney disease prediction model with balanced training data.")
            if st.button("🔄 Create Chronic Kidney Disease Model"):
                with st.spinner("Creating new model... This may take a few moments."):
                    try:
                        chronic_disease_model, accuracy = create_chronic_kidney_model()
                        st.success(f"✅ Model created successfully! Accuracy: {accuracy:.4f}")
                        st.info("🔄 Please refresh the page to use the new model.")
                        st.balloons()
                    except Exception as e:
                        st.error(f"❌ Failed to create model: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
    else:
        st.success("✅ Using proper Chronic Kidney Disease model")

        # Add model retraining option
        with st.expander("🔧 Model Management", expanded=False):
            st.info("If you're experiencing issues with predictions, you can retrain the model.")
            if st.button("🔄 Retrain Chronic Kidney Disease Model"):
                with st.spinner("Training new model..."):
                    try:
                        chronic_disease_model, accuracy = create_chronic_kidney_model()
                        st.success(f"✅ Model retrained successfully! New accuracy: {accuracy:.4f}")
                        st.info("🔄 Please refresh the page to use the new model.")
                    except Exception as e:
                        st.error(f"❌ Failed to retrain model: {str(e)}")

    # Add the image for Chronic Kidney Disease prediction if needed
    name = st.text_input("Name:")
    # Columns
    # No inputs from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Enter your age", 1, 100, 25)  # 2
    with col2:
        bp = st.slider("Enter your Blood Pressure", 50, 200, 120)  # Add your own ranges
    with col3:
        sg = st.slider("Enter your Specific Gravity", 1.0, 1.05, 1.02)  # Add your own ranges

    with col1:
        al = st.slider("Enter your Albumin", 0, 5, 0)  # Add your own ranges
    with col2:
        su = st.slider("Enter your Sugar", 0, 5, 0)  # Add your own ranges
    with col3:
        rbc = st.selectbox("Red Blood Cells", ["Normal", "Abnormal"])
        rbc = 1 if rbc == "Normal" else 0

    with col1:
        pc = st.selectbox("Pus Cells", ["Normal", "Abnormal"])
        pc = 1 if pc == "Normal" else 0
    with col2:
        pcc = st.selectbox("Pus Cell Clumps", ["Present", "Not Present"])
        pcc = 1 if pcc == "Present" else 0
    with col3:
        ba = st.selectbox("Bacteria", ["Present", "Not Present"])
        ba = 1 if ba == "Present" else 0

    with col1:
        bgr = st.slider("Enter your Blood Glucose Random", 50, 200, 120)  # Add your own ranges
    with col2:
        bu = st.slider("Enter your Blood Urea", 10, 200, 60)  # Add your own ranges
    with col3:
        sc = st.slider("Enter your Serum Creatinine", 0, 10, 3)  # Add your own ranges

    with col1:
        sod = st.slider("Enter your Sodium", 100, 200, 140)  # Add your own ranges
    with col2:
        pot = st.slider("Enter your Potassium", 2, 7, 4)  # Add your own ranges
    with col3:
        hemo = st.slider("Enter your Hemoglobin", 3, 17, 12)  # Add your own ranges

    with col1:
        pcv = st.slider("Enter your Packed Cell Volume", 20, 60, 40)  # Add your own ranges
    with col2:
        wc = st.slider("Enter your White Blood Cell Count", 2000, 20000, 10000)  # Add your own ranges
    with col3:
        rc = st.slider("Enter your Red Blood Cell Count", 2, 8, 4)  # Add your own ranges

    with col1:
        htn = st.selectbox("Hypertension", ["Yes", "No"])
        htn = 1 if htn == "Yes" else 0
    with col2:
        dm = st.selectbox("Diabetes Mellitus", ["Yes", "No"])
        dm = 1 if dm == "Yes" else 0
    with col3:
        cad = st.selectbox("Coronary Artery Disease", ["Yes", "No"])
        cad = 1 if cad == "Yes" else 0

    with col1:
        appet = st.selectbox("Appetite", ["Good", "Poor"])
        appet = 1 if appet == "Good" else 0
    with col2:
        pe = st.selectbox("Pedal Edema", ["Yes", "No"])
        pe = 1 if pe == "Yes" else 0
    with col3:
        ane = st.selectbox("Anemia", ["Yes", "No"])
        ane = 1 if ane == "Yes" else 0

    # Code for prediction
    kidney_result = ''

    # Button
    if st.button("Predict Chronic Kidney Disease"):
        # Create a DataFrame with user inputs
        user_input = pd.DataFrame({
            'age': [age],
            'bp': [bp],
            'sg': [sg],
            'al': [al],
            'su': [su],
            'rbc': [rbc],
            'pc': [pc],
            'pcc': [pcc],
            'ba': [ba],
            'bgr': [bgr],
            'bu': [bu],
            'sc': [sc],
            'sod': [sod],
            'pot': [pot],
            'hemo': [hemo],
            'pcv': [pcv],
            'wc': [wc],
            'rc': [rc],
            'htn': [htn],
            'dm': [dm],
            'cad': [cad],
            'appet': [appet],
            'pe': [pe],
            'ane': [ane]
        })

        # Perform prediction
        kidney_prediction = chronic_disease_model.predict(user_input)

        # Get probability if available
        try:
            kidney_prob = chronic_disease_model.predict_proba(user_input)[0][1]
            probability_text = f" (Confidence: {kidney_prob*100:.2f}%)"
        except:
            probability_text = ""

        # Display result
        if kidney_prediction[0] == 1:
            image = Image.open('positive.jpg')
            st.image(image, caption='')
            kidney_prediction_dig = f"We are sorry to inform you that you may have chronic kidney disease{probability_text}."
        else:
            image = Image.open('negative.jpg')
            st.image(image, caption='')
            kidney_prediction_dig = f"Good news! You likely don't have chronic kidney disease{probability_text}."

        st.success(f"{name}, {kidney_prediction_dig}")

        # Display comprehensive model metrics
        chronic_metrics = load_model_metrics("chronic")
        if chronic_metrics:
            display_model_metrics(chronic_metrics, "Chronic Kidney Disease")

        # Show explainable AI if enabled
        if show_explanation:
            st.markdown("### 🤖 AI Explanation: Understanding Your Chronic Kidney Disease Risk")

            # Feature names for chronic kidney disease model
            feature_names = ['Age', 'Blood Pressure', 'Specific Gravity', 'Albumin', 'Sugar',
                            'Red Blood Cells', 'Pus Cells', 'Pus Cell Clumps', 'Bacteria',
                            'Blood Glucose Random', 'Blood Urea', 'Serum Creatinine', 'Sodium',
                            'Potassium', 'Hemoglobin', 'Packed Cell Volume', 'White Blood Cell Count',
                            'Red Blood Cell Count', 'Hypertension', 'Diabetes Mellitus',
                            'Coronary Artery Disease', 'Appetite', 'Pedal Edema', 'Anemia']

            # Convert user data to numpy array for explanation
            input_data = user_input.values

            # Get advanced feature importance
            feature_importance = explain_prediction_advanced(chronic_disease_model, input_data, feature_names, input_data)

            # Plot advanced feature importance
            if feature_importance is not None:
                st.markdown("#### 📊 AI Analysis: Which Parameters Cause High Chronic Kidney Disease Risk")
                fig = plot_feature_importance_advanced(feature_importance, "Chronic Kidney Disease Risk Analysis")
                st.plotly_chart(fig, use_container_width=True)

                # Display detailed risk factors analysis
                display_risk_factors_analysis(feature_importance, "Chronic Kidney Disease")

                # Display top contributing factors with detailed explanation
                st.markdown("#### 🎯 Top 5 Contributing Risk Factors:")
                top_factors = feature_importance.head(5)
                for i, (_, row) in enumerate(top_factors.iterrows()):
                    risk_emoji = "🔴" if row['risk_level'] == 'High' else "🟡" if row['risk_level'] == 'Medium' else "🟢"
                    st.markdown(f"{risk_emoji} **{i+1}. {row['feature']}** ({row['risk_level']} Risk)")
                    st.markdown(f"   - Your Value: {row['input_value']:.2f}")
                    st.markdown(f"   - Contribution Score: {row['contribution']:.4f}")
                    st.markdown(f"   - Model Importance: {row['importance']:.4f}")

                # Enhanced health recommendations based on kidney function analysis
                st.markdown("### 💡 Personalized Kidney Health Recommendations:")
                high_risk_features = feature_importance[feature_importance['risk_level'] == 'High']['feature'].tolist()

                if 'Serum Creatinine' in high_risk_features:
                    st.error("🫘 **CRITICAL: High Serum Creatinine** - Immediate nephrology consultation for kidney function evaluation")
                elif 'Serum Creatinine' in top_factors['feature'].values:
                    st.warning("🫘 **Creatinine elevation** - Monitor kidney function and avoid nephrotoxic medications")

                if 'Blood Urea' in high_risk_features:
                    st.error("🩸 **CRITICAL: High Blood Urea** - Urgent kidney function assessment needed")
                elif 'Blood Urea' in top_factors['feature'].values:
                    st.warning("🩸 **Urea elevation** - Kidney function monitoring and dietary protein management")

                if 'Albumin' in high_risk_features:
                    st.error("🟡 **CRITICAL: Proteinuria** - Significant kidney damage indicated - immediate medical attention")
                elif 'Albumin' in top_factors['feature'].values:
                    st.warning("🟡 **Protein in urine** - Early kidney damage possible - nephrology referral recommended")

                if 'Blood Pressure' in high_risk_features or 'Hypertension' in high_risk_features:
                    st.error("💓 **CRITICAL: High Blood Pressure** - Major kidney disease risk factor - immediate BP control needed")
                elif any(bp in top_factors['feature'].values for bp in ['Blood Pressure', 'Hypertension']):
                    st.warning("💓 **Blood pressure concerns** - Strict BP control essential for kidney protection")

                if 'Diabetes Mellitus' in high_risk_features:
                    st.error("🍯 **CRITICAL: Diabetes** - Leading cause of kidney disease - intensive glucose control needed")
                elif 'Diabetes Mellitus' in top_factors['feature'].values:
                    st.warning("🍯 **Diabetes detected** - Strict glucose control and regular kidney monitoring essential")

                if 'Hemoglobin' in high_risk_features or 'Anemia' in high_risk_features:
                    st.error("🩸 **CRITICAL: Anemia** - Advanced kidney disease indicator - hematology consultation needed")
                elif any(anemia in top_factors['feature'].values for anemia in ['Hemoglobin', 'Anemia']):
                    st.warning("🩸 **Anemia concerns** - Iron studies and kidney function evaluation recommended")

                if 'Pedal Edema' in high_risk_features:
                    st.error("🦵 **CRITICAL: Fluid Retention** - Advanced kidney disease with fluid overload")
                elif 'Pedal Edema' in top_factors['feature'].values:
                    st.warning("🦵 **Swelling detected** - Fluid management and kidney function assessment needed")

                if any(electrolyte in high_risk_features for electrolyte in ['Sodium', 'Potassium']):
                    st.error("⚡ **CRITICAL: Electrolyte Imbalance** - Dangerous kidney function impairment")
                elif any(electrolyte in top_factors['feature'].values for electrolyte in ['Sodium', 'Potassium']):
                    st.warning("⚡ **Electrolyte concerns** - Regular monitoring and dietary management needed")

                # Overall chronic kidney disease risk assessment
                high_risk_count = len(high_risk_features)
                if high_risk_count >= 4:
                    st.error("🚨 **VERY HIGH CHRONIC KIDNEY DISEASE RISK** - Multiple critical factors. Immediate nephrology consultation and possible dialysis evaluation required.")
                elif high_risk_count >= 2:
                    st.warning("⚠️ **ELEVATED CHRONIC KIDNEY DISEASE RISK** - Several concerning factors. Urgent nephrology referral recommended.")
                else:
                    st.success("✅ **LOW CHRONIC KIDNEY DISEASE RISK** - Most parameters within acceptable ranges. Continue kidney-protective lifestyle.")

                # Specific kidney health recommendations
                st.markdown("#### 🎯 Kidney Protection Strategies:")
                st.info("💧 **Hydration**: Maintain adequate fluid intake unless restricted by doctor")
                st.info("🧂 **Low Sodium Diet**: Reduce salt intake to protect kidney function")
                st.info("🥩 **Protein Management**: Moderate protein intake as advised by nephrologist")
                st.info("💊 **Medication Safety**: Avoid NSAIDs and nephrotoxic medications")
                st.info("🩺 **Regular Monitoring**: Frequent kidney function tests and blood pressure checks")
                st.info("🏃 **Exercise**: Regular physical activity within limits set by your doctor")




