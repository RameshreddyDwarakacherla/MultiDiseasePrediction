#!/usr/bin/env python3
"""
Demo script to showcase enhanced features of the Disease Prediction System
"""

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def demo_comprehensive_metrics():
    """Demonstrate comprehensive metrics for all models"""
    print("🎯 COMPREHENSIVE MODEL METRICS DEMONSTRATION")
    print("=" * 60)
    
    try:
        # Load comprehensive metrics
        with open('models/all_metrics_summary.json', 'r') as f:
            all_metrics = json.load(f)
        
        print("\n📊 DETAILED PERFORMANCE METRICS FOR ALL MODELS:")
        print("-" * 60)
        
        for disease_key, metrics in all_metrics.items():
            if isinstance(metrics, dict) and 'accuracy' in metrics:
                disease_name = disease_key.replace('_', ' ').title()
                print(f"\n🏥 {disease_name}:")
                print(f"   🎯 Accuracy:  {metrics.get('accuracy', 0):.4f} ({metrics.get('accuracy', 0)*100:.2f}%)")
                
                if 'precision' in metrics:
                    print(f"   🔍 Precision: {metrics.get('precision', 0):.4f} ({metrics.get('precision', 0)*100:.2f}%)")
                    print(f"   📈 Recall:    {metrics.get('recall', 0):.4f} ({metrics.get('recall', 0)*100:.2f}%)")
                    print(f"   ⚖️  F1 Score:  {metrics.get('f1_score', 0):.4f} ({metrics.get('f1_score', 0)*100:.2f}%)")
                    print(f"   🤖 Model:     {metrics.get('model_type', 'Unknown')}")
                else:
                    print(f"   🤖 Model:     XGBoost (Symptom-based)")
        
        print("\n" + "=" * 60)
        
    except FileNotFoundError:
        print("❌ Comprehensive metrics file not found. Please run enhanced_train_models.py first.")

def demo_explainable_ai():
    """Demonstrate explainable AI features"""
    print("\n🤖 EXPLAINABLE AI FEATURES DEMONSTRATION")
    print("=" * 60)
    
    try:
        # Load a model for demonstration
        diabetes_model = joblib.load('models/diabetes_model.sav')
        
        # Create sample input data
        sample_data = np.array([[2, 140, 85, 25, 120, 28.5, 0.8, 45]])
        feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        
        print("\n📋 SAMPLE PATIENT DATA:")
        for i, (feature, value) in enumerate(zip(feature_names, sample_data[0])):
            print(f"   {feature}: {value}")
        
        # Make prediction
        prediction = diabetes_model.predict(sample_data)
        try:
            probability = diabetes_model.predict_proba(sample_data)[0]
            print(f"\n🎯 PREDICTION: {'Diabetic' if prediction[0] == 1 else 'Non-Diabetic'}")
            print(f"   Confidence: {max(probability)*100:.2f}%")
        except:
            print(f"\n🎯 PREDICTION: {'Diabetic' if prediction[0] == 1 else 'Non-Diabetic'}")
        
        # Feature importance analysis
        if hasattr(diabetes_model, 'feature_importances_'):
            importances = diabetes_model.feature_importances_
            
            # Calculate contributions
            contributions = []
            for importance, value in zip(importances, sample_data[0]):
                contribution = importance * abs(value) if value != 0 else importance * 0.1
                contributions.append(contribution)
            
            # Create analysis dataframe
            analysis_df = pd.DataFrame({
                'Feature': feature_names,
                'Your_Value': sample_data[0],
                'Importance': importances,
                'Contribution': contributions,
                'Risk_Level': ['High' if c > np.mean(contributions) else 'Medium' if c > np.mean(contributions)*0.5 else 'Low' for c in contributions]
            }).sort_values('Contribution', ascending=False)
            
            print("\n🔍 AI EXPLANATION - FEATURE IMPORTANCE ANALYSIS:")
            print("-" * 60)
            print(f"{'Rank':<4} {'Feature':<20} {'Value':<8} {'Risk':<8} {'Contribution':<12}")
            print("-" * 60)
            
            for i, (_, row) in enumerate(analysis_df.head(8).iterrows()):
                risk_emoji = "🔴" if row['Risk_Level'] == 'High' else "🟡" if row['Risk_Level'] == 'Medium' else "🟢"
                print(f"{i+1:<4} {row['Feature']:<20} {row['Your_Value']:<8.1f} {risk_emoji} {row['Risk_Level']:<6} {row['Contribution']:<12.4f}")
            
            # Risk assessment
            high_risk_factors = analysis_df[analysis_df['Risk_Level'] == 'High']
            print(f"\n⚠️  RISK ASSESSMENT:")
            print(f"   High Risk Factors: {len(high_risk_factors)}")
            print(f"   Overall Risk Level: {'HIGH' if len(high_risk_factors) >= 3 else 'MODERATE' if len(high_risk_factors) >= 1 else 'LOW'}")
            
            if not high_risk_factors.empty:
                print(f"\n🔴 HIGH RISK FACTORS IDENTIFIED:")
                for _, factor in high_risk_factors.iterrows():
                    print(f"   • {factor['Feature']}: {factor['Your_Value']:.1f} (Contribution: {factor['Contribution']:.4f})")
        
        print("\n" + "=" * 60)
        
    except FileNotFoundError:
        print("❌ Model files not found. Please run enhanced_train_models.py first.")
    except Exception as e:
        print(f"❌ Error in explainable AI demo: {e}")

def demo_model_comparison():
    """Demonstrate model comparison across diseases"""
    print("\n📊 MODEL PERFORMANCE COMPARISON")
    print("=" * 60)
    
    try:
        with open('models/all_metrics_summary.json', 'r') as f:
            all_metrics = json.load(f)
        
        # Extract metrics for comparison
        comparison_data = []
        for disease_key, metrics in all_metrics.items():
            if isinstance(metrics, dict) and 'accuracy' in metrics:
                disease_name = disease_key.replace('_', ' ').title()
                comparison_data.append({
                    'Disease': disease_name,
                    'Accuracy': metrics.get('accuracy', 0),
                    'Precision': metrics.get('precision', 0),
                    'Recall': metrics.get('recall', 0),
                    'F1_Score': metrics.get('f1_score', 0)
                })
        
        # Sort by F1 score
        comparison_data.sort(key=lambda x: x.get('F1_Score', x.get('Accuracy', 0)), reverse=True)
        
        print("\n🏆 MODEL RANKING (by F1 Score / Accuracy):")
        print("-" * 80)
        print(f"{'Rank':<4} {'Disease':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1 Score':<10}")
        print("-" * 80)
        
        for i, model in enumerate(comparison_data):
            accuracy = f"{model['Accuracy']:.3f}"
            precision = f"{model['Precision']:.3f}" if model['Precision'] > 0 else "N/A"
            recall = f"{model['Recall']:.3f}" if model['Recall'] > 0 else "N/A"
            f1_score = f"{model['F1_Score']:.3f}" if model['F1_Score'] > 0 else "N/A"
            
            print(f"{i+1:<4} {model['Disease']:<20} {accuracy:<10} {precision:<10} {recall:<10} {f1_score:<10}")
        
        # Best and worst performers
        best_model = comparison_data[0]
        worst_model = comparison_data[-1]
        
        print(f"\n🥇 BEST PERFORMER: {best_model['Disease']}")
        print(f"   Overall Score: {best_model.get('F1_Score', best_model['Accuracy']):.3f}")
        
        print(f"\n📈 NEEDS IMPROVEMENT: {worst_model['Disease']}")
        print(f"   Overall Score: {worst_model.get('F1_Score', worst_model['Accuracy']):.3f}")
        
        print("\n" + "=" * 60)
        
    except FileNotFoundError:
        print("❌ Comprehensive metrics file not found.")

def main():
    """Main demo function"""
    print("🎉 ENHANCED DISEASE PREDICTION SYSTEM DEMO")
    print("🤖 Featuring: Comprehensive Metrics + Explainable AI")
    print("=" * 70)
    
    # Demo comprehensive metrics
    demo_comprehensive_metrics()
    
    # Demo explainable AI
    demo_explainable_ai()
    
    # Demo model comparison
    demo_model_comparison()
    
    print("\n✨ ENHANCED FEATURES SUMMARY:")
    print("=" * 70)
    print("✅ Comprehensive Metrics: Accuracy, Precision, Recall, F1-Score")
    print("✅ Explainable AI: Feature importance with risk level analysis")
    print("✅ Risk Assessment: High/Medium/Low risk factor identification")
    print("✅ Personalized Recommendations: Based on individual risk factors")
    print("✅ Model Comparison: Performance ranking across all diseases")
    print("✅ Real-time Analysis: Instant explanations for every prediction")
    
    print("\n🚀 TO RUN THE ENHANCED APPLICATION:")
    print("   streamlit run app.py")
    
    print("\n📚 DOCUMENTATION:")
    print("   • MODEL_TRAINING_README.md - Training guide")
    print("   • IMPLEMENTATION_SUMMARY.md - Feature overview")
    print("   • enhanced_train_models.py - Enhanced training script")

if __name__ == "__main__":
    main()
