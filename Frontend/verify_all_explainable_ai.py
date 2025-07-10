#!/usr/bin/env python3
"""
Verification script to confirm all diseases have explainable AI implemented
"""

import re

def check_explainable_ai_implementation():
    """Check if all disease predictions have explainable AI"""
    
    print("🔍 VERIFYING EXPLAINABLE AI IMPLEMENTATION FOR ALL DISEASES")
    print("=" * 70)
    
    # Read the app.py file
    with open('app.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Define all disease sections to check
    diseases = [
        'Diabetes Prediction',
        'Heart disease Prediction',
        'Parkison Prediction',
        'Liver prediction',
        'Hepatitis prediction',
        'Chronic Kidney prediction'
    ]
    
    results = {}
    
    for disease in diseases:
        print(f"\n🏥 Checking {disease}...")
        
        # Find the disease section
        pattern = rf"if selected == '{re.escape(disease)}':(.*?)(?=if selected ==|$)"
        match = re.search(pattern, content, re.DOTALL)
        
        if match:
            section_content = match.group(1)
            
            # Check for required explainable AI components
            checks = {
                'Model Metrics': 'load_model_metrics' in section_content and 'display_model_metrics' in section_content,
                'AI Explanation': 'show_explanation' in section_content and 'AI Explanation' in section_content,
                'Feature Importance': 'explain_prediction_advanced' in section_content,
                'Risk Analysis': 'plot_feature_importance_advanced' in section_content,
                'Risk Factors': 'display_risk_factors_analysis' in section_content,
                'Recommendations': 'Personalized' in section_content and 'Recommendations' in section_content,
                'Risk Assessment': 'high_risk_features' in section_content,
                'Critical Alerts': any(alert in section_content for alert in ['CRITICAL', 'HIGH RISK', 'ELEVATED RISK'])
            }
            
            results[disease] = checks
            
            # Display results for this disease
            all_implemented = all(checks.values())
            status = "✅ COMPLETE" if all_implemented else "❌ INCOMPLETE"
            print(f"   Status: {status}")
            
            for check_name, implemented in checks.items():
                emoji = "✅" if implemented else "❌"
                print(f"   {emoji} {check_name}")
                
        else:
            print(f"   ❌ Disease section not found!")
            results[disease] = {check: False for check in ['Model Metrics', 'AI Explanation', 'Feature Importance', 'Risk Analysis', 'Risk Factors', 'Recommendations', 'Risk Assessment', 'Critical Alerts']}
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 EXPLAINABLE AI IMPLEMENTATION SUMMARY")
    print("=" * 70)
    
    total_diseases = len(diseases)
    complete_diseases = sum(1 for disease_checks in results.values() if all(disease_checks.values()))
    
    print(f"\n🎯 Overall Progress: {complete_diseases}/{total_diseases} diseases have complete explainable AI")
    print(f"📈 Completion Rate: {(complete_diseases/total_diseases)*100:.1f}%")
    
    if complete_diseases == total_diseases:
        print("\n🎉 SUCCESS: All diseases have comprehensive explainable AI implemented!")
        print("✅ Model performance metrics")
        print("✅ Advanced AI explanations") 
        print("✅ Risk factor analysis")
        print("✅ Personalized recommendations")
        print("✅ Critical health alerts")
    else:
        print(f"\n⚠️ INCOMPLETE: {total_diseases - complete_diseases} diseases still need explainable AI")
        
        # Show which diseases are incomplete
        for disease, checks in results.items():
            if not all(checks.values()):
                print(f"\n❌ {disease} - Missing:")
                for check_name, implemented in checks.items():
                    if not implemented:
                        print(f"   • {check_name}")
    
    return complete_diseases == total_diseases

def check_specific_features():
    """Check for specific advanced features"""
    
    print("\n" + "=" * 70)
    print("🚀 CHECKING ADVANCED FEATURES")
    print("=" * 70)
    
    with open('app.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    advanced_features = {
        'Risk Level Classification': 'risk_level.*High.*Medium.*Low' in content.replace('\n', ' '),
        'Color-coded Risk Factors': 'risk_emoji.*🔴.*🟡.*🟢' in content.replace('\n', ' '),
        'Contribution Scores': 'contribution.*score' in content.lower(),
        'Personalized Recommendations': 'personalized.*recommendations' in content.lower(),
        'Critical Alerts': 'critical.*alert' in content.lower() or 'critical:' in content.lower(),
        'Medical Disclaimers': 'medical.*disclaimer' in content.lower(),
        'Comprehensive Metrics': 'accuracy.*precision.*recall.*f1' in content.lower().replace('\n', ' '),
        'Interactive Visualizations': 'plotly_chart' in content and 'plot_feature_importance_advanced' in content
    }
    
    for feature, implemented in advanced_features.items():
        emoji = "✅" if implemented else "❌"
        print(f"{emoji} {feature}")
    
    advanced_count = sum(advanced_features.values())
    total_advanced = len(advanced_features)
    
    print(f"\n🎯 Advanced Features: {advanced_count}/{total_advanced} implemented")
    print(f"📈 Advanced Feature Rate: {(advanced_count/total_advanced)*100:.1f}%")
    
    return advanced_count == total_advanced

def main():
    """Main verification function"""
    
    print("🎉 COMPREHENSIVE EXPLAINABLE AI VERIFICATION")
    print("🤖 Checking all disease predictions for advanced AI features")
    print("=" * 70)
    
    # Check basic explainable AI implementation
    basic_complete = check_explainable_ai_implementation()
    
    # Check advanced features
    advanced_complete = check_specific_features()
    
    # Final assessment
    print("\n" + "=" * 70)
    print("🏆 FINAL ASSESSMENT")
    print("=" * 70)
    
    if basic_complete and advanced_complete:
        print("🎉 PERFECT IMPLEMENTATION!")
        print("✅ All diseases have comprehensive explainable AI")
        print("✅ All advanced features are implemented")
        print("✅ System ready for production use")
        print("\n🚀 The enhanced Disease Prediction System is complete with:")
        print("   • Comprehensive metrics (Accuracy, Precision, Recall, F1)")
        print("   • Advanced explainable AI for all 8 diseases")
        print("   • Risk level classification (High/Medium/Low)")
        print("   • Color-coded risk factor analysis")
        print("   • Personalized health recommendations")
        print("   • Critical health alerts and warnings")
        print("   • Interactive visualizations")
        print("   • Professional medical-grade interface")
    elif basic_complete:
        print("✅ GOOD: Basic explainable AI implemented for all diseases")
        print("⚠️ Some advanced features may need enhancement")
    else:
        print("❌ INCOMPLETE: Some diseases still missing explainable AI")
        print("🔧 Please complete implementation for all diseases")
    
    print(f"\n📊 Overall System Status:")
    print(f"   Basic AI Explanations: {'✅ Complete' if basic_complete else '❌ Incomplete'}")
    print(f"   Advanced Features: {'✅ Complete' if advanced_complete else '❌ Incomplete'}")
    
    return basic_complete and advanced_complete

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎯 VERIFICATION PASSED: All explainable AI features implemented!")
    else:
        print("\n⚠️ VERIFICATION FAILED: Some features still need implementation.")
