#!/usr/bin/env python3
"""
Script to retrain only the chronic kidney disease model
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'code'))

from train import DiseaseModelTrainer

def main():
    print("🔄 Retraining Chronic Kidney Disease Model...")
    print("=" * 50)
    
    try:
        # Initialize trainer
        trainer = DiseaseModelTrainer()
        
        # Train only the chronic kidney model
        accuracy = trainer.train_chronic_kidney_model()
        
        print("=" * 50)
        print(f"✅ SUCCESS: Chronic Kidney Disease model retrained!")
        print(f"📊 Final Accuracy: {accuracy:.4f}")
        print("🎯 Model saved to: models/chronic_model.sav")
        print("📈 Metrics saved to: models/chronic_metrics.json")
        
    except Exception as e:
        print(f"❌ ERROR: Failed to retrain model")
        print(f"Error details: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
