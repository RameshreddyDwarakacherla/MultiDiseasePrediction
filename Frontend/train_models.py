#!/usr/bin/env python3
"""
Script to train all disease prediction models from scratch
This script replaces the pre-trained models with newly trained ones
"""

import sys
import os

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from code.train import DiseaseModelTrainer

def main():
    """Main function to train all models"""
    print("Starting model training process...")
    print("This will create new models to replace the pre-trained ones.")
    
    # Create trainer instance
    trainer = DiseaseModelTrainer()
    
    # Train all models
    trainer.train_all_models()
    
    print("\nModel training completed!")
    print("You can now run the Streamlit app with your own trained models.")

if __name__ == "__main__":
    main()
