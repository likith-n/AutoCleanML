"""
AutoCleanML - Basic Classification Example
===========================================

This example demonstrates how to use AutoCleanML for a classification task.
We'll predict customer churn with automatic handling of imbalanced data.

Run this script:
    python basic_classification.py
"""

import pandas as pd
from autocleanml import AutoCleanML
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    classification_report,
    confusion_matrix
)
import numpy as np


def main():
    print("="*70)
    print("AutoCleanML - Basic Classification Example")
    print("="*70)
    
    # Step 1: Load your data
    print("\n Step 1: Loading data...")
    # Replace with your actual data path
    # Example: Credit risk, customer churn, fraud detection, etc.
    df = pd.read_csv("../data/german_credit_data.csv")  # Update this path
    print(f"   Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"   Target: Risk (good/bad)")
    
    # Step 2: Initialize AutoCleanML
    print("\n Step 2: Initializing AutoCleanML...")
    cleaner = AutoCleanML(
        target="Risk",
        test_size=0.2,
        random_state=42,
        feature_extraction=True,
        model_type='tree',
        verbose=True
    )
    
    # Step 3: Clean and prepare data (ONE LINE!)
    print("\n🧹 Step 3: Cleaning data with AutoCleanML...")
    X_train, X_test, y_train, y_test, report = cleaner.fit_transform(df)
    
    print(f"\n Data cleaned!")
    print(f"   Training set: {X_train.shape}")
    print(f"   Test set: {X_test.shape}")
    
    # Step 4: Check for class imbalance
    print("\n  Step 4: Checking class imbalance...")
    if 'imbalance' in report and report['imbalance']['is_imbalanced']:
        print("     Dataset is IMBALANCED!")
        print(f"   Class distribution: {report['imbalance']['class_distribution']}")
        print(f"   Imbalance ratio: {report['imbalance']['imbalance_ratio']:.3f}")
        print(f"   Recommended strategy: {report['imbalance']['recommended_strategy']}")
        
        # Get class weights
        class_weights = report['imbalance']['class_weights']
        print(f"\n    Using recommended class weights:")
        for cls, weight in class_weights.items():
            print(f"      {cls}: {weight:.2f}")
    else:
        class_weights = None
        print("   ✓ Dataset is balanced")
    
    # Step 5: Train model with class weights
    print("\n Step 5: Training Random Forest classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        class_weight=class_weights,  # Use AutoCleanML's recommended weights
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("   ✓ Model trained!")
    
    # Step 6: Make predictions
    print("\n Step 6: Making predictions...")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Step 7: Evaluate performance
    print("\n Step 7: Evaluating model performance...")
    
    # Training metrics
    print("\n   Training Performance:")
    print(f"      Accuracy:  {accuracy_score(y_train, y_pred_train):.4f}")
    print(f"      Precision: {precision_score(y_train, y_pred_train, average='weighted'):.4f}")
    print(f"      Recall:    {recall_score(y_train, y_pred_train, average='weighted'):.4f}")
    print(f"      F1-Score:  {f1_score(y_train, y_pred_train, average='weighted'):.4f}")
    
    # Test metrics
    print("\n   Test Performance:")
    print(f"      Accuracy:  {accuracy_score(y_test, y_pred_test):.4f}")
    print(f"      Precision: {precision_score(y_test, y_pred_test, average='weighted'):.4f}")
    print(f"      Recall:    {recall_score(y_test, y_pred_test, average='weighted'):.4f}")
    print(f"      F1-Score:  {f1_score(y_test, y_pred_test, average='weighted'):.4f}")
    
    # Classification report
    print("\n   Detailed Classification Report (Test Set):")
    print("\n" + classification_report(y_test, y_pred_test, indent='      '))
    
    # Confusion matrix
    print("   Confusion Matrix (Test Set):")
    cm = confusion_matrix(y_test, y_pred_test)
    print(f"\n      {cm}\n")
    
    # Step 8: Feature importance
    print("🔝 Step 8: Top 10 Most Important Features:")
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(10)
    
    for idx, row in feature_importance.iterrows():
        print(f"      {idx+1:2d}. {row['feature']:30s} {row['importance']:.4f}")
    
    # Step 9: Check the cleaning report
    print("\n Step 9: Cleaning Report Summary:")
    print(f"   Original columns: {report['summary']['original_shape'][1]}")
    print(f"   Final features: {report['summary']['final_train_shape'][1]}")
    print(f"   Features created: {report['feature_engineering']['total_features_created']}")
    
    # Step 10: Save model
    print("\n Step 10: Saving model for production...")
    import pickle
    pickle.dump(cleaner, open('cleaner_classification.pkl', 'wb'))
    pickle.dump(model, open('model_classification.pkl', 'wb'))
    print("   ✓ Saved: cleaner_classification.pkl, model_classification.pkl")
    
    print("\n" + "="*70)
    print(" EXAMPLE COMPLETE!")
    print("="*70)
    print("\n Key Takeaways:")
    print("   1. AutoCleanML automatically detected class imbalance")
    print("   2. Recommended class weights were applied to the model")
    print("   3. All preprocessing done automatically")
    print("   4. Model trained and evaluated successfully")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
