"""
AutoCleanML - Imbalanced Data Handling Example
===============================================

This example demonstrates AutoCleanML's automatic imbalance detection
and recommendations for handling severely imbalanced datasets.

Perfect for: Fraud detection, anomaly detection, rare event prediction

Run this script:
    python imbalance_handling.py
"""

import pandas as pd
import numpy as np
from autocleanml import AutoCleanML
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)


def main():
    print("="*70)
    print("AutoCleanML - Imbalanced Data Handling Example")
    print("="*70)
    
    # Step 1: Load imbalanced dataset
    print("\n📂 Step 1: Loading imbalanced dataset...")
    # Example: Credit card fraud, rare disease, anomaly detection
    df = pd.read_csv("../data/creditcard.csv")  # Update this path
    
    # Show class distribution
    print(f"   Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"   Target: Class (0=Normal, 1=Fraud)")
    
    target_col = 'Class'
    class_dist = df[target_col].value_counts()
    print(f"\n   📊 Class Distribution:")
    for cls, count in class_dist.items():
        percentage = (count / len(df)) * 100
        print(f"      Class {cls}: {count:,} ({percentage:.2f}%)")
    
    # Step 2: Initialize AutoCleanML
    print("\n🤖 Step 2: Initializing AutoCleanML...")
    cleaner = AutoCleanML(
        target=target_col,
        test_size=0.2,
        random_state=42,
        feature_extraction=False,  # Fraud data already has good features
        model_type='tree',
        verbose=True
    )
    
    # Step 3: Clean and analyze imbalance
    print("\n🧹 Step 3: Cleaning data and analyzing imbalance...")
    X_train, X_test, y_train, y_test, report = cleaner.fit_transform(df)
    
    print(f"\n✅ Data cleaned!")
    print(f"   Training set: {X_train.shape}")
    print(f"   Test set: {X_test.shape}")
    
    # Step 4: Review imbalance report
    print("\n" + "="*70)
    print("IMBALANCE ANALYSIS")
    print("="*70)
    
    imbalance_info = report['imbalance']
    
    if imbalance_info['is_imbalanced']:
        print("\n⚠️  DATASET IS IMBALANCED!")
        print(f"\n   Severity: {imbalance_info['severity']}")
        print(f"   Imbalance Ratio: {imbalance_info['imbalance_ratio']:.4f}")
        print(f"\n   Class Distribution (Training):")
        for cls, count in imbalance_info['class_distribution'].items():
            print(f"      Class {cls}: {count:,}")
        
        print(f"\n   📋 AutoCleanML Recommendation:")
        print(f"      Strategy: {imbalance_info['recommended_strategy']}")
        print(f"      Reason: {imbalance_info['reasoning']}")
        
        if imbalance_info['class_weights']:
            print(f"\n   💡 Recommended Class Weights:")
            for cls, weight in imbalance_info['class_weights'].items():
                print(f"      Class {cls}: {weight:.2f}")
    
    # Step 5: Train model WITHOUT class weights (baseline)
    print("\n" + "="*70)
    print("EXPERIMENT 1: Without Class Weights (Baseline)")
    print("="*70)
    
    print("\n🎯 Training Random Forest WITHOUT class weights...")
    model_baseline = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model_baseline.fit(X_train, y_train)
    y_pred_baseline = model_baseline.predict(X_test)
    
    print("\n   Results:")
    print(f"      Accuracy:  {accuracy_score(y_test, y_pred_baseline):.4f}")
    print(f"      Precision: {precision_score(y_test, y_pred_baseline, average='weighted'):.4f}")
    print(f"      Recall:    {recall_score(y_test, y_pred_baseline, average='weighted'):.4f}")
    print(f"      F1-Score:  {f1_score(y_test, y_pred_baseline, average='weighted'):.4f}")
    
    try:
        auc = roc_auc_score(y_test, model_baseline.predict_proba(X_test)[:, 1])
        print(f"      ROC-AUC:   {auc:.4f}")
    except:
        pass
    
    # Step 6: Train model WITH class weights (recommended)
    print("\n" + "="*70)
    print("EXPERIMENT 2: With AutoCleanML's Recommended Class Weights")
    print("="*70)
    
    print("\n🎯 Training Random Forest WITH class weights...")
    model_weighted = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight=imbalance_info['class_weights'],  # Use AutoCleanML's weights
        n_jobs=-1
    )
    model_weighted.fit(X_train, y_train)
    y_pred_weighted = model_weighted.predict(X_test)
    
    print("\n   Results:")
    print(f"      Accuracy:  {accuracy_score(y_test, y_pred_weighted):.4f}")
    print(f"      Precision: {precision_score(y_test, y_pred_weighted, average='weighted'):.4f}")
    print(f"      Recall:    {recall_score(y_test, y_pred_weighted, average='weighted'):.4f}")
    print(f"      F1-Score:  {f1_score(y_test, y_pred_weighted, average='weighted'):.4f}")
    
    try:
        auc = roc_auc_score(y_test, model_weighted.predict_proba(X_test)[:, 1])
        print(f"      ROC-AUC:   {auc:.4f}")
    except:
        pass
    
    # Step 7: Compare results
    print("\n" + "="*70)
    print("COMPARISON: Baseline vs Weighted")
    print("="*70)
    
    print("\n   Metric               Baseline    Weighted    Improvement")
    print("   " + "-"*60)
    
    metrics = {
        'Accuracy': (
            accuracy_score(y_test, y_pred_baseline),
            accuracy_score(y_test, y_pred_weighted)
        ),
        'Precision': (
            precision_score(y_test, y_pred_baseline, average='weighted'),
            precision_score(y_test, y_pred_weighted, average='weighted')
        ),
        'Recall': (
            recall_score(y_test, y_pred_baseline, average='weighted'),
            recall_score(y_test, y_pred_weighted, average='weighted')
        ),
        'F1-Score': (
            f1_score(y_test, y_pred_baseline, average='weighted'),
            f1_score(y_test, y_pred_weighted, average='weighted')
        ),
    }
    
    for metric_name, (baseline, weighted) in metrics.items():
        improvement = ((weighted - baseline) / baseline) * 100
        arrow = "↑" if improvement > 0 else "↓" if improvement < 0 else "→"
        print(f"   {metric_name:12s}     {baseline:.4f}      {weighted:.4f}      {arrow} {abs(improvement):.2f}%")
    
    # Step 8: Detailed report for minority class
    print("\n" + "="*70)
    print("MINORITY CLASS PERFORMANCE (Most Important!)")
    print("="*70)
    
    print("\n   Classification Report (Weighted Model):")
    print(classification_report(y_test, y_pred_weighted, target_names=['Normal', 'Fraud']))
    
    # Step 9: Save best model
    print("\n💾 Saving best model...")
    import pickle
    pickle.dump(cleaner, open('cleaner_imbalanced.pkl', 'wb'))
    pickle.dump(model_weighted, open('model_imbalanced.pkl', 'wb'))
    print("   ✓ Saved: cleaner_imbalanced.pkl, model_imbalanced.pkl")
    
    print("\n" + "="*70)
    print("✅ EXAMPLE COMPLETE!")
    print("="*70)
    print("\n💡 Key Takeaways:")
    print("   1. AutoCleanML automatically detected severe class imbalance")
    print("   2. Recommended class weights improved minority class detection")
    print("   3. Class weights are crucial for imbalanced datasets")
    print("   4. Always check minority class metrics (not just accuracy!)")
    print("\n   🎯 For Production:")
    print("      - Use the weighted model")
    print("      - Monitor precision and recall for minority class")
    print("      - Consider threshold tuning for your use case")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
