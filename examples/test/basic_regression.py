"""
AutoCleanML - Basic Regression Example
=======================================

This example demonstrates how to use AutoCleanML for a regression task.
We'll predict house prices using the classic dataset.

Run this script:
    python basic_regression.py
"""

import pandas as pd
from autocleanml import AutoCleanML
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np


def main():
    print("="*70)
    print("AutoCleanML - Basic Regression Example")
    print("="*70)
    
    # Step 1: Load your data
    print("\n Step 1: Loading data...")
    # Replace with your actual data path
    df = pd.read_csv("../data/house_prices/train.csv")  # Update this path
    print(f"   Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"   Target: SalePrice")
    
    # Step 2: Initialize AutoCleanML
    print("\n Step 2: Initializing AutoCleanML...")
    cleaner = AutoCleanML(
        target="SalePrice",
        test_size=0.2,
        random_state=42,
        feature_extraction=True,
        model_type='tree',  # For RandomForest, no scaling needed
        verbose=True
    )
    
    # Step 3: Clean and prepare data (ONE LINE!)
    print("\n🧹 Step 3: Cleaning data with AutoCleanML...")
    X_train, X_test, y_train, y_test, report = cleaner.fit_transform(df)
    
    print(f"\n Data cleaned!")
    print(f"   Training set: {X_train.shape}")
    print(f"   Test set: {X_test.shape}")
    
    # Step 4: Train model
    print("\n Step 4: Training Random Forest model...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("   ✓ Model trained!")
    
    # Step 5: Make predictions
    print("\n Step 5: Making predictions...")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Step 6: Evaluate performance
    print("\n Step 6: Evaluating model performance...")
    
    # Training metrics
    train_r2 = r2_score(y_train, y_pred_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    train_mae = mean_absolute_error(y_train, y_pred_train)
    
    # Test metrics
    test_r2 = r2_score(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    print("\n   Training Performance:")
    print(f"      R² Score: {train_r2:.4f}")
    print(f"      RMSE: ${train_rmse:,.2f}")
    print(f"      MAE: ${train_mae:,.2f}")
    
    print("\n   Test Performance:")
    print(f"      R² Score: {test_r2:.4f}")
    print(f"      RMSE: ${test_rmse:,.2f}")
    print(f"      MAE: ${test_mae:,.2f}")
    
    # Step 7: Show top features
    print("\n Step 7: Top 10 Most Important Features:")
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(10)
    
    for idx, row in feature_importance.iterrows():
        print(f"      {idx+1:2d}. {row['feature']:30s} {row['importance']:.4f}")
    
    # Step 8: Check the cleaning report
    print("\n Step 8: Cleaning Report Summary:")
    print(f"   Original columns: {report['summary']['original_shape'][1]}")
    print(f"   Final features: {report['summary']['final_train_shape'][1]}")
    print(f"   Features created: {report['feature_engineering']['total_features_created']}")
    print(f"   Scaling method: {report['scaling']['scaling_method']}")
    
    # Step 9: Save model for production (optional)
    print("\n Step 9: Saving model for production...")
    import pickle
    pickle.dump(cleaner, open('cleaner.pkl', 'wb'))
    pickle.dump(model, open('model.pkl', 'wb'))
    print("   ✓ Saved: cleaner.pkl, model.pkl")
    
    print("\n" + "="*70)
    print(" EXAMPLE COMPLETE!")
    print("="*70)
    print("\n Key Takeaways:")
    print("   1. AutoCleanML reduced preprocessing from hours to minutes")
    print("   2. All data cleaning done automatically with smart decisions")
    print("   3. Model achieved good performance with clean data")
    print("   4. Ready for production deployment with saved models")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
