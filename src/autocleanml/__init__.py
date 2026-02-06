"""
AutoCleanML - Automated Machine Learning Data Cleaning Pipeline
================================================================

AutoCleanML automates the tedious process of data cleaning and preprocessing
for machine learning projects. It provides a simple, powerful API with
intelligent, context-aware cleaning and automatic feature engineering.

Key Features:
- Smart missing value imputation (context-aware, not blind)
- Advanced outlier detection (IQR, Z-score, Isolation Forest)
- Automatic data type correction (handles currencies, dates, etc.)
- Feature extraction (text, datetime, categorical, numeric)
- Algorithm-aware preprocessing
- Detailed reporting with reasoning for every decision

Quick Start:
-----------
>>> from autocleanml import AutoCleanML
>>> 
>>> # Simple API - handles everything
>>> cleaner = AutoCleanML(target="price")
>>> X_train, X_test, y_train, y_test, report = cleaner.fit_transform("data.csv")
>>> 
>>> # Train your model
>>> from sklearn.ensemble import RandomForestRegressor
>>> model = RandomForestRegressor()
>>> model.fit(X_train, y_train)

Alternative (Advanced):
----------------------
>>> from autocleanml import AutoCleanPipeline
>>> 
>>> # More control over the pipeline
>>> pipeline = AutoCleanPipeline(target_col="target", model_type="xgboost")
>>> clean_train, clean_test = pipeline.clean(df)

For more information, visit: https://github.com/likith-n/AutoCleanML
"""

from autocleanml.main import AutoCleanML

__version__ = "0.1.0"
__author__ = "Likith.N"
__email__ = "nlikith54@gmail.com"

__all__ = [
    "AutoCleanML",         # Main entry point (recommended)
]
