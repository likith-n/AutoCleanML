"""
AutoCleanML - Main Entry Point
==============================

Simple, powerful API for automated data cleaning and preprocessing.

Usage:
    from autocleanml import AutoCleanML
    
    cleaner = AutoCleanML(target="price")
    X_train, X_test, y_train, y_test, report = cleaner.fit_transform("data.csv")
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Union, Tuple, Dict
import warnings

from autocleanml.cleaning.smart_missing_handler import SmartMissingHandler
from autocleanml.cleaning.smart_outlier_handler import AdvancedOutlierHandler
from autocleanml.cleaning.datatype_corrector import DataTypeCorrector
from autocleanml.cleaning.categorical_handler import CategoricalHandler
from autocleanml.cleaning.transformers import FeatureScaler
from autocleanml.cleaning.feature_filter import FeatureFilter
from autocleanml.cleaning.feature_selector import FeatureSelector
from autocleanml.cleaning.imbalance_handler import ImbalanceHandler

from autocleanml.feature_engineering.text_features import TextFeatureExtractor
from autocleanml.feature_engineering.datetime_features import DateTimeFeatureExtractor
from autocleanml.feature_engineering.categorical_features import CategoricalFeatureExtractor
from autocleanml.feature_engineering.numeric_features import NumericFeatureExtractor


class AutoCleanML:
    """
    Automated Machine Learning Data Cleaning Pipeline.
    
    This is the main entry point for AutoCleanML. It provides a simple API
    that handles everything from raw CSV to clean train/test splits.
    
    Features:
    ---------
    1. Smart Missing Value Handling (context-aware)
       - Numerical: median for skewed, KNN for correlated, mean otherwise
       - Categorical: most_frequent for low cardinality, 'Missing' for high
       - Target: warns and drops rows
    
    2. Advanced Outlier Detection
       - IQR (robust), Z-score (normal data), Isolation Forest (multivariate)
       - Actions: cap (default), remove, or flag
    
    3. Data Type Correction
       - Auto-detects and fixes numeric stored as strings
       - Handles currency symbols, commas, percentages
       - Detects and converts datetime columns
    
    4. Feature Extraction
       - Text: length, keywords, numbers, patterns
       - Datetime: year, month, day, cyclical features
       - Categorical: multi-value expansion, location parsing
       - Numeric: polynomials, ratios, interactions
    
    5. Duplicate & Leakage Detection
       - Removes duplicate rows
       - Detects highly correlated features (multicollinearity)
       - Prevents data leakage via proper train/test splitting
    
    6. Detailed Reporting
       - Every decision is logged with reasoning
       - Before/after comparisons
       - Feature importance rankings
    
    Parameters
    ----------
    target : str, required
        Name of the target column (what you're predicting)
    
    test_size : float, default=0.2
        Proportion of data to use for testing (0.0 to 1.0)
    
    random_state : int, default=42
        Random seed for reproducible results
    
    outlier_method : str, default='auto'
        Method for outlier detection: 'iqr', 'zscore', 'isolation_forest', 'auto'
    
    outlier_action : str, default='cap'
        What to do with outliers: 'cap' (winsorize), 'remove', 'flag'
    
    feature_extraction : bool, default=True
        Whether to perform automatic feature extraction
    
    model_type : str, default='auto'
        Type of model for scaling: 'linear', 'tree', 'nn', 'auto'
    
    max_features : int, default=100
        Maximum number of features to keep after selection
    
    verbose : bool, default=True
        Whether to print progress and reports
    
    Examples
    --------
    >>> # Basic usage
    >>> from autocleanml import AutoCleanML
    >>> 
    >>> cleaner = AutoCleanML(target="price")
    >>> X_train, X_test, y_train, y_test, report = cleaner.fit_transform("house_prices.csv")
    >>> 
    >>> # Now train your model
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> model = RandomForestRegressor()
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)
    
    >>> # Advanced usage with custom settings
    >>> cleaner = AutoCleanML(
    ...     target="price",
    ...     test_size=0.3,
    ...     outlier_method='isolation_forest',
    ...     outlier_action='remove',
    ...     feature_extraction=True,
    ...     max_features=50,
    ...     verbose=True
    ... )
    >>> X_train, X_test, y_train, y_test, report = cleaner.fit_transform("data.csv")
    >>> 
    >>> # Examine the report
    >>> print(report['summary'])
    >>> print(report['missing_values'])
    >>> print(report['outliers'])
    >>> print(report['feature_engineering'])
    """
    
    def __init__(
        self,
        target: str,
        test_size: float = 0.2,
        random_state: int = 42,
        outlier_method: str = 'auto',
        outlier_action: str = 'cap',
        feature_extraction: bool = True,
        model_type: str = 'auto',
        model = None,  # NEW: Pass actual model object
        max_features: int = 100,
        verbose: bool = True
    ):
        """Initialize AutoCleanML pipeline."""
        if not target:
            raise ValueError("❌ 'target' parameter is REQUIRED!")
        
        self.target = target
        self.test_size = test_size
        self.random_state = random_state
        self.outlier_method = outlier_method
        self.outlier_action = outlier_action
        self.feature_extraction = feature_extraction
        self.max_features = max_features
        self.verbose = verbose
        
        # Detect model type from model object if provided
        if model is not None:
            self.model_type = self._detect_model_type(model)
            self._print(f"🤖 Auto-detected model type: {self.model_type} from {type(model).__name__}")
        else:
            self.model_type = model_type
        
        # Initialize components
        self.dtype_corrector = DataTypeCorrector()
        self.missing_handler = SmartMissingHandler(target_col=target)
        self.outlier_handler = AdvancedOutlierHandler(
            method=outlier_method,
            action=outlier_action,
            target_col=target
        )
        
        # Feature extraction
        self.text_extractor = TextFeatureExtractor()
        self.datetime_extractor = DateTimeFeatureExtractor()
        self.categorical_extractor = CategoricalFeatureExtractor()
        self.numeric_extractor = NumericFeatureExtractor()
        
        # Encoding and scaling
        self.cat_handler = CategoricalHandler(
            target_col=target,
            max_categories=15,
            top_n_categories=10
        )
        # Scaler will be initialized AFTER feature selection (to avoid feature mismatch)
        self.scaler = None
        self.scaler_model_type = model_type  # Store for later use
        
        # Feature selection
        self.variance_filter = FeatureFilter()
        self.feature_selector = FeatureSelector(
            target_col=target,
            max_features=max_features,
            correlation_threshold=0.05
        )
        
        # Imbalance handler (for classification)
        self.imbalance_handler = ImbalanceHandler(
            strategy='auto',
            model_type=model_type
        )
        
        # Report storage
        self.report = {}
        self.original_shape = None
        self.final_train_shape = None
        self.final_test_shape = None
        self.task_type = None
    
    def _print(self, message: str):
        """Print if verbose is True."""
        if self.verbose:
            print(message)
    
    def _detect_model_type(self, model):
        """
        Auto-detect model type from model object.
        
        Detects:
        - Tree-based: RandomForest, XGBoost, LightGBM, DecisionTree, etc.
        - Linear: LinearRegression, LogisticRegression, Ridge, Lasso, etc.
        - Neural Network: MLPClassifier, MLPRegressor, Keras models
        - Distance-based: KNN, SVM
        """
        model_name = type(model).__name__.lower()
        model_module = type(model).__module__.lower()
        
        # Tree-based models (no scaling needed)
        tree_keywords = [
            'tree', 'forest', 'xgb', 'lgbm', 'lightgbm', 'catboost',
            'gradientboosting', 'adaboost', 'extratrees'
        ]
        if any(keyword in model_name for keyword in tree_keywords):
            return 'tree'
        
        # Neural networks (MinMaxScaler)
        nn_keywords = ['mlp', 'neural', 'keras', 'tensorflow', 'torch', 'pytorch']
        if any(keyword in model_name or keyword in model_module for keyword in nn_keywords):
            return 'nn'
        
        # Linear models (StandardScaler)
        linear_keywords = [
            'linear', 'logistic', 'ridge', 'lasso', 'elasticnet',
            'sgd', 'perceptron', 'passiveaggressive'
        ]
        if any(keyword in model_name for keyword in linear_keywords):
            return 'linear'
        
        # Distance-based (StandardScaler)
        distance_keywords = ['knn', 'svm', 'svc', 'svr', 'neighbors']
        if any(keyword in model_name for keyword in distance_keywords):
            return 'linear'  # Use same as linear (StandardScaler)
        
        # Default to auto if can't detect
        return 'auto'
    
    def _load_data(self, data: Union[str, pd.DataFrame]) -> pd.DataFrame:
        """Load data from CSV or use DataFrame."""
        if isinstance(data, str):
            self._print(f"\n📂 Loading data from: {data}")
            df = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError("❌ Data must be a CSV filepath or pandas DataFrame!")
        
        
        unnamed_cols = [col for col in df.columns if col.startswith('Unnamed:')]
        if unnamed_cols:
            df = df.drop(columns=unnamed_cols)
            self._print(f"🗑️  Removed {len(unnamed_cols)} index column(s): {unnamed_cols}")
        
        return df
    
    def _split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and test sets."""
        self._print(f"\n✂️  Splitting data: {int((1-self.test_size)*100)}% train, {int(self.test_size*100)}% test")
        
        # Detect task type for stratification
        task_type = self._detect_task_type(df[self.target])
        
        try:
            if task_type == 'classification':
                train_df, test_df = train_test_split(
                    df,
                    test_size=self.test_size,
                    random_state=self.random_state,
                    stratify=df[self.target],
                    shuffle=True
                )
                self._print(f"   Using stratified split (classification)")
            else:
                train_df, test_df = train_test_split(
                    df,
                    test_size=self.test_size,
                    random_state=self.random_state,
                    shuffle=True
                )
                self._print(f"   Using random split (regression)")
        except Exception as e:
            self._print(f"   Warning: Stratification failed, using random split")
            train_df, test_df = train_test_split(
                df,
                test_size=self.test_size,
                random_state=self.random_state,
                shuffle=True
            )
        
        self._print(f"   Train: {train_df.shape[0]} rows")
        self._print(f"   Test:  {test_df.shape[0]} rows")
        
        return train_df, test_df
    
    def _detect_task_type(self, target_series: pd.Series) -> str:
        """Detect if classification or regression."""
        if pd.api.types.is_numeric_dtype(target_series):
            n_unique = target_series.nunique()
            if n_unique <= 20:
                return 'classification'
            else:
                return 'regression'
        else:
            return 'classification'
    
    def fit_transform(
        self,
        data: Union[str, pd.DataFrame]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict]:
        """
        Main method: Load, clean, split, and return ready-to-use data.
        
        Parameters
        ----------
        data : str or pd.DataFrame
            CSV filepath or pandas DataFrame
        
        Returns
        -------
        X_train : pd.DataFrame
            Training features
        X_test : pd.DataFrame
            Test features
        y_train : pd.Series
            Training target
        y_test : pd.Series
            Test target
        report : dict
            Detailed report of all operations
        
        Examples
        --------
        >>> cleaner = AutoCleanML(target="price")
        >>> X_train, X_test, y_train, y_test, report = cleaner.fit_transform("data.csv")
        >>> 
        >>> # Train model
        >>> from sklearn.ensemble import RandomForestRegressor
        >>> model = RandomForestRegressor()
        >>> model.fit(X_train, y_train)
        """
        # Load data
        df = self._load_data(data)
        self.original_shape = df.shape
        
        # Check target exists
        if self.target not in df.columns:
            raise ValueError(f"❌ Target column '{self.target}' not found in data!")
        
        self._print("\n" + "="*80)
        self._print(" "*20 + "🚀 AutoCleanML - Smart Data Cleaning Pipeline")
        self._print("="*80)
        self._print(f"\n📊 Original dataset: {df.shape[0]} rows × {df.shape[1]} columns")
        self._print(f"🎯 Target column: '{self.target}'")
        self._print(f"📋 Task type: {self._detect_task_type(df[self.target])}")
        
        # Split first (to prevent data leakage)
        train_df, test_df = self._split_data(df)
        
        # Start cleaning pipeline
        self._print("\n" + "="*80)
        self._print(" "*25 + "CLEANING TRAINING DATA")
        self._print("="*80)
        
        # Step 1: Remove duplicates
        self._print("\n🧹 Step 1/11: Removing duplicate rows...")
        before = len(train_df)
        train_df = train_df.drop_duplicates()
        removed = before - len(train_df)
        if removed > 0:
            self._print(f"   ✓ Removed {removed} duplicate rows")
        else:
            self._print(f"   ✓ No duplicates found")
        self.report['duplicates_removed'] = removed
        
        # Step 2: Data type correction
        self._print("\n🔧 Step 2/11: Correcting data types...")
        train_df = self.dtype_corrector.fit_transform(train_df)
        if self.dtype_corrector.numeric_cols or self.dtype_corrector.datetime_cols:
            self._print(f"   ✓ Converted {len(self.dtype_corrector.numeric_cols)} numeric columns")
            self._print(f"   ✓ Converted {len(self.dtype_corrector.datetime_cols)} datetime columns")
        else:
            self._print(f"   ✓ All data types correct")
        
        # Step 3: Smart missing value handling
        self._print("\n💉 Step 3/11: Smart missing value imputation...")
        train_df = self.missing_handler.fit(train_df).transform(train_df)
        self.missing_handler.print_report()
        self.report['missing_values'] = self.missing_handler.get_report()
        
        # Step 4: Outlier handling
        self._print("\n🎯 Step 4/11: Advanced outlier detection...")
        train_df = self.outlier_handler.fit(train_df).transform(train_df)
        self.outlier_handler.print_report()
        self.report['outliers'] = self.outlier_handler.get_report()
        
        # Step 5: Feature extraction (if enabled)
        if self.feature_extraction:
            self._print("\n✨ Step 5/11: Feature extraction (creating new features)...")
            
            # Text features
            train_df = self.text_extractor.fit_transform(train_df)
            text_report = self.text_extractor.get_extraction_report()
            if text_report['total_features_created'] > 0:
                self._print(f"   ✓ Text: Created {text_report['total_features_created']} features")
            
            # Datetime features
            train_df = self.datetime_extractor.fit_transform(train_df)
            dt_report = self.datetime_extractor.get_extraction_report()
            if dt_report['total_features_created'] > 0:
                self._print(f"   ✓ DateTime: Created {dt_report['total_features_created']} features")
            
            # Categorical features
            train_df = self.categorical_extractor.fit_transform(train_df)
            cat_report = self.categorical_extractor.get_extraction_report()
            if cat_report['total_features_created'] > 0:
                self._print(f"   ✓ Categorical: Created {cat_report['total_features_created']} features")
            
            # Numeric features
            train_df = self.numeric_extractor.fit_transform(train_df, target=self.target)
            num_report = self.numeric_extractor.get_extraction_report()
            if num_report['total_features_created'] > 0:
                self._print(f"   ✓ Numeric: Created {num_report['total_features_created']} features")
            
            self.report['feature_engineering'] = {
                'text': text_report,
                'datetime': dt_report,
                'categorical': cat_report,
                'numeric': num_report,
                'total_features_created': (
                    text_report['total_features_created'] +
                    dt_report['total_features_created'] +
                    cat_report['total_features_created'] +
                    num_report['total_features_created']
                )
            }
        else:
            self._print("\n✨ Step 5/11: Feature extraction (skipped)")
        
        # Step 6: Categorical encoding
        self._print("\n🏷️  Step 6/11: Smart categorical encoding...")
        train_df = self.cat_handler.fit(train_df).transform(train_df)
        self._print(f"   ✓ Encoded categorical features")
        
        # Step 7: Feature filtering
        self._print("\n🔍 Step 7/11: Removing zero-variance features...")
        train_df = self.variance_filter.fit(train_df).transform(train_df)
        if self.variance_filter.drop_cols:
            self._print(f"   ✓ Removed {len(self.variance_filter.drop_cols)} constant features")
        
        # Step 8: Feature selection
        self._print(f"\n⚡ Step 8/11: Selecting important features (max {self.max_features})...")
        features_before = len(train_df.columns) - 1
        train_df = self.feature_selector.fit(train_df).transform(train_df)
        features_after = len(train_df.columns) - 1
        
        if features_before != features_after:
            self._print(f"   ✓ Selected {features_after} most important features (from {features_before})")
        self.report['feature_selection'] = {
            'features_before': features_before,
            'features_after': features_after,
            'features_removed': features_before - features_after
        }
        
        # Step 9: Intelligent Scaling (based on model type)
        self._print(f"\n📏 Step 9/11: Intelligent transformation & scaling (model_type='{self.model_type}')...")
        # Separate features and target for scaling
        X_train_temp = train_df.drop(columns=[self.target])
        y_train_temp = train_df[self.target]
        
        # Initialize scaler HERE (after feature selection) to avoid feature mismatch
        self.scaler = FeatureScaler(
            method='auto',
            model_type=self.scaler_model_type,
            exclude_cols=[]  # Already separated target, no need to exclude
        )
        
        X_train_temp = self.scaler.fit(X_train_temp).transform(X_train_temp)
        train_df = pd.concat([X_train_temp, y_train_temp], axis=1)
        
        scaling_report = self.scaler.get_report()
        
        # Show transformation details
        if scaling_report['skewed_features_transformed'] > 0:
            self._print(f"   ✓ Transformed {scaling_report['skewed_features_transformed']} skewed features")
            for detail in scaling_report['transformation_details']:
                self._print(f"      - {detail['column']}: {detail['method']} (skewness was {detail['original_skewness']})")
        
        # Show scaling details
        self._print(f"   ✓ Scaling: {scaling_report['scaling_method']}")
        for reason in scaling_report['reasoning']:
            self._print(f"      {reason}")
        
        self.report['scaling'] = scaling_report
        
        # Step 10: Imbalance Detection (for classification)
        self.task_type = self._detect_task_type(y_train_temp)
        if self.task_type == 'classification':
            self._print(f"\n⚖️  Step 10/11: Checking for class imbalance...")
            X_train_temp = train_df.drop(columns=[self.target])
            y_train_temp = train_df[self.target]
            
            self.imbalance_handler.fit(X_train_temp, y_train_temp)
            imbalance_report = self.imbalance_handler.get_report()
            
            if imbalance_report['is_imbalanced']:
                self._print(f"   ⚠️  Dataset is IMBALANCED!")
                self._print(f"   Class distribution: {imbalance_report['class_distribution']}")
                self._print(f"   Imbalance ratio: {imbalance_report['imbalance_ratio']:.3f}")
                self._print(f"   ✓ Recommended: {imbalance_report['recommended_strategy']}")
                self._print(f"   Reason: {imbalance_report['reasoning']}")
                
                if imbalance_report['class_weights']:
                    self._print(f"\n   💡 Use these class weights in your model:")
                    for cls, weight in imbalance_report['class_weights'].items():
                        self._print(f"      {cls}: {weight:.2f}")
            else:
                self._print(f"   ✓ Dataset is balanced (ratio: {imbalance_report['imbalance_ratio']:.3f})")
            
            self.report['imbalance'] = imbalance_report
        else:
            self._print(f"\n⚖️  Step 10/11: Imbalance check (skipped - regression task)")
        
        # Apply same transformations to test data
        self._print("\n" + "="*80)
        self._print(" "*25 + "CLEANING TEST DATA (Step 11/11)")
        self._print("="*80)
        
        test_df = self.dtype_corrector.transform(test_df)
        test_df = self.missing_handler.transform(test_df)
        test_df = self.outlier_handler.transform(test_df)
        
        if self.feature_extraction:
            test_df = self.text_extractor.transform(test_df)
            test_df = self.datetime_extractor.transform(test_df)
            test_df = self.categorical_extractor.transform(test_df)
            test_df = self.numeric_extractor.transform(test_df)
        
        test_df = self.cat_handler.transform(test_df)
        test_df = self.variance_filter.transform(test_df)
        test_df = self.feature_selector.transform(test_df)
        
        # Apply scaling to test data
        X_test_temp = test_df.drop(columns=[self.target])
        y_test_temp = test_df[self.target]
        X_test_temp = self.scaler.transform(X_test_temp)
        test_df = pd.concat([X_test_temp, y_test_temp], axis=1)
        
        self._print(f"\n✓ Test data cleaned using training parameters (no data leakage)")
        
        # Prepare final outputs
        X_train = train_df.drop(columns=[self.target])
        y_train = train_df[self.target]
        
        X_test = test_df.drop(columns=[self.target])
        y_test = test_df[self.target]
        
        # CRITICAL: Final NaN check and cleanup
        # Check training data
        train_nans = X_train.isna().sum().sum()
        if train_nans > 0:
            self._print(f"\n⚠️  WARNING: {train_nans} NaN values found in training data!")
            self._print("   Performing emergency cleanup...")
            for col in X_train.columns:
                if X_train[col].isna().any():
                    if pd.api.types.is_numeric_dtype(X_train[col]):
                        fill_val = X_train[col].median() if X_train[col].notna().any() else 0
                        X_train[col] = X_train[col].fillna(fill_val)
                        self._print(f"   Fixed '{col}' with {fill_val}")
                    else:
                        fill_val = X_train[col].mode()[0] if X_train[col].notna().any() else 'Unknown'
                        X_train[col] = X_train[col].fillna(fill_val)
                        self._print(f"   Fixed '{col}' with '{fill_val}'")
        
        # Check test data
        test_nans = X_test.isna().sum().sum()
        if test_nans > 0:
            self._print(f"\n⚠️  WARNING: {test_nans} NaN values found in test data!")
            self._print("   Performing emergency cleanup...")
            for col in X_test.columns:
                if X_test[col].isna().any():
                    if pd.api.types.is_numeric_dtype(X_test[col]):
                        # Use training statistics
                        if col in X_train.columns:
                            fill_val = X_train[col].median() if X_train[col].notna().any() else 0
                        else:
                            fill_val = X_test[col].median() if X_test[col].notna().any() else 0
                        X_test[col] = X_test[col].fillna(fill_val)
                        self._print(f"   Fixed '{col}' with {fill_val}")
                    else:
                        if col in X_train.columns:
                            fill_val = X_train[col].mode()[0] if X_train[col].notna().any() else 'Unknown'
                        else:
                            fill_val = X_test[col].mode()[0] if X_test[col].notna().any() else 'Unknown'
                        X_test[col] = X_test[col].fillna(fill_val)
                        self._print(f"   Fixed '{col}' with '{fill_val}'")
        
        # Final verification
        final_train_nans = X_train.isna().sum().sum()
        final_test_nans = X_test.isna().sum().sum()
        
        if final_train_nans > 0 or final_test_nans > 0:
            raise ValueError(
                f"CRITICAL: Still have NaN values after cleanup! "
                f"Train: {final_train_nans}, Test: {final_test_nans}"
            )
        
        self.final_train_shape = X_train.shape
        self.final_test_shape = X_test.shape
        
        # Final report
        self._print("\n" + "="*80)
        self._print(" "*30 + "✅ CLEANING COMPLETE!")
        self._print("="*80)
        
        self._print(f"\n📊 Original data: {self.original_shape[0]} rows × {self.original_shape[1]} columns")
        self._print(f"📊 Training set: {X_train.shape[0]} rows × {X_train.shape[1]} features")
        self._print(f"📊 Test set:     {X_test.shape[0]} rows × {X_test.shape[1]} features")
        self._print(f"\n✅ VERIFIED: Training set has {X_train.isna().sum().sum()} NaN values")
        self._print(f"\n✅ VERIFIED: Test set has {X_test.isna().sum().sum()} NaN values")
        
        feature_change = X_train.shape[1] - (self.original_shape[1] - 1)
        if feature_change > 0:
            self._print(f"\n✨ Added {feature_change} engineered features")
        elif feature_change < 0:
            self._print(f"\n✨ Removed {abs(feature_change)} irrelevant features")
        
        self._print(f"\n🎯 Ready for model training!")
        self._print(f"🎯 No data leakage - test data never seen during training")
        self._print("\n" + "="*80)
        
        # Build final report
        self.report['summary'] = {
            'original_shape': self.original_shape,
            'final_train_shape': self.final_train_shape,
            'final_test_shape': self.final_test_shape,
            'target_column': self.target,
            'task_type': self._detect_task_type(y_train),
            'feature_change': feature_change
        }
        
        return X_train, X_test, y_train, y_test, self.report
