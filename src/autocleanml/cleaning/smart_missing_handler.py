"""
Smart Missing Value Handler
===========================

Context-aware missing value imputation with detailed logging.
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from typing import Dict, List


class SmartMissingHandler:
    """
    Smart, context-aware missing value imputation.
    
    Strategy Selection:
    - Numerical columns:
        * If skewed (skew > threshold) → median
        * If correlated with other features → KNN imputation
        * If time-series → forward/backward fill
        * Otherwise → mean
    
    - Categorical columns:
        * Low cardinality → most frequent
        * High cardinality → create 'Missing' category
    
    - Target column:
        * NEVER impute → drop rows or warn user
    
    Examples
    --------
    >>> handler = SmartMissingHandler(target_col='price')
    >>> clean_df, report = handler.fit_transform(df)
    >>> print(report)
    """
    
    def __init__(self, target_col=None, skew_threshold=1.0, 
                 correlation_threshold=0.3, knn_neighbors=5,
                 high_cardinality_threshold=50):
        """
        Parameters
        ----------
        target_col : str, optional
            Target column name (won't be imputed)
        skew_threshold : float
            Skewness threshold for median vs mean
        correlation_threshold : float
            Correlation threshold for KNN imputation
        knn_neighbors : int
            Number of neighbors for KNN imputation
        high_cardinality_threshold : int
            Threshold for high cardinality categorical columns
        """
        self.target_col = target_col
        self.skew_threshold = skew_threshold
        self.correlation_threshold = correlation_threshold
        self.knn_neighbors = knn_neighbors
        self.high_cardinality_threshold = high_cardinality_threshold
        
        self.imputation_strategies_ = {}
        self.imputers_ = {}
        self.imputation_log_ = []
        self.target_missing_rows_ = []
    
    def _get_numeric_strategy(self, df, col):
        """Determine best imputation strategy for numeric column"""
        series = df[col]
        
        # Calculate skewness
        skew = series.skew()
        
        # Check if time-series (increasing/decreasing trend)
        is_timeseries = False
        if len(series) > 10:
            # Simple trend detection
            diff = series.diff().dropna()
            if (diff > 0).sum() / len(diff) > 0.7 or (diff < 0).sum() / len(diff) > 0.7:
                is_timeseries = True
        
        # Check correlation with other numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c != col and c != self.target_col]
        
        max_correlation = 0
        if len(numeric_cols) > 0:
            correlations = df[numeric_cols + [col]].corr()[col].drop(col).abs()
            max_correlation = correlations.max() if len(correlations) > 0 else 0
        
        # Decision logic
        if is_timeseries:
            reason = "detected time-series pattern"
            return 'ffill', reason
        
        elif abs(skew) > self.skew_threshold:
            reason = f"right skew detected (skew={skew:.2f})"
            return 'median', reason
        
        elif max_correlation > self.correlation_threshold and len(numeric_cols) >= 3:
            reason = f"high correlation with other features (max corr={max_correlation:.2f})"
            return 'knn', reason
        
        else:
            reason = "default strategy for normal distribution"
            return 'mean', reason
    
    def _get_categorical_strategy(self, df, col):
        """Determine best imputation strategy for categorical column"""
        n_unique = df[col].nunique()
        
        if n_unique <= 10:
            reason = f"low cardinality ({n_unique} unique values)"
            return 'most_frequent', reason
        
        elif n_unique <= self.high_cardinality_threshold:
            reason = f"medium cardinality ({n_unique} unique values)"
            return 'most_frequent', reason
        
        else:
            reason = f"high cardinality ({n_unique} unique values)"
            return 'missing_category', reason
    
    def fit(self, df):
        """
        Learn imputation strategies from data.
        
        Parameters
        ----------
        df : pd.DataFrame
            Training data
        """
        # Check target column for missing values
        if self.target_col and self.target_col in df.columns:
            target_missing = df[self.target_col].isna()
            if target_missing.any():
                self.target_missing_rows_ = df[target_missing].index.tolist()
                print(f"⚠️  WARNING: Target column '{self.target_col}' has {target_missing.sum()} missing values!")
                print(f"   Recommendation: Drop these rows before training")
        
        # Analyze each column with missing values
        for col in df.columns:
            if col == self.target_col:
                continue  # Skip target column
            
            missing_count = df[col].isna().sum()
            if missing_count == 0:
                continue
            
            missing_pct = (missing_count / len(df)) * 100
            
            # Determine strategy based on dtype
            if pd.api.types.is_numeric_dtype(df[col]):
                strategy, reason = self._get_numeric_strategy(df, col)
                
                # Create imputer
                if strategy == 'knn':
                    # Use only numeric columns for KNN
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    if self.target_col in numeric_cols:
                        numeric_cols.remove(self.target_col)
                    
                    self.imputers_[col] = {
                        'type': 'knn',
                        'imputer': KNNImputer(n_neighbors=self.knn_neighbors),
                        'columns': numeric_cols
                    }
                
                elif strategy in ['mean', 'median']:
                    self.imputers_[col] = {
                        'type': strategy,
                        'imputer': SimpleImputer(strategy=strategy)
                    }
                
                elif strategy == 'ffill':
                    self.imputers_[col] = {
                        'type': 'ffill',
                        'imputer': None
                    }
            
            else:  # Categorical
                strategy, reason = self._get_categorical_strategy(df, col)
                
                if strategy == 'most_frequent':
                    self.imputers_[col] = {
                        'type': 'most_frequent',
                        'imputer': SimpleImputer(strategy='most_frequent')
                    }
                
                elif strategy == 'missing_category':
                    self.imputers_[col] = {
                        'type': 'missing_category',
                        'imputer': None,
                        'fill_value': 'Missing'
                    }
            
            # Fit imputer
            if col in self.imputers_:
                imputer_info = self.imputers_[col]
                
                if imputer_info['type'] == 'knn':
                    # Fit on numeric columns only
                    numeric_data = df[imputer_info['columns']].copy()
                    imputer_info['imputer'].fit(numeric_data)
                
                elif imputer_info['imputer'] is not None:
                    imputer_info['imputer'].fit(df[[col]])
                
                # Log the decision
                self.imputation_log_.append({
                    'column': col,
                    'missing_count': int(missing_count),
                    'missing_pct': f'{missing_pct:.1f}%',
                    'strategy': strategy,
                    'reason': reason
                })
        
        return self
    
    def transform(self, df):
        """
        Apply imputation to data.
        
        Parameters
        ----------
        df : pd.DataFrame
            Data to impute
            
        Returns
        -------
        pd.DataFrame
            Imputed data
        """
        df = df.copy()
        
        # Drop rows with missing target (if any)
        if self.target_col and self.target_col in df.columns:
            before_len = len(df)
            df = df.dropna(subset=[self.target_col])
            after_len = len(df)
            
            if before_len != after_len:
                print(f"   ✓ Dropped {before_len - after_len} rows with missing target values")
        
        # Apply imputation
        for col, imputer_info in self.imputers_.items():
            if col not in df.columns:
                continue
            
            strategy = imputer_info['type']
            
            try:
                if strategy == 'knn':
                    # KNN imputation
                    numeric_cols = imputer_info['columns']
                    # Only use columns that exist in current df
                    numeric_cols = [c for c in numeric_cols if c in df.columns]
                    
                    if len(numeric_cols) > 0 and col in numeric_cols:
                        numeric_data = df[numeric_cols].copy()
                        imputed_data = imputer_info['imputer'].transform(numeric_data)
                        
                        # Replace only the target column
                        col_idx = numeric_cols.index(col)
                        df[col] = imputed_data[:, col_idx]
                    else:
                        # Fallback to median if KNN can't be applied
                        df[col] = df[col].fillna(df[col].median())
                
                elif strategy == 'ffill':
                    # Forward fill (use ffill() and bfill() directly)
                    df[col] = df[col].ffill().bfill()
                    # If still has NaN (all NaN column), fill with median or 0
                    if df[col].isna().any():
                        if pd.api.types.is_numeric_dtype(df[col]):
                            df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0)
                        else:
                            df[col] = df[col].fillna('Unknown')
                
                elif strategy == 'missing_category':
                    # Fill with 'Missing'
                    df[col] = df[col].fillna(imputer_info['fill_value'])
                
                elif imputer_info['imputer'] is not None:
                    # Simple imputation (mean, median, most_frequent)
                    imputed = imputer_info['imputer'].transform(df[[col]])
                    df[col] = imputed[:, 0]
            
            except Exception as e:
                # Fallback imputation if strategy fails
                print(f"   Warning: Failed to impute '{col}' with {strategy}, using fallback")
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0)
                else:
                    df[col] = df[col].fillna(df[col].mode()[0] if df[col].notna().any() else 'Unknown')
        
        # Final safety check - fill any remaining NaN values
        for col in df.columns:
            if col == self.target_col:
                continue
            if df[col].isna().any():
                if pd.api.types.is_numeric_dtype(df[col]):
                    fill_value = df[col].median() if df[col].notna().any() else 0
                    df[col] = df[col].fillna(fill_value)
                    print(f"   Warning: Filled remaining NaN in '{col}' with {fill_value}")
                else:
                    fill_value = df[col].mode()[0] if df[col].notna().any() else 'Unknown'
                    df[col] = df[col].fillna(fill_value)
                    print(f"   Warning: Filled remaining NaN in '{col}' with '{fill_value}'")
        
        return df
    
    def fit_transform(self, df):
        """
        Fit and transform in one step.
        
        Returns
        -------
        tuple
            (imputed_df, report)
        """
        self.fit(df)
        df_imputed = self.transform(df)
        
        return df_imputed, self.get_report()
    
    def get_report(self):
        """Get detailed imputation report"""
        return {
            'target_column': self.target_col,
            'target_missing_rows': len(self.target_missing_rows_),
            'columns_imputed': len(self.imputation_log_),
            'total_missing_cells': sum(log['missing_count'] for log in self.imputation_log_),
            'details': self.imputation_log_
        }
    
    def print_report(self):
        """Print formatted imputation report"""
        print("\n" + "="*70)
        print(" "*20 + "MISSING VALUE IMPUTATION REPORT")
        print("="*70)
        
        if self.target_missing_rows_:
            print(f"\n⚠️  Target column '{self.target_col}': {len(self.target_missing_rows_)} missing values")
            print(f"   → These rows will be dropped")
        
        if not self.imputation_log_:
            print("\n✓ No missing values detected!")
            return
        
        print(f"\n📊 Columns with missing values: {len(self.imputation_log_)}")
        print(f"📊 Total missing cells: {sum(log['missing_count'] for log in self.imputation_log_)}")
        
        print("\n" + "-"*70)
        print(f"{'Column':<25} {'Missing':<12} {'Strategy':<15} {'Reason'}")
        print("-"*70)
        
        for log in self.imputation_log_:
            col_name = log['column'][:23]
            missing_info = f"{log['missing_count']} ({log['missing_pct']})"
            strategy = log['strategy']
            reason = log['reason'][:35]
            
            print(f"{col_name:<25} {missing_info:<12} {strategy:<15} {reason}")
        
        print("="*70)
