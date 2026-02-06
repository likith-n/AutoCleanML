"""
Numeric Feature Engineering
===========================

Creates polynomial and interaction features from numeric columns.
"""

import pandas as pd
import numpy as np
from typing import List, Dict


class NumericFeatureExtractor:
    """
    Create derived features from numeric columns.
    
    Features Created:
    - Polynomial features (squared, sqrt, log)
    - Ratios between correlated features
    - Binning/discretization
    
    Examples
    --------
    >>> extractor = NumericFeatureExtractor()
    >>> new_df = extractor.fit_transform(df, target='price')
    """
    
    def __init__(self, create_polynomials=True, create_ratios=True,
                 polynomial_degree=2, ratio_threshold=0.5):
        """
        Parameters
        ----------
        create_polynomials : bool
            Create squared, sqrt, log features
        create_ratios : bool
            Create ratio features for correlated columns
        polynomial_degree : int
            Degree for polynomial features (2 = squared)
        ratio_threshold : float
            Correlation threshold for creating ratios
        """
        self.create_polynomials = create_polynomials
        self.create_ratios = create_ratios
        self.polynomial_degree = polynomial_degree
        self.ratio_threshold = ratio_threshold
        
        self.polynomial_cols_ = []
        self.ratio_pairs_ = []
        self.extraction_log_ = []
    
    def fit(self, df, target=None):
        """
        Identify numeric columns for feature engineering.
        
        Parameters
        ----------
        df : pd.DataFrame
            Training data
        target : str, optional
            Target column (excluded from feature engineering)
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target if specified
        if target and target in numeric_cols:
            numeric_cols.remove(target)
        
        # Select columns for polynomial features (skip binary columns)
        for col in numeric_cols:
            n_unique = df[col].nunique()
            if n_unique > 10:  # Not binary or low-cardinality
                self.polynomial_cols_.append(col)
        
        # Find correlated pairs for ratio features
        if self.create_ratios and len(numeric_cols) > 1:
            correlation_matrix = df[numeric_cols].corr().abs()
            
            # Find pairs with correlation above threshold
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    col1, col2 = numeric_cols[i], numeric_cols[j]
                    corr = correlation_matrix.iloc[i, j]
                    
                    if corr >= self.ratio_threshold:
                        self.ratio_pairs_.append((col1, col2))
        
        return self
    
    def transform(self, df):
        """
        Create engineered features.
        
        Parameters
        ----------
        df : pd.DataFrame
            Data to transform
            
        Returns
        -------
        pd.DataFrame
            DataFrame with engineered numeric features
        """
        df = df.copy()
        
        # Polynomial features
        if self.create_polynomials:
            created_features = []
            
            for col in self.polynomial_cols_:
                if col not in df.columns:
                    continue
                
                # Squared
                if self.polynomial_degree >= 2:
                    df[f'{col}_squared'] = df[col] ** 2
                    created_features.append(f'{col}_squared')
                
                # Square root (only for non-negative)
                if (df[col] >= 0).all():
                    df[f'{col}_sqrt'] = np.sqrt(df[col])
                    created_features.append(f'{col}_sqrt')
                
                # Log (only for positive)
                if (df[col] > 0).all():
                    df[f'{col}_log'] = np.log1p(df[col])
                    created_features.append(f'{col}_log')
            
            if created_features:
                self.extraction_log_.append({
                    'type': 'polynomial',
                    'features_created': created_features,
                    'count': len(created_features)
                })
        
        # Ratio features
        if self.create_ratios:
            created_features = []
            
            for col1, col2 in self.ratio_pairs_:
                if col1 not in df.columns or col2 not in df.columns:
                    continue
                
                # Avoid division by zero
                if (df[col2] == 0).any():
                    continue
                
                feat_name = f'{col1}_div_{col2}'
                df[feat_name] = df[col1] / df[col2]
                created_features.append(feat_name)
            
            if created_features:
                self.extraction_log_.append({
                    'type': 'ratio',
                    'features_created': created_features,
                    'count': len(created_features)
                })
        
        return df
    
    def fit_transform(self, df, target=None):
        """Fit and transform in one step"""
        self.fit(df, target)
        return self.transform(df)
    
    def get_extraction_report(self):
        """Get report of features extracted"""
        return {
            'polynomial_columns': self.polynomial_cols_,
            'ratio_pairs': self.ratio_pairs_,
            'total_features_created': sum(log['count'] for log in self.extraction_log_),
            'details': self.extraction_log_
        }
