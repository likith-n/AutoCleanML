"""
Categorical Feature Extraction
==============================

Handles multi-value categorical columns and extracts meaningful features.
"""

import pandas as pd
import numpy as np
from typing import List, Dict


class CategoricalFeatureExtractor:
    """
    Extract features from categorical columns with multiple values.
    
    Handles:
    - Multi-value columns (e.g., "A|B|C" -> has_A, has_B, has_C)
    - Location extraction (e.g., "Bangalore, Karnataka, India" -> city, state, country)
    - Tier classification (Tier-1, Tier-2 cities)
    
    Examples
    --------
    >>> extractor = CategoricalFeatureExtractor()
    >>> new_df = extractor.fit_transform(df)
    """
    
    def __init__(self, max_cardinality=50, min_frequency=0.05):
        """
        Parameters
        ----------
        max_cardinality : int
            Maximum unique values to expand into binary features
        min_frequency : float
            Minimum frequency (0-1) to create a binary feature
        """
        self.max_cardinality = max_cardinality
        self.min_frequency = min_frequency
        
        self.multi_value_columns_ = {}
        self.location_columns_ = []
        self.extraction_log_ = []
    
    def _detect_separator(self, series):
        """Detect common separator in multi-value column"""
        sample = series.dropna().head(100).astype(str)
        separators = ['|', ',', ';', '/']
        
        for sep in separators:
            if sample.str.contains(f'\\{sep}', regex=True).sum() > len(sample) * 0.3:
                return sep
        return None
    
    def _is_location_column(self, series, col_name):
        """Detect if column contains location data"""
        # Check column name
        location_keywords = ['address', 'location', 'city', 'place', 'area']
        if any(kw in col_name.lower() for kw in location_keywords):
            return True
        
        # Check content (sample)
        sample = series.dropna().head(50).astype(str)
        
        # Check if contains comma-separated values (typical of addresses)
        has_commas = sample.str.count(',').mean() >= 1
        
        # Check for common location words
        location_words = ['india', 'bangalore', 'mumbai', 'delhi', 'state', 'country']
        has_location_words = any(
            sample.str.lower().str.contains(word).sum() > len(sample) * 0.2
            for word in location_words
        )
        
        return has_commas and has_location_words
    
    def fit(self, df):
        """
        Detect multi-value and location columns.
        
        Parameters
        ----------
        df : pd.DataFrame
            Training data
        """
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            # Check for multi-value columns
            separator = self._detect_separator(df[col])
            if separator:
                # Extract unique values
                all_values = []
                for val in df[col].dropna():
                    all_values.extend([v.strip() for v in str(val).split(separator)])
                
                # Count frequencies
                from collections import Counter
                value_counts = Counter(all_values)
                
                # Keep only frequent values
                total = sum(value_counts.values())
                min_count = total * self.min_frequency
                
                frequent_values = [
                    val for val, count in value_counts.most_common(self.max_cardinality)
                    if count >= min_count
                ]
                
                if len(frequent_values) > 0 and len(frequent_values) <= self.max_cardinality:
                    self.multi_value_columns_[col] = {
                        'separator': separator,
                        'values': frequent_values
                    }
            
            # Check for location columns
            if self._is_location_column(df[col], col):
                self.location_columns_.append(col)
        
        return self
    
    def transform(self, df):
        """
        Extract features from categorical columns.
        
        Parameters
        ----------
        df : pd.DataFrame
            Data to transform
            
        Returns
        -------
        pd.DataFrame
            DataFrame with expanded categorical features
        """
        df = df.copy()
        
        # Handle multi-value columns
        for col, info in self.multi_value_columns_.items():
            if col not in df.columns:
                continue
            
            separator = info['separator']
            values = info['values']
            created_features = []
            
            for value in values:
                # Create binary feature
                safe_value = value.replace(' ', '_').replace('-', '_').lower()
                feat_name = f'{col}_has_{safe_value}'
                
                df[feat_name] = df[col].astype(str).str.contains(
                    re.escape(value), case=False, regex=True
                ).fillna(False).astype(int)
                
                created_features.append(feat_name)
            
            if created_features:
                self.extraction_log_.append({
                    'original_column': col,
                    'type': 'multi_value',
                    'separator': separator,
                    'features_created': created_features,
                    'count': len(created_features)
                })
        
        # Handle location columns
        for col in self.location_columns_:
            if col not in df.columns:
                continue
            
            created_features = []
            
            # Try to extract city, state, country (format: City, State, Country)
            parts = df[col].astype(str).str.split(',', expand=True)
            
            if parts.shape[1] >= 1:
                df[f'{col}_city'] = parts[0].str.strip()
                created_features.append(f'{col}_city')
            
            if parts.shape[1] >= 2:
                df[f'{col}_state'] = parts[1].str.strip()
                created_features.append(f'{col}_state')
            
            if parts.shape[1] >= 3:
                df[f'{col}_country'] = parts[2].str.strip()
                created_features.append(f'{col}_country')
            
            # Tier classification (India-specific, can be extended)
            tier1_cities = [
                'bangalore', 'mumbai', 'delhi', 'hyderabad', 'chennai',
                'kolkata', 'pune', 'ahmedabad'
            ]
            
            df[f'{col}_is_tier1'] = df[col].astype(str).str.lower().apply(
                lambda x: 1 if any(city in x for city in tier1_cities) else 0
            )
            created_features.append(f'{col}_is_tier1')
            
            if created_features:
                self.extraction_log_.append({
                    'original_column': col,
                    'type': 'location',
                    'features_created': created_features,
                    'count': len(created_features)
                })
        
        return df
    
    def fit_transform(self, df):
        """Fit and transform in one step"""
        self.fit(df)
        return self.transform(df)
    
    def get_extraction_report(self):
        """Get report of features extracted"""
        return {
            'multi_value_columns': list(self.multi_value_columns_.keys()),
            'location_columns': self.location_columns_,
            'total_features_created': sum(log['count'] for log in self.extraction_log_),
            'details': self.extraction_log_
        }


import re  # Add import at top
