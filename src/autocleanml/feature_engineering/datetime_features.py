"""
DateTime Feature Extraction
===========================

Extracts temporal features from datetime columns.
"""

import pandas as pd
import numpy as np
from typing import List, Dict


class DateTimeFeatureExtractor:
    """
    Extract features from datetime columns.
    
    Features Extracted:
    - Year, Month, Day
    - Day of week, Weekend flag
    - Quarter, Week of year
    - Time-based features (if datetime includes time)
    - Days since epoch (for ML algorithms)
    
    Examples
    --------
    >>> extractor = DateTimeFeatureExtractor()
    >>> new_df = extractor.fit_transform(df, datetime_columns=['date', 'timestamp'])
    """
    
    def __init__(self, extract_cyclical=True, extract_time=True):
        """
        Parameters
        ----------
        extract_cyclical : bool
            Extract cyclical features (sin/cos encoding for month, day)
        extract_time : bool
            Extract time-based features (hour, minute) if datetime includes time
        """
        self.extract_cyclical = extract_cyclical
        self.extract_time = extract_time
        
        self.datetime_columns_ = []
        self.extraction_log_ = []
    
    def fit(self, df, datetime_columns=None):
        """
        Identify datetime columns.
        
        Parameters
        ----------
        df : pd.DataFrame
            Training data
        datetime_columns : list, optional
            List of datetime column names. If None, auto-detect.
        """
        if datetime_columns is None:
            # Auto-detect datetime columns
            datetime_columns = list(df.select_dtypes(include=['datetime64']).columns)
            
            # Also check object columns that might be dates
            import warnings
            for col in df.select_dtypes(include=['object']).columns:
                try:
                    # Suppress UserWarning about format inference
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', category=UserWarning)
                        sample = pd.to_datetime(df[col].head(100), errors='coerce')
                    # If >80% successfully converted, it's likely a date
                    if sample.notna().sum() / len(sample) > 0.8:
                        datetime_columns.append(col)
                except:
                    pass
        
        self.datetime_columns_ = datetime_columns
        return self
    
    def transform(self, df):
        """
        Extract datetime features.
        
        Parameters
        ----------
        df : pd.DataFrame
            Data to transform
            
        Returns
        -------
        pd.DataFrame
            DataFrame with new datetime-based features
        """
        df = df.copy()
        
        for col in self.datetime_columns_:
            if col not in df.columns:
                continue
            
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=UserWarning)
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            
            created_features = []
            
            # Basic features
            df[f'{col}_year'] = df[col].dt.year
            df[f'{col}_month'] = df[col].dt.month
            df[f'{col}_day'] = df[col].dt.day
            df[f'{col}_dayofweek'] = df[col].dt.dayofweek
            df[f'{col}_quarter'] = df[col].dt.quarter
            df[f'{col}_weekofyear'] = df[col].dt.isocalendar().week
            
            created_features.extend([
                f'{col}_year', f'{col}_month', f'{col}_day',
                f'{col}_dayofweek', f'{col}_quarter', f'{col}_weekofyear'
            ])
            
            # Boolean features
            df[f'{col}_is_weekend'] = (df[col].dt.dayofweek >= 5).astype(int)
            df[f'{col}_is_month_start'] = df[col].dt.is_month_start.astype(int)
            df[f'{col}_is_month_end'] = df[col].dt.is_month_end.astype(int)
            
            created_features.extend([
                f'{col}_is_weekend', f'{col}_is_month_start', f'{col}_is_month_end'
            ])
            
            # Time features (if datetime has time component)
            if self.extract_time:
                # Check if time component exists (not all midnight)
                has_time = (df[col].dt.hour != 0).any() or (df[col].dt.minute != 0).any()
                
                if has_time:
                    df[f'{col}_hour'] = df[col].dt.hour
                    df[f'{col}_minute'] = df[col].dt.minute
                    df[f'{col}_is_business_hours'] = (
                        (df[col].dt.hour >= 9) & (df[col].dt.hour < 17)
                    ).astype(int)
                    
                    created_features.extend([
                        f'{col}_hour', f'{col}_minute', f'{col}_is_business_hours'
                    ])
            
            # Cyclical encoding (for better ML performance)
            if self.extract_cyclical:
                # Month (1-12) as cyclical
                df[f'{col}_month_sin'] = np.sin(2 * np.pi * df[col].dt.month / 12)
                df[f'{col}_month_cos'] = np.cos(2 * np.pi * df[col].dt.month / 12)
                
                # Day of week (0-6) as cyclical
                df[f'{col}_dayofweek_sin'] = np.sin(2 * np.pi * df[col].dt.dayofweek / 7)
                df[f'{col}_dayofweek_cos'] = np.cos(2 * np.pi * df[col].dt.dayofweek / 7)
                
                created_features.extend([
                    f'{col}_month_sin', f'{col}_month_cos',
                    f'{col}_dayofweek_sin', f'{col}_dayofweek_cos'
                ])
            
            # Days since epoch (useful for ML)
            df[f'{col}_days_since_epoch'] = (
                df[col] - pd.Timestamp('1970-01-01')
            ).dt.days
            created_features.append(f'{col}_days_since_epoch')
            
            # Log extraction
            self.extraction_log_.append({
                'original_column': col,
                'features_created': created_features,
                'count': len(created_features)
            })
            
            # Optionally drop original column
            # df = df.drop(columns=[col])
        
        return df
    
    def fit_transform(self, df, datetime_columns=None):
        """Fit and transform in one step"""
        self.fit(df, datetime_columns)
        return self.transform(df)
    
    def get_extraction_report(self):
        """Get report of features extracted"""
        return {
            'datetime_columns_processed': self.datetime_columns_,
            'total_features_created': sum(log['count'] for log in self.extraction_log_),
            'details': self.extraction_log_
        }
