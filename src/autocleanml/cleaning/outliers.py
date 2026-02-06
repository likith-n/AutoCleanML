import numpy as np
import pandas as pd


class OutlierHandler:
    """
    Handles outliers in numeric columns using IQR method or log transformation.
    
    For highly skewed data: applies log transformation
    For normal-ish data: clips outliers using IQR (Interquartile Range)
    
    Parameters
    ----------
    skew_threshold : float, default=1
        Absolute skewness above which log transformation is applied
    
    Examples
    --------
    >>> handler = OutlierHandler()
    >>> handler.fit(train_df)
    >>> clean_df = handler.transform(test_df)
    """
    
    def __init__(self, skew_threshold=1):
        self.skew_threshold = skew_threshold
        self.iqr_bounds = {}
        self.log_cols = []
    
    def fit(self, df):
        """Learn outlier handling strategy from training data"""
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
        
        for col in numeric_cols:
            # Skip if column is all NaN
            if df[col].isnull().all():
                continue
            
            # Calculate skewness
            skew = df[col].skew()
            
            # Check if log transformation is appropriate
            # Only if highly skewed AND all values are positive
            if abs(skew) > self.skew_threshold and (df[col].dropna() > 0).all():
                self.log_cols.append(col)
            else:
                # Use IQR method for clipping
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                
                # Only store bounds if IQR is not zero
                if iqr > 0:
                    self.iqr_bounds[col] = (
                        q1 - 1.5 * iqr,
                        q3 + 1.5 * iqr
                    )
        
        return self
    
    def transform(self, df):
        """Apply outlier handling to data"""
        df = df.copy()
        
        # Apply log transformation to skewed columns
        for col in self.log_cols:
            if col in df.columns:
                # Extra safety check at transform time
                if (df[col].dropna() > 0).all():
                    df[col] = np.log1p(df[col])
                else:
                    # If negatives/zeros appeared in test data, skip transformation
                    print(f"Warning: Skipping log transform for {col} due to non-positive values")
        
        # Apply IQR clipping
        for col, (low, high) in self.iqr_bounds.items():
            if col in df.columns:
                df[col] = df[col].clip(low, high)
        
        return df
    
    def fit_transform(self, df):
        """Fit and transform in one step"""
        self.fit(df)
        return self.transform(df)
