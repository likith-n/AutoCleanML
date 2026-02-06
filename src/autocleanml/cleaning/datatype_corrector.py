import pandas as pd


class DataTypeCorrector:
    """
    Automatically detects and corrects data types in DataFrame.
    
    Converts object columns to:
    - Numeric (int/float) if >threshold% can be converted
    - Datetime if >threshold% can be parsed as dates
    
    Parameters
    ----------
    threshold : float, default=0.8
        Minimum ratio (0-1) of successful conversions required to change type
    
    Examples
    --------
    >>> corrector = DataTypeCorrector(threshold=0.8)
    >>> corrector.fit(train_df)
    >>> fixed_df = corrector.transform(test_df)
    """
    
    def __init__(self, threshold=0.8):
        self.threshold = threshold
        self.numeric_cols = []
        self.datetime_cols = []
    
    def fit(self, df):
        """
        Learn which columns should be converted
        
        Parameters
        ----------
        df : pd.DataFrame
            Training data to learn from
        
        Returns
        -------
        self
        """
        for col in df.columns:
            if df[col].dtype == "object":
                # Try numeric conversion
                converted = pd.to_numeric(df[col], errors="coerce")
                ratio = converted.notna().mean()
                
                if ratio > self.threshold:
                    self.numeric_cols.append(col)
                else:
                    # Try datetime
                    dt = pd.to_datetime(df[col], errors="coerce")
                    if dt.notna().mean() > self.threshold:
                        self.datetime_cols.append(col)
        
        return self
    
    def transform(self, df):
        """
        Apply data type corrections
        
        Parameters
        ----------
        df : pd.DataFrame
            Data to transform
        
        Returns
        -------
        pd.DataFrame
            DataFrame with corrected data types
        """
        df = df.copy()
        
        for col in self.numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        for col in self.datetime_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        
        return df
    
    def fit_transform(self, df):
        """Fit and transform in one step"""
        self.fit(df)
        return self.transform(df)
