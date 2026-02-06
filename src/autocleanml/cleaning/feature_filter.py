import pandas as pd
from sklearn.feature_selection import VarianceThreshold


class FeatureFilter:
    """
    Removes features with low variance or zero variance.
    
    Features with zero variance (constant values) provide no information.
    Features with very low variance are often not useful.
    
    Parameters
    ----------
    variance_thresh : float, default=0.0
        Variance threshold. Features with variance below this are removed.
        Use 0.0 to remove only zero-variance features.
    
    Examples
    --------
    >>> filter = FeatureFilter(variance_thresh=0.01)
    >>> filter.fit(train_df)
    >>> filtered_df = filter.transform(test_df)
    """
    
    def __init__(self, variance_thresh=0.0):
        self.variance_thresh = variance_thresh
        self.drop_cols = []
        self.variance_selector = None
        self.numeric_cols = []
    
    def fit(self, df):
        """
        Identify features to remove
        
        Parameters
        ----------
        df : pd.DataFrame
            Training data
        
        Returns
        -------
        self
        """
        # Remove zero-variance features (constant columns)
        for col in df.columns:
            if df[col].nunique() <= 1:
                self.drop_cols.append(col)
        
        # If variance threshold > 0, also apply variance-based filtering on numeric columns
        if self.variance_thresh > 0:
            self.numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
            self.numeric_cols = [c for c in self.numeric_cols if c not in self.drop_cols]
            
            if self.numeric_cols:
                self.variance_selector = VarianceThreshold(threshold=self.variance_thresh)
                try:
                    self.variance_selector.fit(df[self.numeric_cols])
                    
                    # Find columns that will be dropped by variance selector
                    mask = self.variance_selector.get_support()
                    low_var_cols = [col for col, keep in zip(self.numeric_cols, mask) if not keep]
                    self.drop_cols.extend(low_var_cols)
                except:
                    # If variance selection fails, just skip it
                    pass
        
        return self
    
    def transform(self, df):
        """
        Remove low-variance features
        
        Parameters
        ----------
        df : pd.DataFrame
            Data to filter
        
        Returns
        -------
        pd.DataFrame
            Filtered DataFrame
        """
        df = df.copy()
        
        # Drop identified columns
        cols_to_drop = [c for c in self.drop_cols if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop, errors="ignore")
        
        return df
    
    def fit_transform(self, df):
        """Fit and transform in one step"""
        self.fit(df)
        return self.transform(df)
