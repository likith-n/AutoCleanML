import pandas as pd
from sklearn.impute import KNNImputer


class StatImputer:
    """
    Handles missing data imputation using statistical methods or KNN.
    
    Parameters
    ----------
    strategy : str, default="median"
        Imputation strategy: "median", "mean", "mode", "knn"
    n_neighbors : int, default=5
        Number of neighbors for KNN imputation (only used if strategy="knn")
    
    Examples
    --------
    >>> imputer = StatImputer(strategy="median")
    >>> df_clean = imputer.fit_transform(df)
    """
    
    def __init__(self, strategy="median", n_neighbors=5):
        self.strategy = strategy
        self.n_neighbors = n_neighbors
        self.impute_values = {}
        self.knn_imputer = None
        self.numeric_cols = []
        self.categorical_cols = []
    
    def fit(self, df):
        """Learn imputation values from training data"""
        if self.strategy == "knn":
            # KNN imputation only works on numeric data
            self.numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
            if self.numeric_cols:
                self.knn_imputer = KNNImputer(n_neighbors=self.n_neighbors)
                self.knn_imputer.fit(df[self.numeric_cols])
            
            # Handle categorical separately with mode
            self.categorical_cols = df.select_dtypes(exclude=["int64", "float64"]).columns.tolist()
            for col in self.categorical_cols:
                if df[col].isnull().sum() > 0:
                    mode_result = df[col].mode()
                    self.impute_values[col] = mode_result[0] if len(mode_result) > 0 else "MISSING"
        else:
            # Statistical imputation
            for col in df.columns:
                if df[col].isnull().sum() > 0:
                    if df[col].dtype in ["int64", "float64"]:
                        if self.strategy == "mean":
                            val = df[col].mean()
                        elif self.strategy == "median":
                            val = df[col].median()
                        else:  # mode
                            mode_result = df[col].mode()
                            val = mode_result[0] if len(mode_result) > 0 else 0
                        
                        # Handle all-NaN columns
                        self.impute_values[col] = val if not pd.isna(val) else 0
                    else:
                        # Categorical: use mode
                        mode_result = df[col].mode()
                        self.impute_values[col] = mode_result[0] if len(mode_result) > 0 else "MISSING"
        
        return self
    
    def transform(self, df):
        """Apply imputation to data"""
        df = df.copy()
        
        if self.strategy == "knn" and self.knn_imputer:
            # Apply KNN to numeric columns
            if self.numeric_cols:
                df[self.numeric_cols] = self.knn_imputer.transform(df[self.numeric_cols])
            
            # Apply mode to categorical
            for col, value in self.impute_values.items():
                if col in df.columns:
                    df[col].fillna(value, inplace=True)
        else:
            # Apply statistical imputation
            for col, value in self.impute_values.items():
                if col in df.columns:
                    df[col].fillna(value, inplace=True)
        
        return df
    
    def fit_transform(self, df):
        """Fit and transform in one step"""
        self.fit(df)
        return self.transform(df)
