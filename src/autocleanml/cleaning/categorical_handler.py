import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


class CategoricalHandler:
    """
    Smart categorical encoding that prevents feature explosion.
    
    Strategy:
    - Low cardinality (<15 unique): One-hot encode all
    - Medium cardinality (15-50): One-hot encode top N most frequent, group rest as 'Other'
    - High cardinality (>50): Use frequency encoding or drop
    
    Parameters
    ----------
    target_col : str, optional
        Name of target column (for correlation-based selection)
    max_categories : int, default=15
        Maximum unique values for full one-hot encoding
    top_n_categories : int, default=10
        For medium cardinality, keep only top N categories
    encoding_strategy : str, default='smart'
        'smart': Adaptive based on cardinality
        'onehot': Force one-hot encoding
        'frequency': Frequency encoding for all
    min_frequency : float, default=0.01
        Minimum frequency (as proportion) to keep a category
    
    Examples
    --------
    >>> handler = CategoricalHandler(
    ...     target_col='price',
    ...     max_categories=15,
    ...     top_n_categories=10
    ... )
    >>> handler.fit(train_df)
    >>> clean_train = handler.transform(train_df)
    """
    
    def __init__(
        self,
        target_col=None,
        max_categories=15,
        top_n_categories=10,
        encoding_strategy='smart',
        min_frequency=0.01,
        drop_first=True
    ):
        self.target_col = target_col
        self.max_categories = max_categories
        self.top_n_categories = top_n_categories
        self.encoding_strategy = encoding_strategy
        self.min_frequency = min_frequency
        self.drop_first = drop_first
        
        # Learned parameters
        self.onehot_cols = []
        self.frequency_cols = {}
        self.drop_cols = []
        self.category_mappings = {}  # For grouping rare categories
        self.onehot_encoder = None
        self.label_encoder = None
        
        # Statistics
        self.feature_count_before = 0
        self.feature_count_after = 0
    
    def fit(self, df):
        """
        Learn encoding strategy for each categorical column.
        
        Parameters
        ----------
        df : pd.DataFrame
            Training data
        
        Returns
        -------
        self
        """
        self.feature_count_before = len(df.columns)
        
        # Identify categorical columns
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove target from encoding
        if self.target_col and self.target_col in cat_cols:
            cat_cols.remove(self.target_col)
        
        for col in cat_cols:
            n_unique = df[col].nunique()
            
            # Decide encoding strategy based on cardinality
            if n_unique <= 1:
                # Constant column - drop it
                self.drop_cols.append(col)
                print(f"   Dropping '{col}': only {n_unique} unique value(s)")
                
            elif n_unique == 2:
                # Binary - simple label encoding
                self.frequency_cols[col] = 'binary'
                
            elif n_unique <= self.max_categories:
                # Low cardinality - one-hot encode all
                self.onehot_cols.append(col)
                
            elif n_unique <= 50:
                # Medium cardinality - one-hot encode top N only
                # Find top N most frequent categories
                top_categories = df[col].value_counts().nlargest(self.top_n_categories).index.tolist()
                
                # Create mapping: keep top N, group rest as 'Other'
                self.category_mappings[col] = top_categories
                self.onehot_cols.append(col)
                
                print(f"   '{col}': {n_unique} unique → keeping top {self.top_n_categories}, grouping rest as 'Other'")
                
            else:
                # High cardinality - use frequency encoding
                freq_map = df[col].value_counts(normalize=True).to_dict()
                self.frequency_cols[col] = freq_map
                
                print(f"   '{col}': {n_unique} unique → using frequency encoding")
        
        # Fit one-hot encoder if we have any columns to encode
        if self.onehot_cols:
            # Apply category grouping first
            df_temp = df.copy()
            for col in self.onehot_cols:
                if col in self.category_mappings:
                    top_cats = self.category_mappings[col]
                    df_temp[col] = df_temp[col].apply(
                        lambda x: x if x in top_cats else 'Other'
                    )
            
            # Fit encoder
            self.onehot_encoder = OneHotEncoder(
                drop='first' if self.drop_first else None,
                sparse_output=False,
                handle_unknown='ignore'
            )
            self.onehot_encoder.fit(df_temp[self.onehot_cols])
        
        # Fit label encoder for target if provided AND it's categorical
        if self.target_col and self.target_col in df.columns:
            # Only encode if target is categorical (not numeric)
            if not pd.api.types.is_numeric_dtype(df[self.target_col]):
                self.label_encoder = LabelEncoder()
                self.label_encoder.fit(df[self.target_col])
            else:
                # Target is numeric (regression), don't encode it
                self.label_encoder = None
        
        return self
    
    def transform(self, df):
        """
        Apply learned encoding to data.
        
        Parameters
        ----------
        df : pd.DataFrame
            Data to transform
        
        Returns
        -------
        pd.DataFrame
            Transformed data with encoded categories
        """
        df = df.copy()
        
        # Drop columns marked for dropping
        df = df.drop(columns=self.drop_cols, errors='ignore')
        
        # Apply frequency encoding
        for col, encoding in self.frequency_cols.items():
            if col not in df.columns:
                continue
                
            if encoding == 'binary':
                # Simple label encoding for binary
                df[col] = LabelEncoder().fit_transform(df[col].astype(str))
            else:
                # Frequency encoding
                df[col] = df[col].map(encoding).fillna(0)
        
        # Apply one-hot encoding
        if self.onehot_cols and self.onehot_encoder:
            # Apply category grouping
            for col in self.onehot_cols:
                if col in self.category_mappings and col in df.columns:
                    top_cats = self.category_mappings[col]
                    df[col] = df[col].apply(
                        lambda x: x if x in top_cats else 'Other'
                    )
            
            # Get columns that actually exist
            cols_to_encode = [c for c in self.onehot_cols if c in df.columns]
            
            if cols_to_encode:
                # Encode
                encoded = self.onehot_encoder.transform(df[cols_to_encode])
                feature_names = self.onehot_encoder.get_feature_names_out(cols_to_encode)
                
                # Create encoded dataframe
                encoded_df = pd.DataFrame(
                    encoded,
                    columns=feature_names,
                    index=df.index
                )
                
                # Drop original categorical columns
                df = df.drop(columns=cols_to_encode)
                
                # Add encoded columns
                df = pd.concat([df, encoded_df], axis=1)
        
        # Encode target if provided and label encoder exists (categorical target only)
        if self.target_col and self.label_encoder and self.target_col in df.columns:
            try:
                df[self.target_col] = self.label_encoder.transform(df[self.target_col])
            except ValueError as e:
                # Handle unseen labels in test set by replacing with most common class
                print(f"   Warning: Test set has unseen categories in '{self.target_col}'. Using most common class.")
                known_classes = set(self.label_encoder.classes_)
                df[self.target_col] = df[self.target_col].apply(
                    lambda x: x if x in known_classes else self.label_encoder.classes_[0]
                )
                df[self.target_col] = self.label_encoder.transform(df[self.target_col])
        
        self.feature_count_after = len(df.columns)
        
        return df
    
    def fit_transform(self, df):
        """Fit and transform in one step."""
        self.fit(df)
        return self.transform(df)
    
    def get_feature_stats(self):
        """Get statistics about encoding."""
        return {
            'features_before': self.feature_count_before,
            'features_after': self.feature_count_after,
            'onehot_encoded': len(self.onehot_cols),
            'frequency_encoded': len(self.frequency_cols),
            'dropped': len(self.drop_cols)
        }
