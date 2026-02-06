import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression


class FeatureSelector:
    """
    Select features based on correlation with target and remove redundant features.
    
    This prevents feature explosion and improves model accuracy by keeping only
    relevant features.
    
    Parameters
    ----------
    target_col : str
        Name of target column
    task : str, default='auto'
        'classification', 'regression', or 'auto' (auto-detect)
    max_features : int, default=50
        Maximum number of features to keep
    correlation_threshold : float, default=0.05
        Minimum absolute correlation with target to keep feature
    multicollinearity_threshold : float, default=0.95
        Remove features with correlation > this value with each other
    method : str, default='correlation'
        'correlation': Pearson correlation
        'mutual_info': Mutual information
        'f_test': F-statistic
        'auto': Choose best method
    
    Examples
    --------
    >>> selector = FeatureSelector(
    ...     target_col='price',
    ...     max_features=50,
    ...     correlation_threshold=0.05
    ... )
    >>> selector.fit(train_df)
    >>> clean_train = selector.transform(train_df)
    """
    
    def __init__(
        self,
        target_col,
        task='auto',
        max_features=50,
        correlation_threshold=0.05,
        multicollinearity_threshold=0.95,
        method='correlation'
    ):
        self.target_col = target_col
        self.task = task
        self.max_features = max_features
        self.correlation_threshold = correlation_threshold
        self.multicollinearity_threshold = multicollinearity_threshold
        self.method = method
        
        # Learned parameters
        self.selected_features = []
        self.feature_scores = {}
        self.removed_features = []
        self.features_before = 0
        self.features_after = 0
    
    def _detect_task(self, df):
        """Auto-detect if classification or regression."""
        if self.task != 'auto':
            return self.task
        
        if self.target_col not in df.columns:
            return 'classification'  # default
        
        target = df[self.target_col]
        
        # Check if target is numeric and has many unique values
        if pd.api.types.is_numeric_dtype(target):
            n_unique = target.nunique()
            if n_unique > 10:
                return 'regression'
            else:
                return 'classification'
        else:
            return 'classification'
    
    def fit(self, df):
        """
        Learn which features to keep based on correlation with target.
        
        Parameters
        ----------
        df : pd.DataFrame
            Training data with target column
        
        Returns
        -------
        self
        """
        if self.target_col not in df.columns:
            print(f"Warning: Target column '{self.target_col}' not found. Skipping feature selection.")
            self.selected_features = [c for c in df.columns if c != self.target_col]
            return self
        
        # Detect task type
        detected_task = self._detect_task(df)
        print(f"   Detected task: {detected_task}")
        
        # Separate features and target
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]
        
        self.features_before = len(X.columns)
        
        # Only work with numeric features
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_features) == 0:
            print("   Warning: No numeric features found for selection")
            self.selected_features = X.columns.tolist()
            return self
        
        # Calculate feature scores
        if self.method == 'correlation' or self.method == 'auto':
            # Use correlation
            feature_scores = {}
            for col in numeric_features:
                try:
                    corr = abs(X[col].corr(y))
                    if not np.isnan(corr):
                        feature_scores[col] = corr
                except:
                    pass
        
        elif self.method == 'mutual_info':
            # Use mutual information
            if detected_task == 'classification':
                scores = mutual_info_classif(X[numeric_features], y, random_state=42)
            else:
                scores = mutual_info_regression(X[numeric_features], y, random_state=42)
            
            feature_scores = dict(zip(numeric_features, scores))
        
        elif self.method == 'f_test':
            # Use F-statistic
            if detected_task == 'classification':
                scores = f_classif(X[numeric_features], y)[0]
            else:
                scores = f_regression(X[numeric_features], y)[0]
            
            feature_scores = dict(zip(numeric_features, scores))
        
        self.feature_scores = feature_scores
        
        # Filter by correlation threshold
        important_features = [
            feat for feat, score in feature_scores.items()
            if score >= self.correlation_threshold
        ]
        
        # Sort by score
        important_features = sorted(
            important_features,
            key=lambda x: feature_scores[x],
            reverse=True
        )
        
        # Limit to max_features
        if len(important_features) > self.max_features:
            print(f"   Limiting features: {len(important_features)} → {self.max_features}")
            important_features = important_features[:self.max_features]
        
        # Remove multicollinear features
        if len(important_features) > 1:
            important_features = self._remove_multicollinear(X, important_features)
        
        # Add back non-numeric features (if any)
        non_numeric = [c for c in X.columns if c not in numeric_features]
        
        self.selected_features = important_features + non_numeric
        self.features_after = len(self.selected_features)
        
        # Track removed features
        self.removed_features = [c for c in X.columns if c not in self.selected_features]
        
        print(f"   Feature selection: {self.features_before} → {self.features_after} features")
        if self.removed_features:
            print(f"   Removed {len(self.removed_features)} low-importance features")
        
        return self
    
    def _remove_multicollinear(self, X, features):
        """Remove highly correlated features (keeps feature with higher target correlation)."""
        if len(features) <= 1:
            return features
        
        # Calculate correlation matrix for selected features
        try:
            corr_matrix = X[features].corr().abs()
        except:
            return features
        
        # Find pairs of highly correlated features
        to_remove = set()
        
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                feat_i = features[i]
                feat_j = features[j]
                
                if feat_i in to_remove or feat_j in to_remove:
                    continue
                
                correlation = corr_matrix.loc[feat_i, feat_j]
                
                if correlation > self.multicollinearity_threshold:
                    # Remove feature with lower target correlation
                    score_i = self.feature_scores.get(feat_i, 0)
                    score_j = self.feature_scores.get(feat_j, 0)
                    
                    if score_i >= score_j:
                        to_remove.add(feat_j)
                    else:
                        to_remove.add(feat_i)
        
        if to_remove:
            print(f"   Removed {len(to_remove)} multicollinear features")
        
        return [f for f in features if f not in to_remove]
    
    def transform(self, df):
        """
        Keep only selected features.
        
        Parameters
        ----------
        df : pd.DataFrame
            Data to transform
        
        Returns
        -------
        pd.DataFrame
            Data with only selected features (and target)
        """
        df = df.copy()
        
        # Keep selected features + target
        cols_to_keep = self.selected_features.copy()
        if self.target_col in df.columns and self.target_col not in cols_to_keep:
            cols_to_keep.append(self.target_col)
        
        # Only keep columns that exist in df
        cols_to_keep = [c for c in cols_to_keep if c in df.columns]
        
        return df[cols_to_keep]
    
    def fit_transform(self, df):
        """Fit and transform in one step."""
        self.fit(df)
        return self.transform(df)
    
    def get_feature_importance(self, top_n=20):
        """
        Get top N most important features.
        
        Returns
        -------
        pd.DataFrame
            Features sorted by importance
        """
        if not self.feature_scores:
            return pd.DataFrame()
        
        scores_df = pd.DataFrame(
            list(self.feature_scores.items()),
            columns=['Feature', 'Score']
        )
        
        return scores_df.sort_values('Score', ascending=False).head(top_n)
