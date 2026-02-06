import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, QuantileTransformer
from scipy import stats


class FeatureScaler:
    """
    Intelligent feature scaling and transformation based on model type and data distribution.
    
    Transformations Applied (in order):
    1. Log/Power Transform for highly skewed data (optional, auto-detected)
    2. Scaling: StandardScaler, MinMaxScaler, or RobustScaler
    
    Strategy:
    - Tree-based models: NO transformation (scale-invariant)
    - Linear models: 
        * Highly skewed → Log/PowerTransform first, then StandardScaler
        * Normal → StandardScaler only
        * Outliers → RobustScaler
    - Neural Networks: MinMaxScaler (0-1 bounded)
    - Distance-based (KNN, SVM): Same as linear models
    """

    def __init__(self, method="auto", model_type="auto", exclude_cols=None, handle_skewness=True):
        """
        Parameters
        ----------
        method : str
            'auto', 'standard', 'minmax', 'robust'
        model_type : str
            'auto', 'linear', 'tree', 'nn', 'knn', 'svm'
        exclude_cols : list
            Columns to exclude from scaling
        handle_skewness : bool
            Whether to apply log/power transform for skewed data
        """
        self.method = method
        self.model_type = model_type
        self.exclude_cols = exclude_cols or []
        self.handle_skewness = handle_skewness
        
        self.scaler = None
        self.transformers = {}  # Store one transformer per column (for yeo-johnson)
        self.numeric_cols = []
        self.skewed_cols = []  # Columns that were transformed
        
        self.scaling_method_used = None
        self.transformation_applied = None
        self.reasoning = []

    def _check_skewness(self, series):
        """Check if data is highly skewed (>1 or <-1)"""
        try:
            skew = series.skew()
            return abs(skew) > 1.0, skew
        except:
            return False, 0

    def _detect_outliers(self, df, col):
        """Check if column has outliers using IQR method"""
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outlier_count = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
        return outlier_count > len(df) * 0.05  # >5% outliers

    def _apply_skewness_transformation(self, df):
        """
        Apply log or power transformation to highly skewed features.
        
        Strategy:
        - If all values > 0: Try log transform
        - If some values <= 0: Use Yeo-Johnson power transform
        - Only transform if skewness > 1.0
        """
        df = df.copy()
        
        if not self.handle_skewness:
            return df
        
        for col in self.numeric_cols:
            if col not in df.columns:
                continue
            
            is_skewed, skew_value = self._check_skewness(df[col])
            
            if is_skewed:
                # Check if all values are positive
                if (df[col] > 0).all():
                    # Log transform (log1p handles zeros)
                    df[col] = np.log1p(df[col])
                    self.skewed_cols.append((col, 'log', skew_value))
                    self.reasoning.append(
                        f"{col}: Applied log transform (original skewness={skew_value:.2f})"
                    )
                else:
                    # Yeo-Johnson transform (handles negative values)
                    # Create a NEW transformer for this specific column
                    transformer = PowerTransformer(method='yeo-johnson', standardize=False)
                    df[[col]] = transformer.fit_transform(df[[col]])
                    
                    # Store the transformer for this column
                    self.transformers[col] = transformer
                    
                    self.skewed_cols.append((col, 'yeo-johnson', skew_value))
                    self.reasoning.append(
                        f"{col}: Applied Yeo-Johnson transform (original skewness={skew_value:.2f})"
                    )
        
        if self.skewed_cols:
            self.transformation_applied = f"Transformed {len(self.skewed_cols)} skewed features"
        else:
            self.transformation_applied = "No transformation needed (data not skewed)"
        
        return df

    def _choose_scaler(self, df):
        """Choose best scaler based on model type and data characteristics"""
        
        # Manual override
        if self.method != "auto":
            scaler_map = {
                "standard": StandardScaler(),
                "minmax": MinMaxScaler(),
                "robust": RobustScaler(),
                "quantile": QuantileTransformer(output_distribution='normal')
            }
            self.scaling_method_used = self.method
            self.reasoning.append(f"User specified '{self.method}' scaling")
            return scaler_map.get(self.method, StandardScaler())
        
        # Tree-based models DON'T need scaling
        if self.model_type in ["tree", "rf", "random_forest", "xgboost", "lightgbm", "catboost", "decision_tree"]:
            self.scaling_method_used = "none"
            self.reasoning.append("Tree-based models are scale-invariant, no scaling needed")
            return None
        
        # Neural Networks need [0,1] bounded features
        if self.model_type in ["nn", "neural_network", "deep", "mlp"]:
            self.scaling_method_used = "minmax"
            self.reasoning.append("Neural networks: MinMaxScaler for [0,1] bounded features (better for activation functions)")
            return MinMaxScaler()
        
        # For linear and distance-based models, check data characteristics
        if self.model_type in ["linear", "logistic", "ridge", "lasso", "elasticnet", "svm", "knn", "auto"]:
            # Check for outliers in numeric columns
            has_outliers = False
            outlier_cols = []
            
            if len(self.numeric_cols) > 0:
                for col in self.numeric_cols[:10]:  # Check first 10 columns
                    if col in df.columns and self._detect_outliers(df, col):
                        has_outliers = True
                        outlier_cols.append(col)
            
            if has_outliers:
                self.scaling_method_used = "robust"
                self.reasoning.append(
                    f"Data has outliers in {len(outlier_cols)} columns → RobustScaler "
                    f"(uses median and IQR, less sensitive to outliers)"
                )
                return RobustScaler()
            else:
                self.scaling_method_used = "standard"
                self.reasoning.append(
                    "Linear/distance-based model with clean data → StandardScaler "
                    "(assumes normal distribution, zero mean, unit variance)"
                )
                return StandardScaler()
        
        # Default fallback
        self.scaling_method_used = "standard"
        self.reasoning.append("Default: StandardScaler (standardization to zero mean, unit variance)")
        return StandardScaler()

    def fit(self, df):
        """
        Learn transformation and scaling parameters from training data.
        
        Process:
        1. Detect skewed features and apply log/power transform
        2. Choose appropriate scaler based on model type
        3. Fit scaler on transformed data
        """
        df = df.copy()
        
        # Identify numeric columns (excluding target)
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.numeric_cols = [c for c in self.numeric_cols if c not in self.exclude_cols]
        
        if len(self.numeric_cols) == 0:
            self.scaler = None
            self.scaling_method_used = "none"
            self.transformation_applied = "none"
            self.reasoning.append("No numeric columns to scale")
            return self
        
        # Step 1: Handle skewed data (log/power transform)
        if self.handle_skewness and self.model_type not in ["tree", "rf", "random_forest", "xgboost", "lightgbm", "catboost"]:
            df = self._apply_skewness_transformation(df)
        else:
            self.transformation_applied = "Skipped (tree-based model or disabled)"
        
        # Step 2: Choose and fit scaler
        self.scaler = self._choose_scaler(df)
        
        if self.scaler:
            self.scaler.fit(df[self.numeric_cols])
        
        return self

    def transform(self, df):
        """
        Apply learned transformations and scaling to data.
        
        Process:
        1. Apply same log/power transforms as in fit()
        2. Apply scaling
        """
        df = df.copy()
        
        # Step 1: Apply skewness transformations
        if self.skewed_cols:
            for col, method, _ in self.skewed_cols:
                if col not in df.columns:
                    continue
                
                if method == 'log':
                    df[col] = np.log1p(df[col])
                elif method == 'yeo-johnson':
                    # Use the transformer that was fitted for THIS specific column
                    if col in self.transformers:
                        df[[col]] = self.transformers[col].transform(df[[col]])
        
        # Step 2: Apply scaling
        if self.scaler and len(self.numeric_cols) > 0:
            cols_to_scale = [c for c in self.numeric_cols if c in df.columns]
            if cols_to_scale:
                df[cols_to_scale] = self.scaler.transform(df[cols_to_scale])
        
        return df

    def fit_transform(self, df):
        """Fit and transform in one step"""
        self.fit(df)
        return self.transform(df)
    
    def get_report(self):
        """Get detailed transformation and scaling report"""
        return {
            'transformation_applied': self.transformation_applied,
            'skewed_features_transformed': len(self.skewed_cols),
            'transformation_details': [
                {
                    'column': col,
                    'method': method,
                    'original_skewness': round(skew, 2)
                }
                for col, method, skew in self.skewed_cols
            ] if self.skewed_cols else [],
            'scaling_method': self.scaling_method_used,
            'columns_scaled': len(self.numeric_cols),
            'reasoning': self.reasoning
        }
