"""
Advanced Outlier Handler
========================

Multiple outlier detection and handling methods with detailed reporting.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Literal


class AdvancedOutlierHandler:
    """
    Advanced outlier detection and handling.
    
    Methods:
    - IQR (Interquartile Range) - default, robust
    - Z-score - for normally distributed data
    - Isolation Forest - for complex multivariate outliers
    
    Actions:
    - Cap (winsorize) - default
    - Remove - if outliers are < threshold
    - Flag - create binary feature
    
    Examples
    --------
    >>> handler = AdvancedOutlierHandler(method='iqr', action='cap')
    >>> clean_df, report = handler.fit_transform(df)
    >>> print(report)
    """
    
    def __init__(
        self,
        method: Literal['iqr', 'zscore', 'isolation_forest', 'auto'] = 'auto',
        action: Literal['cap', 'remove', 'flag'] = 'cap',
        iqr_multiplier: float = 1.5,
        zscore_threshold: float = 3.0,
        isolation_contamination: float = 0.05,
        max_removal_pct: float = 5.0,
        target_col: str = None
    ):
        """
        Parameters
        ----------
        method : str
            Detection method: 'iqr', 'zscore', 'isolation_forest', 'auto'
        action : str
            What to do with outliers: 'cap', 'remove', 'flag'
        iqr_multiplier : float
            IQR multiplier (1.5 = standard, 3.0 = extreme)
        zscore_threshold : float
            Z-score threshold (3.0 = standard)
        isolation_contamination : float
            Expected proportion of outliers (for Isolation Forest)
        max_removal_pct : float
            Maximum % of data to remove (safety check)
        target_col : str
            Target column to exclude from outlier detection
        """
        self.method = method
        self.action = action
        self.iqr_multiplier = iqr_multiplier
        self.zscore_threshold = zscore_threshold
        self.isolation_contamination = isolation_contamination
        self.max_removal_pct = max_removal_pct
        self.target_col = target_col
        
        self.outlier_info_ = {}
        self.outlier_log_ = []
        self.rows_removed_ = []
        self.isolation_forest_ = None
    
    def _detect_method_iqr(self, series):
        """Detect outliers using IQR method"""
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - self.iqr_multiplier * iqr
        upper_bound = q3 + self.iqr_multiplier * iqr
        
        outliers = (series < lower_bound) | (series > upper_bound)
        
        return outliers, lower_bound, upper_bound
    
    def _detect_method_zscore(self, series):
        """Detect outliers using Z-score method"""
        z_scores = np.abs(stats.zscore(series, nan_policy='omit'))
        outliers = z_scores > self.zscore_threshold
        
        # Calculate bounds for capping
        mean = series.mean()
        std = series.std()
        lower_bound = mean - self.zscore_threshold * std
        upper_bound = mean + self.zscore_threshold * std
        
        return outliers, lower_bound, upper_bound
    
    def _choose_method(self, series):
        """Auto-select best method based on data distribution"""
        # Check normality using Shapiro-Wilk test (for small samples)
        if len(series) < 5000:
            try:
                _, p_value = stats.shapiro(series.dropna().sample(min(len(series), 1000)))
                is_normal = p_value > 0.05
            except:
                is_normal = False
        else:
            # Use skewness for large samples
            skew = abs(series.skew())
            is_normal = skew < 0.5
        
        if is_normal:
            return 'zscore', 'data appears normally distributed'
        else:
            return 'iqr', 'data is skewed, IQR is more robust'
    
    def fit(self, df):
        """
        Learn outlier patterns from data.
        
        Parameters
        ----------
        df : pd.DataFrame
            Training data
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target column
        if self.target_col and self.target_col in numeric_cols:
            numeric_cols.remove(self.target_col)
        
        # Isolation Forest (multivariate)
        if self.method == 'isolation_forest':
            from sklearn.ensemble import IsolationForest
            
            self.isolation_forest_ = IsolationForest(
                contamination=self.isolation_contamination,
                random_state=42
            )
            
            # Fit on all numeric columns
            numeric_data = df[numeric_cols].fillna(df[numeric_cols].median())
            self.isolation_forest_.fit(numeric_data)
            
            # Get outlier predictions
            outlier_pred = self.isolation_forest_.predict(numeric_data)
            outlier_mask = outlier_pred == -1
            
            self.outlier_log_.append({
                'method': 'isolation_forest',
                'columns': numeric_cols,
                'outliers_detected': int(outlier_mask.sum()),
                'outlier_pct': f'{(outlier_mask.sum() / len(df)) * 100:.2f}%'
            })
            
            return self
        
        # Univariate methods (IQR, Z-score, auto)
        for col in numeric_cols:
            series = df[col].dropna()
            
            if len(series) < 10:  # Skip if too few values
                continue
            
            # Choose method
            if self.method == 'auto':
                chosen_method, reason = self._choose_method(series)
            else:
                chosen_method = self.method
                reason = f"user-specified method"
            
            # Detect outliers
            if chosen_method == 'iqr':
                outliers, lower_bound, upper_bound = self._detect_method_iqr(df[col])
            elif chosen_method == 'zscore':
                outliers, lower_bound, upper_bound = self._detect_method_zscore(df[col])
            
            outlier_count = outliers.sum()
            outlier_pct = (outlier_count / len(df)) * 100
            
            # Only process if outliers found
            if outlier_count > 0:
                self.outlier_info_[col] = {
                    'method': chosen_method,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'outlier_count': int(outlier_count),
                    'outlier_pct': outlier_pct
                }
                
                self.outlier_log_.append({
                    'column': col,
                    'method': chosen_method,
                    'outliers_detected': int(outlier_count),
                    'outlier_pct': f'{outlier_pct:.2f}%',
                    'action': self.action,
                    'reason': reason
                })
        
        return self
    
    def transform(self, df):
        """
        Handle outliers in data.
        
        Parameters
        ----------
        df : pd.DataFrame
            Data to clean
            
        Returns
        -------
        pd.DataFrame
            Cleaned data
        """
        df = df.copy()
        
        if self.method == 'isolation_forest' and self.isolation_forest_ is not None:
            numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                          if col != self.target_col]
            
            # Get outlier predictions
            numeric_data = df[numeric_cols].fillna(df[numeric_cols].median())
            outlier_pred = self.isolation_forest_.predict(numeric_data)
            outlier_mask = outlier_pred == -1
            
            if self.action == 'remove':
                outlier_pct = (outlier_mask.sum() / len(df)) * 100
                if outlier_pct <= self.max_removal_pct:
                    self.rows_removed_ = df[outlier_mask].index.tolist()
                    df = df[~outlier_mask]
                else:
                    print(f"⚠️  Warning: Would remove {outlier_pct:.1f}% of data, capping instead")
                    # Fall back to capping - handled by univariate methods
            
            elif self.action == 'flag':
                df['is_outlier'] = outlier_mask.astype(int)
            
            return df
        
        # Univariate outlier handling
        for col, info in self.outlier_info_.items():
            if col not in df.columns:
                continue
            
            lower_bound = info['lower_bound']
            upper_bound = info['upper_bound']
            
            if self.action == 'cap':
                # Winsorize (cap at bounds)
                df[col] = df[col].clip(lower_bound, upper_bound)
            
            elif self.action == 'flag':
                # Create binary outlier flag
                outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
                df[f'{col}_is_outlier'] = outliers.astype(int)
            
            elif self.action == 'remove':
                # Remove rows with outliers (risky!)
                outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
                outlier_pct = (outliers.sum() / len(df)) * 100
                
                if outlier_pct <= self.max_removal_pct:
                    self.rows_removed_.extend(df[outliers].index.tolist())
                    df = df[~outliers]
                else:
                    print(f"⚠️  Warning: Column '{col}' has {outlier_pct:.1f}% outliers, capping instead")
                    df[col] = df[col].clip(lower_bound, upper_bound)
        
        return df
    
    def fit_transform(self, df):
        """
        Fit and transform in one step.
        
        Returns
        -------
        tuple
            (cleaned_df, report)
        """
        self.fit(df)
        df_cleaned = self.transform(df)
        
        return df_cleaned, self.get_report()
    
    def get_report(self):
        """Get detailed outlier handling report"""
        return {
            'method': self.method,
            'action': self.action,
            'columns_processed': len(self.outlier_info_),
            'total_outliers': sum(info['outlier_count'] for info in self.outlier_info_.values()),
            'rows_removed': len(set(self.rows_removed_)),
            'details': self.outlier_log_
        }
    
    def print_report(self):
        """Print formatted outlier report"""
        print("\n" + "="*70)
        print(" "*20 + "OUTLIER DETECTION & HANDLING REPORT")
        print("="*70)
        
        print(f"\n📊 Method: {self.method.upper()}")
        print(f"📊 Action: {self.action.upper()}")
        
        if not self.outlier_log_:
            print("\n✓ No outliers detected!")
            return
        
        print(f"\n📊 Columns processed: {len(self.outlier_info_)}")
        total_outliers = sum(info['outlier_count'] for info in self.outlier_info_.values())
        print(f"📊 Total outliers: {total_outliers}")
        
        if self.rows_removed_:
            print(f"📊 Rows removed: {len(set(self.rows_removed_))}")
        
        print("\n" + "-"*70)
        print(f"{'Column':<20} {'Outliers':<15} {'Method':<12} {'Reason'}")
        print("-"*70)
        
        for log in self.outlier_log_:
            col_name = log.get('column', 'Multiple')[:18]
            outlier_info = f"{log['outliers_detected']} ({log['outlier_pct']})"
            method = log['method']
            reason = log.get('reason', '')[:30]
            
            print(f"{col_name:<20} {outlier_info:<15} {method:<12} {reason}")
        
        print("="*70)
