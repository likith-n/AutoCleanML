"""
Imbalanced Dataset Handler for Classification
=============================================

Handles class imbalance using intelligent strategies based on dataset characteristics.
"""

import pandas as pd
import numpy as np
from collections import Counter


class ImbalanceHandler:
    """
    Handles imbalanced classification datasets using appropriate techniques.
    
    Strategies:
    1. Class Weights: For tree-based models (no resampling needed)
    2. SMOTE: For synthetic oversampling (when minority < 20%)
    3. Random Undersampling: When majority is too large
    4. Balanced: Combination approach
    
    Auto-detection:
    - Calculates imbalance ratio
    - Recommends best strategy based on:
      * Dataset size
      * Imbalance severity
      * Model type
    """
    
    def __init__(self, strategy='auto', model_type='auto', threshold=0.3):
        """
        Parameters
        ----------
        strategy : str
            'auto', 'class_weight', 'oversample', 'undersample', 'smote', 'none'
        model_type : str
            'tree', 'linear', 'nn', 'auto'
        threshold : float
            Imbalance threshold (minority_class / majority_class)
        """
        self.strategy = strategy
        self.model_type = model_type
        self.threshold = threshold
        
        self.is_imbalanced = False
        self.imbalance_ratio = None
        self.class_distribution = None
        self.recommended_strategy = None
        self.reasoning = None
        self.action_taken = None
        
    def _calculate_imbalance(self, y):
        """Calculate class imbalance ratio"""
        class_counts = Counter(y)
        self.class_distribution = dict(class_counts)
        
        if len(class_counts) < 2:
            return 1.0  # Only one class
        
        min_count = min(class_counts.values())
        max_count = max(class_counts.values())
        
        return min_count / max_count if max_count > 0 else 0
    
    def _detect_imbalance_strategy(self, y, sample_size):
        """
        Intelligently choose imbalance handling strategy
        
        Decision Logic:
        1. If ratio > 0.7 → Not imbalanced, do nothing
        2. If ratio 0.3-0.7 → Mild imbalance
           - Tree models: use class_weight
           - Others: mild oversampling
        3. If ratio < 0.3 → Severe imbalance
           - Small dataset (<1000): SMOTE (synthetic samples)
           - Large dataset: class_weight or undersample
        """
        
        if self.imbalance_ratio > 0.7:
            return 'none', 'Dataset is balanced (ratio > 0.7), no action needed'
        
        # Mild imbalance (30-70%)
        if self.imbalance_ratio > self.threshold:
            if self.model_type in ['tree', 'rf', 'random_forest', 'xgboost', 'lightgbm']:
                return 'class_weight', f'Mild imbalance (ratio={self.imbalance_ratio:.2f}), using class_weight for tree-based model'
            else:
                return 'oversample', f'Mild imbalance (ratio={self.imbalance_ratio:.2f}), using random oversampling'
        
        # Severe imbalance (<30%)
        else:
            if sample_size < 1000:
                return 'smote', f'Severe imbalance (ratio={self.imbalance_ratio:.2f}) with small dataset, using SMOTE for synthetic samples'
            else:
                if self.model_type in ['tree', 'rf', 'random_forest', 'xgboost', 'lightgbm']:
                    return 'class_weight', f'Severe imbalance (ratio={self.imbalance_ratio:.2f}), using class_weight (efficient for large tree-based models)'
                else:
                    return 'combined', f'Severe imbalance (ratio={self.imbalance_ratio:.2f}), using combined over+undersampling'
    
    def fit(self, X, y):
        """
        Analyze dataset and determine strategy
        
        Parameters
        ----------
        X : pd.DataFrame
            Features
        y : pd.Series
            Target labels
        """
        # Calculate imbalance
        self.imbalance_ratio = self._calculate_imbalance(y)
        self.is_imbalanced = self.imbalance_ratio < 0.7
        
        # Determine strategy
        if self.strategy == 'auto':
            self.recommended_strategy, self.reasoning = self._detect_imbalance_strategy(y, len(y))
        else:
            self.recommended_strategy = self.strategy
            self.reasoning = f'User specified strategy: {self.strategy}'
        
        return self
    
    def transform(self, X, y):
        """
        Apply imbalance handling strategy
        
        Note: This returns recommendations for model training, not resampled data
        (because resampling should only be done on training set)
        """
        if not self.is_imbalanced or self.recommended_strategy == 'none':
            self.action_taken = 'none'
            return X, y
        
        self.action_taken = self.recommended_strategy
        
        # For now, we return original data and provide recommendations
        # Actual resampling should be done by user if needed
        return X, y
    
    def fit_transform(self, X, y):
        """Fit and transform in one step"""
        self.fit(X, y)
        return self.transform(X, y)
    
    def get_class_weights(self):
        """
        Calculate class weights for imbalanced data
        
        Returns dict suitable for sklearn models: {class_label: weight}
        """
        if not self.is_imbalanced:
            return None
        
        class_counts = self.class_distribution
        total = sum(class_counts.values())
        
        # Calculate weights: total / (n_classes * count)
        n_classes = len(class_counts)
        weights = {cls: total / (n_classes * count) for cls, count in class_counts.items()}
        
        return weights
    
    def get_report(self):
        """Get detailed imbalance analysis report"""
        if self.class_distribution is None:
            return {
                'is_imbalanced': False,
                'message': 'No analysis performed yet'
            }
        
        return {
            'is_imbalanced': self.is_imbalanced,
            'imbalance_ratio': round(self.imbalance_ratio, 3),
            'class_distribution': self.class_distribution,
            'recommended_strategy': self.recommended_strategy,
            'reasoning': self.reasoning,
            'action_taken': self.action_taken,
            'class_weights': self.get_class_weights() if self.is_imbalanced else None
        }
    
    def get_model_params(self):
        """
        Get recommended model parameters for handling imbalance
        
        Returns dict of parameters to pass to model
        """
        if not self.is_imbalanced or self.recommended_strategy not in ['class_weight']:
            return {}
        
        class_weights = self.get_class_weights()
        
        # Return format suitable for sklearn models
        return {
            'class_weight': class_weights,
            'note': f'Use these class weights in your model. Reasoning: {self.reasoning}'
        }
