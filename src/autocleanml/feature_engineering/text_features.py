"""
Text Feature Extraction
=======================

Extracts meaningful features from text columns using regex and NLP techniques.
"""

import pandas as pd
import numpy as np
import re
from typing import List, Dict, Any


class TextFeatureExtractor:
    """
    Extract features from text columns.
    
    Features Extracted:
    - Length statistics (char count, word count)
    - Numeric extraction (years, numbers, prices)
    - Keyword presence (customizable)
    - Sentiment indicators (basic)
    - Pattern matching (emails, URLs, phone numbers)
    
    Examples
    --------
    >>> extractor = TextFeatureExtractor()
    >>> new_df = extractor.fit_transform(df, text_columns=['description', 'title'])
    """
    
    def __init__(self, extract_keywords=True, extract_numbers=True, 
                 extract_length=True, max_keywords=50):
        """
        Parameters
        ----------
        extract_keywords : bool
            Extract common keyword presence
        extract_numbers : bool
            Extract numeric values from text
        extract_length : bool
            Extract length-based features
        max_keywords : int
            Maximum number of keywords to extract per column
        """
        self.extract_keywords = extract_keywords
        self.extract_numbers = extract_numbers
        self.extract_length = extract_length
        self.max_keywords = max_keywords
        
        self.text_columns_ = []
        self.common_keywords_ = {}
        self.extraction_log_ = []
    
    def _extract_years(self, text):
        """Extract years from text (1900-2099)"""
        if pd.isna(text):
            return None
        matches = re.findall(r'\b(19\d{2}|20\d{2})\b', str(text))
        return int(matches[0]) if matches else None
    
    def _extract_numbers(self, text):
        """Extract first number from text"""
        if pd.isna(text):
            return None
        # Match numbers with optional commas and decimals
        matches = re.findall(r'\d+(?:,\d{3})*(?:\.\d+)?', str(text))
        if matches:
            # Remove commas and convert to float
            return float(matches[0].replace(',', ''))
        return None
    
    def _extract_experience_years(self, text):
        """Extract experience in years (e.g., '5 years', '5+ years')"""
        if pd.isna(text):
            return None
        patterns = [
            r'(\d+)\+?\s*(?:years?|yrs?)',
            r'(\d+)\+?\s*(?:year|yr)',
        ]
        for pattern in patterns:
            matches = re.findall(pattern, str(text).lower())
            if matches:
                return int(matches[0])
        return None
    
    def _has_pattern(self, text, pattern):
        """Check if text contains pattern"""
        if pd.isna(text):
            return 0
        return 1 if re.search(pattern, str(text), re.IGNORECASE) else 0
    
    def _get_common_keywords(self, series, top_n=20):
        """Extract most common words from a text series"""
        all_words = []
        for text in series.dropna():
            # Simple word extraction (lowercase, alphanumeric)
            words = re.findall(r'\b[a-z]{3,}\b', str(text).lower())
            all_words.extend(words)
        
        # Count frequency
        from collections import Counter
        word_counts = Counter(all_words)
        
        # Remove common stopwords
        stopwords = {'the', 'and', 'for', 'with', 'this', 'that', 'from', 
                    'have', 'has', 'are', 'was', 'were', 'been', 'will'}
        
        keywords = [(word, count) for word, count in word_counts.most_common(top_n * 2)
                   if word not in stopwords][:top_n]
        
        return [word for word, _ in keywords]
    
    def fit(self, df, text_columns=None):
        """
        Learn text patterns from data.
        
        Parameters
        ----------
        df : pd.DataFrame
            Training data
        text_columns : list, optional
            List of text column names. If None, auto-detect object columns.
        """
        if text_columns is None:
            # Auto-detect text columns (object dtype with long strings)
            text_columns = []
            for col in df.select_dtypes(include=['object']).columns:
                avg_len = df[col].astype(str).str.len().mean()
                if avg_len > 10:  # Likely text, not category
                    text_columns.append(col)
        
        self.text_columns_ = text_columns
        
        # Extract common keywords for each text column
        if self.extract_keywords:
            for col in self.text_columns_:
                self.common_keywords_[col] = self._get_common_keywords(
                    df[col], top_n=min(10, self.max_keywords)
                )
        
        return self
    
    def transform(self, df):
        """
        Extract features from text columns.
        
        Parameters
        ----------
        df : pd.DataFrame
            Data to transform
            
        Returns
        -------
        pd.DataFrame
            DataFrame with new text-based features
        """
        df = df.copy()
        
        for col in self.text_columns_:
            if col not in df.columns:
                continue
            
            created_features = []
            
            # Length features
            if self.extract_length:
                df[f'{col}_char_count'] = df[col].astype(str).str.len()
                df[f'{col}_word_count'] = df[col].astype(str).str.split().str.len()
                created_features.extend([f'{col}_char_count', f'{col}_word_count'])
            
            # Extract numbers
            if self.extract_numbers:
                # Try to extract years
                years = df[col].apply(self._extract_years)
                if years.notna().sum() > len(df) * 0.1:  # If found in >10% of rows
                    df[f'{col}_year'] = years
                    created_features.append(f'{col}_year')
                
                # Try to extract experience years
                exp_years = df[col].apply(self._extract_experience_years)
                if exp_years.notna().sum() > len(df) * 0.1:
                    df[f'{col}_experience_years'] = exp_years
                    created_features.append(f'{col}_experience_years')
                
                # General number extraction
                numbers = df[col].apply(self._extract_numbers)
                if numbers.notna().sum() > len(df) * 0.1:
                    df[f'{col}_number'] = numbers
                    created_features.append(f'{col}_number')
            
            # Pattern detection
            patterns = {
                'has_email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                'has_url': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                'has_phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
                'has_senior': r'\b(?:senior|sr\.?|lead)\b',
                'has_junior': r'\b(?:junior|jr\.?)\b',
            }
            
            for feat_name, pattern in patterns.items():
                has_pattern = df[col].apply(lambda x: self._has_pattern(x, pattern))
                if has_pattern.sum() > 0:  # Only add if pattern found
                    df[f'{col}_{feat_name}'] = has_pattern
                    created_features.append(f'{col}_{feat_name}')
            
            # Keyword presence
            if self.extract_keywords and col in self.common_keywords_:
                for keyword in self.common_keywords_[col][:5]:  # Top 5 keywords
                    feat_name = f'{col}_has_{keyword}'
                    df[feat_name] = df[col].astype(str).str.lower().str.contains(keyword, regex=False).astype(int)
                    created_features.append(feat_name)
            
            # Log what was created
            if created_features:
                self.extraction_log_.append({
                    'original_column': col,
                    'features_created': created_features,
                    'count': len(created_features)
                })
        
        return df
    
    def fit_transform(self, df, text_columns=None):
        """Fit and transform in one step"""
        self.fit(df, text_columns)
        return self.transform(df)
    
    def get_extraction_report(self):
        """Get report of features extracted"""
        return {
            'text_columns_processed': self.text_columns_,
            'total_features_created': sum(log['count'] for log in self.extraction_log_),
            'details': self.extraction_log_
        }
