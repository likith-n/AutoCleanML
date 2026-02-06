
# Changelog

All notable changes to AutoCleanML will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-02-07

### Added
- Initial release of AutoCleanML
- Smart missing value imputation (KNN, median, mean, mode)
- Advanced outlier detection (IQR, Z-score, Isolation Forest)
- Automatic data type correction
- Feature extraction (text, datetime, categorical, numeric)
- Model-aware scaling (tree, linear, neural network)
- Feature selection based on correlation
- Class imbalance detection and handling
- Comprehensive reporting system
- No data leakage (proper train/test splitting)
- One-line API for data cleaning
- Production deployment support (pickle)
- Streamlit integration examples

### Fixed
- Feature mismatch bug between scaler and feature selection
- PowerTransformer per-column fitting issue
- Eliminated data leakage by enforcing train/test split before fitting all preprocessing steps
- Corrected automatic data type detection and conversion for numeric, categorical, datetime, and text features
- Fixed feature selection logic to ensure correlation-based selection is applied only on training data
- Improved target encoding by introducing algorithm-aware encoding strategies
- Automatic removal of Unnamed:0 columns (1% accuracy boost)

### Features
- **Context-aware imputation:** Median for skewed, KNN for correlated, mean for normal
- **Smart outlier handling:** Multiple detection methods with configurable actions
- **Model-aware preprocessing:** Auto-detects 15+ model types
- **Feature engineering:** Creates 50+ features automatically
- **Imbalance handling:** Detects and recommends strategies for classification
- **Zero data leakage:** Split first, fit on train, transform test

### Documentation
- Comprehensive README with examples
- API reference documentation
- Deployment guide
- Streamlit integration guide
- Production deployment examples
- Comparison notebooks (manual vs AutoCleanML)

### Examples
- Basic regression example
- Basic classification example
- Streamlit web app template
- Production deployment pipeline
- Jupyter notebook comparisons

## [Unreleased]

### Planned
- Additional feature engineering methods
- Support for time series data
- Enhanced visualization capabilities
- Integration with popular AutoML frameworks


---

## Version History

### [0.1.0] - 2026-02-07
Initial release with core features and documentation.