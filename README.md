# AutoCleanML

**Stop wasting hours cleaning data. Let AutoCleanML do it for you.**

AutoCleanML automatically cleans and prepares your messy data for machine learning. Just give it your data and target column - it handles the rest.

---

## Why Use AutoCleanML?

**Before AutoCleanML:**
- Spend hours handling missing values
- Manually encode categorical variables  
- Figure out which scaling to use
- Deal with imbalanced datasets
- Wonder if you're doing it right

**With AutoCleanML:**
```python
from autocleanml import AutoCleanML

cleaner = AutoCleanML(target="Price")
X_train, X_test, y_train, y_test, report = cleaner.fit_transform(df)

# Done! Your data is ready for any model
```

---

## What It Does

1. **Fixes data types** - Converts strings to numbers, handles dates
2. **Handles missing values** - Smartly imputes using KNN, median, or mode
3. **Removes outliers** - Detects and handles outliers intelligently
4. **Transforms skewed features** - Applies log/power transforms for highly skewed data
5. **Engineers features** - Creates useful features from text, dates, numbers
6. **Encodes categories** - Handles categorical variables without exploding features
7. **Scales features** - Chooses right scaling based on your model type
8. **Handles imbalance** - Detects and suggests fixes for imbalanced classes
9. **Removes useless features** - Gets rid of constants and highly correlated features

**And it tells you WHY it made each decision.**

---

## Installation

```bash
pip install -e .
```

---

## Quick Start

### Example 1: Predicting House Prices (Regression)

```python
import pandas as pd
from autocleanml import AutoCleanML
from sklearn.ensemble import RandomForestRegressor

# Load your messy data
df = pd.read_csv("house_prices.csv")

# Method 1: Pass your model (AutoCleanML auto-detects optimal preprocessing)
model = RandomForestRegressor()
cleaner = AutoCleanML(target="price", model=model)
X_train, X_test, y_train, y_test, report = cleaner.fit_transform(df)
#  Auto-detected: tree → skips scaling (trees don't need it!)

# Train
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Method 2: Or just specify model type
cleaner = AutoCleanML(target="price", model_type='tree')
X_train, X_test, y_train, y_test, report = cleaner.fit_transform(df)
```
```

### Example 2: Customer Churn (Classification)

```python
# For classification, AutoCleanML detects imbalanced classes
cleaner = AutoCleanML(target="churned")
X_train, X_test, y_train, y_test, report = cleaner.fit_transform(df)

# It tells you if your classes are imbalanced and what to do
print(report['imbalance'])
# Shows: Class weights to use, recommended strategy, reasoning

# Train with recommended class weights
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(class_weight=report['imbalance']['class_weights'])
model.fit(X_train, y_train)
```

---

## Key Features

### Smart Scaling Based on Model Type

**AutoCleanML has TWO ways to be model-aware:**

#### **Method 1: Pass Your Model (Automatic Detection)** ⭐ RECOMMENDED

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

# AutoCleanML detects the model type automatically!
model = RandomForestRegressor()
cleaner = AutoCleanML(target="price", model=model)
#  Auto-detected: tree → No scaling needed

model = LinearRegression()
cleaner = AutoCleanML(target="price", model=model)
#  Auto-detected: linear → StandardScaler + log transforms

model = MLPRegressor()
cleaner = AutoCleanML(target="price", model=model)
#  Auto-detected: nn → MinMaxScaler [0,1]
```

**Supported models auto-detection:**
- 🌳 **Tree-based:** RandomForest, XGBoost, LightGBM, CatBoost, DecisionTree
- 📊 **Linear:** LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet, SGD
- 🧠 **Neural Network:** MLPClassifier, MLPRegressor, Keras, PyTorch models
- 📍 **Distance-based:** KNN, SVM

#### **Method 2: Specify Model Type Manually**

```python
# If you don't have the model object yet
cleaner = AutoCleanML(target="price", model_type='linear')
cleaner = AutoCleanML(target="price", model_type='tree')
cleaner = AutoCleanML(target="price", model_type='nn')
cleaner = AutoCleanML(target="price", model_type='auto')  # Let it guess
```

**Automatic Transformations:**
- **Highly skewed features (skewness > 1)** → Log transform or Yeo-Johnson power transform
- **Features with outliers** → RobustScaler (uses median, less sensitive)
- **Normal distribution** → StandardScaler (zero mean, unit variance)
- **Neural networks** → MinMaxScaler (0-1 bounded for activation functions)

Example output:
```
Transformed 3 skewed features:
  - income: log transform (skewness was 2.34)
  - sales: yeo-johnson (skewness was -1.89)
  - amount: log transform (skewness was 3.12)
  
Scaling: StandardScaler
Reason: Linear model with clean data after transformation
```

### Imbalanced Dataset Handling

For classification, it automatically:
- Detects class imbalance
- Recommends best strategy (class weights, SMOTE, etc.)
- Provides ready-to-use class weights
- Explains why it recommends that strategy

```python
cleaner = AutoCleanML(target="fraud")  # Highly imbalanced dataset
X_train, X_test, y_train, y_test, report = cleaner.fit_transform(df)

# Check imbalance report
if report['imbalance']['is_imbalanced']:
    print(f"Dataset is imbalanced!")
    print(f"Ratio: {report['imbalance']['imbalance_ratio']}")
    print(f"Recommended: {report['imbalance']['recommended_strategy']}")
    print(f"Reason: {report['imbalance']['reasoning']}")
    
    # Use recommended class weights
    model = RandomForestClassifier(
        class_weight=report['imbalance']['class_weights']
    )
```

### Detailed Reporting

Every decision is explained:

```python
# After cleaning
print(report['summary'])  # Overall summary
print(report['missing_values'])  # How missing values were handled
print(report['outliers'])  # Outlier detection details
print(report['scaling'])  # Why this scaling was chosen
print(report['imbalance'])  # Imbalance analysis (classification)
print(report['feature_engineering'])  # Features created
```

Example report:
```
Scaling: RobustScaler
Reason: Data has outliers (>3 columns), using RobustScaler (less sensitive to outliers)

Imbalance: SEVERE (ratio=0.12)
Recommended: class_weight
Reason: Severe imbalance (ratio=0.12) with large dataset, 
        using class_weight (efficient for tree-based models)
Class weights: {0: 1.0, 1: 7.33}
```

---

## Configuration Options

```python
cleaner = AutoCleanML(
    target="price",              # Required: your target column
    
    # Train/test split
    test_size=0.2,              # 80-20 split
    random_state=42,            # For reproducibility
    
    # Outlier handling
    outlier_method='auto',      # 'iqr', 'zscore', 'isolation_forest'
    outlier_action='cap',       # 'cap', 'remove', 'flag'
    
    # Feature engineering
    feature_extraction=True,    # Create new features
    max_features=100,          # Limit feature count
    
    # Model optimization
    model_type='auto',         # 'linear', 'tree', 'nn', 'auto'
    
    # Verbosity
    verbose=True               # Show progress
)
```

---

## What Makes AutoCleanML Smart?

### 1. Context-Aware Missing Value Imputation

Not all missing values should be filled the same way:
- **Skewed data?** → Uses median (not affected by outliers)
- **Correlated features?** → Uses KNN (preserves relationships)
- **Normal distribution?** → Uses mean
- **Categories?** → Uses most frequent value

### 2. Intelligent Scaling

Chooses scaling based on:
- Your model type (tree models don't need scaling!)
- Your data characteristics (outliers? → RobustScaler)
- Task requirements (neural nets → MinMaxScaler)

### 3. Imbalance Awareness

For classification:
- Detects severity of imbalance
- Considers dataset size
- Recommends appropriate strategy
- Provides ready-to-use class weights

### 4. No Data Leakage

Always:
1. Splits data FIRST
2. Fits transformations on training data ONLY
3. Applies learned transformations to test data

You'll never accidentally leak information from test to train.

### 5. Guaranteed Clean Output

- **Zero NaN values** - Triple-layer protection ensures no missing values
- **All features encoded** - Everything is numeric and ready for models
- **Proper scaling** - Features scaled appropriately for your model type

---

## Common Use Cases

### Use Case 1: Quick Model Baseline

```python
# Get a clean baseline fast
cleaner = AutoCleanML(target="target")
X_train, X_test, y_train, y_test, _ = cleaner.fit_transform(df)

# Try multiple models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

for Model in [LinearRegression, RandomForestRegressor]:
    model = Model()
    model.fit(X_train, y_train)
    print(f"{Model.__name__}: {model.score(X_test, y_test):.3f}")
```

### Use Case 2: Production Pipeline

```python
# Save the cleaner for production
import pickle

cleaner = AutoCleanML(target="price")
X_train, X_test, y_train, y_test, report = cleaner.fit_transform(train_df)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save both
pickle.dump(cleaner, open('cleaner.pkl', 'wb'))
pickle.dump(model, open('model.pkl', 'wb'))

# In production
cleaner = pickle.load(open('cleaner.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Clean new data the same way
new_data_clean = cleaner.transform(new_data)
predictions = model.predict(new_data_clean)
```

### Use Case 3: Kaggle Competitions

```python
# Quick clean for competitions
cleaner = AutoCleanML(
    target="target",
    feature_extraction=True,    # Create extra features
    max_features=200,          # Keep more features
    model_type='tree'          # No scaling for XGBoost
)
X_train, X_test, y_train, y_test, report = cleaner.fit_transform(train_df)

# Check what was done
print(f"Created {report['feature_engineering']['total_features_created']} new features")
print(f"Final feature count: {X_train.shape[1]}")
```

---

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- scipy

Install dependencies:
```bash
pip install pandas numpy scikit-learn scipy
```

---

## Tips

### Tip 1: Check the Report

Always look at the report to understand what was done:
```python
cleaner = AutoCleanML(target="price")
X_train, X_test, y_train, y_test, report = cleaner.fit_transform(df)

# See what happened
print(report['scaling'])  # Why this scaling?
print(report['imbalance'])  # Is data imbalanced?
```

### Tip 2: Match Model Type

Tell AutoCleanML what model you'll use:
```python
# For tree-based models (no scaling needed)
cleaner = AutoCleanML(target="price", model_type='tree')

# For linear models (needs scaling)
cleaner = AutoCleanML(target="price", model_type='linear')
```

### Tip 3: Handle Imbalanced Data

For classification with imbalanced classes:
```python
cleaner = AutoCleanML(target="fraud")
X_train, X_test, y_train, y_test, report = cleaner.fit_transform(df)

# Use recommended class weights
if report['imbalance']['is_imbalanced']:
    model = RandomForestClassifier(
        class_weight=report['imbalance']['class_weights']
    )
    model.fit(X_train, y_train)
```

---

## Troubleshooting

**Q: Getting import errors?**
```bash
cd AutoCleanML
python -m pip uninstall -y autocleanml
python -m pip install -e .
```

**Q: Model performance seems off?**
- Check `model_type` matches your model
- Review scaling report: `print(report['scaling'])`
- For classification, check imbalance report

**Q: Want more features?**
```python
cleaner = AutoCleanML(target="price", max_features=200)
```

**Q: Want less processing?**
```python
cleaner = AutoCleanML(
    target="price",
    feature_extraction=False,  # Skip feature engineering
    model_type='tree'          # Skip scaling
)
```

---

## What's Next?

After cleaning with AutoCleanML:

1. **Train models** - Your data is ready for any sklearn model
2. **Tune hyperparameters** - Use GridSearchCV or RandomizedSearchCV
3. **Deploy** - Save the cleaner with your model for production

---

## License

MIT License - Use it however you want!

---

## Summary

**AutoCleanML makes ML data preprocessing automatic and intelligent.**

✅ One line to clean data  
✅ Smart decisions based on data characteristics  
✅ Model-aware preprocessing  
✅ Handles imbalanced datasets  
✅ Explains every decision  
✅ Guaranteed clean output  
✅ No data leakage  

**Stop cleaning data manually. Start using AutoCleanML.**

```python
from autocleanml import AutoCleanML

cleaner = AutoCleanML(target="your_target")
X_train, X_test, y_train, y_test, report = cleaner.fit_transform(df)

# Done! Train your model now.
```
