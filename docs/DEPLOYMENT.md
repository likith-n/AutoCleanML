# Deployment Guide

## Installation
```bash
pip install autocleanml
```

## Usage
```python
from autocleanml import AutoCleanML

cleaner = AutoCleanML(target="price")
X_train, X_test, y_train, y_test, report = cleaner.fit_transform("data.csv")
```

## Streamlit Deployment
See examples/streamlit_app.py
