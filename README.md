# Motor Fault Detection and Classification

This notebook implements a two-stage fault detection and classification system for motor vibration data using LightGBM models.

## Data Preprocessing

### Feature Extraction
- Uses wavelet decomposition (bior3.1 wavelet, level 4) to extract time-frequency domain features
- Features extracted from approximation and detail coefficients:
  - Statistical features: mean, std dev, variance, RMS
  - Shape features: kurtosis, skewness 
  - Zero crossing rate and mean crossing rate
  - Signal entropy
  - Hilbert transform features
- Total 273 features extracted from multiple vibration sensor signals

### Data Preparation
- Uses MAFAULDA dataset containing vibration measurements 
  - Dataset publically available at: https://www02.smt.ufrj.br/~offshore/mfs/page_01.html
  - Contains machinery fault simulator data with multiple fault types
  - Includes accelerometer measurements from different sensor positions
- 10 fault classes labeled A-J:
  - Normal (A)
  - Imbalance (B)  
  - Horizontal misalignment (C)
  - Vertical misalignment (D)
  - Overhang/underhang bearing faults (E-J)
- Train-test split: 80-20
- SMOTE applied to handle class imbalance

## Model Architecture

### Stage 1: Binary Classification
- LightGBM binary classifier to detect fault vs normal operation
- Parameters:
  - learning_rate: 0.05
  - num_leaves: 31
  - n_estimators: 100
  
### Stage 2: Multi-class Classification  
- LightGBM multi-class classifier to identify specific fault type
- 9 fault classes (excludes normal operation)
- Same hyperparameters as binary model
- Uses one-vs-rest approach

## Results

### Binary Classification Performance
- Accuracy: 100% 
- Perfect separation between normal and faulty operation
- Confusion matrix shows zero misclassifications

### Multi-class Classification Performance  
- Accuracy: 99.48%
- Excellent discrimination between different fault types
- Confusion matrix shows minimal misclassification between fault classes

## Usage

The trained models can be used to:
1. Detect presence of fault (binary classification)
2. If fault detected, identify specific fault type (multi-class)

Example usage:
```python
# Load saved models
binary_model = joblib.load("lgbm_binary_model.joblib")
multi_model = joblib.load("lgbm_multi_model.joblib") 

# Make predictions
binary_pred = binary_model.predict(X_test)
if binary_pred == 1:
    fault_type = multi_model.predict(X_test)
```

The high accuracy of both models demonstrates the effectiveness of the wavelet-based feature extraction and two-stage classification approach for motor fault diagnosis.

Link to Kaggle Notebook: https://www.kaggle.com/code/ayushraj911/vibrational-analysis-of-motor
