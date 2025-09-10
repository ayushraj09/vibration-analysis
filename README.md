# Motor Fault Detection System

A Streamlit web application that implements a two-stage fault detection and classification system for motor vibration data using LightGBM models.

## Project Structure
```
streamlit-fault-prediction/
├── src/
│   ├── app.py          # Streamlit application entry point
│   ├── utils.py        # Utility functions for data processing
│   ├── lgbm_binary_model.joblib    # Binary classifier model
│   ├── lgbm_multi_model.joblib     # Multi-class classifier model
│   └── label_encoder.joblib        # Label encoder for fault classes
├── requirements.txt    # Project dependencies
└── README.md          # Documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/streamlit-fault-prediction.git
cd streamlit-fault-prediction
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Mac/Linux
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit application:
```bash
streamlit run src/app.py
```

2. Upload a CSV file from the MAFAULDA dataset through the web interface
3. View the prediction results

## Technical Details

### Data Preprocessing

#### Feature Extraction
- Uses wavelet decomposition (bior3.1 wavelet, level 4) for time-frequency domain features
- Features from approximation and detail coefficients include:
  - Statistical features: mean, std dev, variance, RMS
  - Shape features: kurtosis, skewness 
  - Zero crossing rate and mean crossing rate
  - Signal entropy
  - Hilbert transform features
- Total 273 features extracted from vibration sensor signals

### Model Architecture

#### Stage 1: Binary Classification
- LightGBM binary classifier (Normal vs Fault)
- Parameters:
  - learning_rate: 0.05
  - num_leaves: 31
  - n_estimators: 100
  
#### Stage 2: Multi-class Classification  
- LightGBM multi-class classifier for specific fault types
- Fault classes:
  - Normal (A)
  - Imbalance (B)  
  - Horizontal misalignment (C)
  - Vertical misalignment (D)
  - Overhang Ball Fault (E)
  - Overhang Cage Fault (F)
  - Overhang Outer Race (G)
  - Underhang Ball Fault (H)
  - Underhang Cage Fault (I)
  - Underhang Outer Race (J)

### Performance
- Binary Classification Accuracy: 100%
- Multi-class Classification Accuracy: 99.48%

## Dataset

Uses MAFAULDA dataset containing vibration measurements:
- Dataset available at: https://www02.smt.ufrj.br/~offshore/mfs/page_01.html
- Contains machinery fault simulator data
- Includes accelerometer measurements from different sensor positions

## License

This project is licensed under the MIT License.

## Links
- Original Kaggle Notebook: https://www.kaggle.com/code/ayushraj911/vibrational-analysis-of-motor