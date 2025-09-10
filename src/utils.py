import numpy as np
import pandas as pd
import pywt
from scipy.stats import skew, kurtosis
from scipy.signal import hilbert

def extract_features(signal, wavelet='bior3.1', level=4):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    approx_coeff = coeffs[0]
    detail_coeffs = coeffs[1:]
    
    features = {}
    features['mean_approx'] = np.mean(approx_coeff)
    features['std_approx'] = np.std(approx_coeff)
    features['var_approx'] = np.var(approx_coeff)
    features['rms_approx'] = np.sqrt(np.mean(np.square(approx_coeff)))
    features['kurt_approx'] = kurtosis(approx_coeff)
    features['skew_approx'] = skew(approx_coeff)
    features['zcr_approx'] = np.sum(np.abs(np.diff(np.sign(approx_coeff)))) / len(approx_coeff)
    features['mcr_approx'] = np.sum(np.abs(np.diff(np.sign(approx_coeff - np.mean(approx_coeff))))) / len(approx_coeff)
    features['entropy_approx'] = -np.sum(np.histogram(approx_coeff, bins=10, density=True)[0] * np.log2(np.histogram(approx_coeff, bins=10, density=True)[0] + 1e-6))
    
    for i, d_coeff in enumerate(detail_coeffs, 1):
        features[f'mean_d{i}'] = np.mean(d_coeff)
        features[f'std_d{i}'] = np.std(d_coeff)
        features[f'var_d{i}'] = np.var(d_coeff)
        features[f'rms_d{i}'] = np.sqrt(np.mean(np.square(d_coeff)))
        features[f'kurt_d{i}'] = kurtosis(d_coeff)
        features[f'skew_d{i}'] = skew(d_coeff)
        features[f'entropy_d{i}'] = -np.sum(np.histogram(d_coeff, bins=10, density=True)[0] * np.log2(np.histogram(d_coeff, bins=10, density=True)[0] + 1e-6))
    
    analytic_signal = hilbert(approx_coeff)
    features['hilbert_mean'] = np.mean(np.abs(analytic_signal))
    features['hilbert_std'] = np.std(np.abs(analytic_signal))
    
    return features

def preprocess_csv(file):
    df = pd.read_csv(file, header=None)
    df = df.iloc[:, :-1]
    
    combined_features = {}
    for col in range(df.shape[1]):
        signal = df.iloc[:, col].values
        features = extract_features(signal)
        
        for key, value in features.items():
            combined_features[f'col{col+1}_{key}'] = value
    
    return pd.DataFrame([combined_features])

def predict_faults(df_features, binary_model, multi_model, label_encoder):
    binary_prediction = binary_model.predict(df_features)
    
    if binary_prediction[0] == 0:
        return "Normal"
    else:
        multi_prediction = multi_model.predict(df_features)
        decoded_label = label_encoder.inverse_transform(multi_prediction)
        label_mapping = {
            "A": "Normal",
            "B": "Imbalance",
            "C": "Horizontal Misalignment",
            "D": "Vertical Misalignment",
            "E": "Overhang Ball_Fault",
            "F": "Overhang Cage_Fault",
            "G": "Overhang Outer_Race",
            "H": "Underhang Ball_Fault",
            "I": "Underhang Cage_Fault",
            "J": "Underhang Outer_Race"
        }
        return label_mapping.get(decoded_label[0], "Unknown Fault")