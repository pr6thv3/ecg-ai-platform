import os
import argparse
from collections import Counter
import numpy as np
import wfdb
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

# AAMI Standard Mapping for MIT-BIH Arrhythmia Database
# Normal (N), Premature Ventricular Contraction (V), Atrial Premature Beat (A),
# Left bundle branch block (L), Right bundle branch block (R)
AAMI_MAPPING = {
    'N': 0, 'L': 3, 'R': 4, 'V': 1, 'A': 2,
    # Additional mappings can be added here (e.g., mapping 'a' to 'A')
}

def load_data(data_path, window_size=360, seed=42):
    """
    Loads MIT-BIH records using wfdb, extracts windowed beats around R-peaks,
    maps labels to AAMI standard, and returns balanced stratified splits.
    """
    # For a real run, this would read all records. E.g., records = [str(i) for i in range(100, 240) if os.path.exists(...)]
    records = ['100', '101', '103', '105', '111'] 
    
    X, y = [], []
    print("Loading MIT-BIH records...")
    
    for record in records:
        record_path = os.path.join(data_path, record)
        if not os.path.exists(record_path + '.dat'):
            continue
            
        try:
            # Read signal and annotations
            sig, fields = wfdb.rdsamp(record_path)
            annotation = wfdb.rdann(record_path, 'atr')
            
            # Typically Channel 0 is MLII which provides the clearest R-peaks
            signal = sig[:, 0]
            
            # Segment the beats
            for symbol, sample in zip(annotation.symbol, annotation.sample):
                if symbol in AAMI_MAPPING:
                    left = sample - window_size // 2
                    right = sample + window_size // 2
                    
                    # Ensure window doesn't go out of bounds
                    if left >= 0 and right < len(signal):
                        X.append(signal[left:right])
                        y.append(AAMI_MAPPING[symbol])
        except Exception as e:
            print(f"Error processing record {record}: {e}")
            
    X = np.array(X)
    y = np.array(y)
    
    if len(X) == 0:
        raise ValueError(f"No records found in {data_path}. Please download the MIT-BIH database.")

    print(f"\nOriginal class distribution: {Counter(y)}")
    
    # 70/15/15 Split via two-stage stratify
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=seed)
    
    # Split the remaining 85% to get 70% train / 15% val (15 / 85 = 0.1764)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1764, stratify=y_temp, random_state=seed)
    
    # Handle Class Imbalance ONLY on the training set to prevent data leakage!
    # Using RandomOversampler for 1D signals to preserve real physiological structures without SMOTE interpolation artifacts
    ros = RandomOverSampler(random_state=seed)
    X_train_res, y_train_res = ros.fit_resample(X_train, y_train)
    
    print("\n--- Distribution After Split & Balancing ---")
    print(f"Train (Balanced): {Counter(y_train_res)}")
    print(f"Validation      : {Counter(y_val)}")
    print(f"Test            : {Counter(y_test)}")
    
    return (X_train_res, y_train_res), (X_val, y_val), (X_test, y_test)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MIT-BIH Data Loader & Balancer")
    parser.add_argument('--data_path', type=str, default='../datasets/mit-bih', help='Path to WFDB records')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Quick sanity check run
    # load_data(args.data_path, seed=args.seed)
