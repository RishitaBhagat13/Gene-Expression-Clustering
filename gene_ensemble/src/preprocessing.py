import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

print("preprocessing module loaded")   


def load_data(path):
    df = pd.read_csv(path)

    label_row = df.iloc[1, 2:].values

    y = []
    for val in label_row:
        val = str(val).strip().upper()
        if "ALL" in val:
            y.append(0)
        elif "AML" in val:
            y.append(1)
        else:
            y.append(0)

    y = np.array(y)

    data = df.iloc[2:, 2:]

    data = data.apply(pd.to_numeric, errors='coerce')

    X = data.T.values

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Label distribution:", np.unique(y, return_counts=True))

    return X, y

def preprocess(X):
    X = np.nan_to_num(X)

    # Normalize per sample
    X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-6)

    # Feature selection
    variances = X.var(axis=0)
    top_idx = np.argsort(variances)[-100:]
    X = X[:, top_idx]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled
