from sklearn.impute import KNNImputer
import numpy as np
import torch

def knn_impute(x_input):
    imputer = KNNImputer(n_neighbors=7)
    x_input[x_input == 0] = np.nan
    x_reconstructed = imputer.fit_transform(x_input)
    return torch.tensor(x_reconstructed).round().to(torch.uint8)