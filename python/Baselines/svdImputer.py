import numpy as np
import torch
import joblib
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer



def svd_training(data,path, rank=20):
    svd = TruncatedSVD(n_components=rank, random_state=42)
    imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    data_stacked = np.vstack(data)
    filled_data = imp.fit_transform(data_stacked)
    svd.fit(filled_data)
    joblib.dump(svd, path)
    return svd


def svd_inf(batch_data,path, rank=20):

    batch_size = batch_data.shape[0]
    reconstructed_batch = np.zeros_like(batch_data)
    svd = joblib.load(path)
    imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    
    for i in range(batch_size):
        matrix = batch_data[i] 
        filled_matrix = imp.fit_transform(matrix)
        
        low_dim_repr = svd.transform(filled_matrix)
        reconstructed_batch[i] = svd.inverse_transform(low_dim_repr)
    clamped = np.clip(reconstructed_batch, 1, 3)
    reconstructed_batch = np.round(clamped)
    reshaped_batch = reconstructed_batch.reshape(-1,90)
    return reshaped_batch