from sklearn.impute import KNNImputer
import joblib
import numpy as np
import torch

def knn_train(x,masks):
    knn_model = KNNImputer(n_neighbors=7)
    x_train = x.copy().astype(float)
    x_train[masks == 0] = np.nan 
    knn_model.fit(x_train)
    joblib.dump(knn_model, 'baseline_knn_v1.pkl')
    print("kNN Training finishes")

def knn_inf(x_input,mask, model_path):
    x_test = x_input.copy().astype(float)
    x_test[mask == 0]=np.nan
    loaded_knn = joblib.load(model_path)
    predictions = loaded_knn.transform(x_test)
    return predictions

def knn_impute(x_input):
    imputer = KNNImputer(n_neighbors=7)
    x_input[x_input == 0] = np.nan
    x_reconstructed = imputer.fit_transform(x_input)
    return torch.tensor(x_reconstructed).round().to(torch.uint8)