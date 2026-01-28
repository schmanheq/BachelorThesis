from sklearn.impute import KNNImputer
import joblib
import numpy as np
import torch
import torch_directhtml
from hummingbird.ml import convert

def knn_train(x,masks):
    knn_model = KNNImputer(n_neighbors=7)
    x_train = x.copy().astype(float)
    x_train[masks == 0] = np.nan 
    knn_model.fit(x_train)
    joblib.dump(knn_model, 'baseline_knn_v1.pkl')
    print("kNN Training finishes")

def knn_inf(model,x_input, mask, gpu=False):
    x_test = x_input.copy().astype('float32')
    x_test[mask == 0] = np.nan
    if gpu:        
        x_test_torch = torch.from_numpy(x_test).to(device)
        with torch.no_grad():
            device = torch_directml.device()        
            predictions = model.transform(x_test_torch)
        return predictions.cpu().numpy()
    else:
        return model.transform(x_test)

def knn_impute(x_input):
    imputer = KNNImputer(n_neighbors=7)
    x_input[x_input == 0] = np.nan
    x_reconstructed = imputer.fit_transform(x_input)
    return torch.tensor(x_reconstructed).round().to(torch.uint8)