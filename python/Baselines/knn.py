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

def knn_inf(model,x_input, mask):
    print("KNN prediction start")
    full_x = x_input.numpy().astype('float32')
    full_mask = mask.numpy()
    full_x[full_mask == 0] = np.nan
    predictions = model.transform(full_x)
    return predictions

def knn_train_inf(x,masks):
    knn_model = KNNImputer(n_neighbors=7)
    x_train = x.copy().astype(float)
    x_train[masks == 0] = np.nan 
    predictions = knn_model.fit_transform(x_train)
    return predictions