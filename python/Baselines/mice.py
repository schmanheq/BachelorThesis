from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import numpy as np
import joblib

def mice_train(x,masks):
    mice_model = IterativeImputer(max_iter=10, random_state=0)
    x_train = x.copy().astype(float)
    x_train[masks == 0] = np.nan 
    mice_model.fit(x_train)
    joblib.dump(mice_model, 'baseline_mice_v1.pkl')
    print("mice Training finishes")

def mice_inf(x_input,mask, model_path):
    x_test = x_input.copy().astype(float)
    x_test[mask == 0]=np.nan
    loaded_mice = joblib.load(model_path)
    predictions = loaded_mice.transform(x_test)
    return predictions


def mice(x_input):
    x_input = x_input.detach().cpu().numpy().astype(np.float64)
    x_input[x_input == 0] = np.nan
    imputer = IterativeImputer(max_iter=10, random_state=0)
    x_reconstructed = imputer.fit_transform(x_input)
    return x_reconstructed

