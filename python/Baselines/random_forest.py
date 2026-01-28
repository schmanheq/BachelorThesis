import pandas as pd
import numpy as np
import torch
from missforest import MissForest
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import joblib


def rf_train(x_full, mask_full):
    fast_rf = RandomForestRegressor(
        n_estimators=7, 
        max_depth=3,
        max_features='sqrt',     
        n_jobs=-1, 
        random_state=0
    )

    rf_model = MissForest(rgr=fast_rf)
    x_train = x_full.copy().astype(float)
    x_train[mask_full == 0] = np.nan 
    rf_model.fit(x_train)
    joblib.dump(rf_model, 'baseline_rf_v1.pkl')
    print("RF training finished")

def rf_inference(x_input,mask, model_path):
    x_test = x_input.copy().astype(float)
    x_test[mask == 0]=np.nan
    loaded_rf = joblib.load(model_path)
    predictions = loaded_rf.predict(x_test)
    return predictions

def randomforest_impute(x_input):
    x_np = x_input.detach().cpu().numpy().astype(np.float64)
    x_np[x_np == 0] = np.nan
    df_input = pd.DataFrame(x_np)
    fast_rf = RandomForestRegressor(
        n_estimators=7, 
        max_depth=3,
        max_features='sqrt',     
        n_jobs=-1, 
        random_state=0
    )
    mf = MissForest(clf=fast_rf, max_iter=2)
    x_filled = mf.fit_transform(df_input)
    if isinstance(x_filled, pd.DataFrame):
        x_filled = x_filled.values
    return torch.from_numpy(x_filled).round().clamp(1, 3).to(torch.uint8)