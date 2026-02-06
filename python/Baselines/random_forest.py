import pandas as pd
import numpy as np
import torch
from missforest import MissForest
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import joblib
import pandas as pd

def rf_train(x_full, mask_full, model_path):
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
    joblib.dump(rf_model, model_path)
    print("RF training finished")

def rf_inf(x_input, model):
    x_test = x_input.astype(np.float64)
    x_df = pd.DataFrame(x_test)
    imputed_values = model.transform(x_df)
    imputed_values = np.round(imputed_values).astype(int)
    return imputed_values

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