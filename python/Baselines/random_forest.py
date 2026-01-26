import pandas as pd
import numpy as np
import torch
from missforest import MissForest
from sklearn.ensemble import RandomForestRegressor

def randomforest_impute(x_input):
    x_np = x_input.detach().cpu().numpy().astype(np.float64)
    x_np[x_np == 0] = np.nan
    df_input = pd.DataFrame(x_np)

    fast_rf = RandomForestRegressor(
        n_estimators=7, 
        max_depth=3,     
        n_jobs=-1, 
        random_state=0
    )

    mf = MissForest(clf=fast_rf, max_iter=2)
    x_filled = mf.fit_transform(df_input)
    
    if isinstance(x_filled, pd.DataFrame):
        x_filled = x_filled.values
        
    return torch.from_numpy(x_filled).round().clamp(1, 3).to(torch.uint8)