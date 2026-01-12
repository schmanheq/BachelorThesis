from missforest import MissForest
import pandas as pd
import numpy as np
from python.VGAE.VGAE_training import fetch_data_optimized, create_mask_optimize
from python.Datageneration.datageneration import save_snapshots_fast



def randomforest(adj_path,snapshots_path,fraction_missing_data):
    _, node_features = fetch_data_optimized(adj_path, snapshots_path)
    node_features = np.rot90(node_features, axes=(1, 2)).astype(int)
    masks = [create_mask_optimize(node_feature, fraction_missing_data) for node_feature in node_features]
    masks = np.array(masks)
    node_features = node_features*masks
    node_features = node_features[0]
    categorical = [1,2,3] 
    cat_cols = list(range(node_features.shape[1]))
    mf = MissForest(n_estimators = 100, random_state = 42, categorical=categorical)
    node_features[node_features==0]=np.nan
    X_imputed = mf.fit_transform(node_features, cat_vars = cat_cols)
    save_snapshots_fast(X_imputed, 'test.csv')
    return