import torch
from torch_geometric.loader import DataLoader
from ..Datageneration.dataloader import MyGraphDataset
from ..Evaluation.evaluation_metrics import basic_evaluation_metric, custom_evaluation_metric, custom_evaluation_metric_strict
from ..Baselines.random_forest import rf_inf
from ..Baselines.knn import knn_inf
from ..Baselines.majority_class_predictor import fast_majority_class_predictor
from ..Baselines.simpleImputer import simpleImpute
from concurrent.futures import ProcessPoolExecutor
import os
from itertools import repeat
from tqdm import tqdm
import joblib
import numpy as np

def inference_baselines(PROCESSED_GRAPH_PATH, randomforest_path, knn_path, majority_predictor):
    inference_data = MyGraphDataset(root_dir=PROCESSED_GRAPH_PATH)
    dataloader = DataLoader(inference_data, batch_size=16)
    loop = tqdm(dataloader, total = len(dataloader))
    num_cores = os.cpu_count()
    total_f1_rf = torch.tensor([0.0,0.0,0.0])
    total_f1_knn = torch.tensor([0.0,0.0,0.0])
    total_f1_maj = torch.tensor([0.0,0.0,0.0])
    x_all =[]
    masks_all = []
    ##### Random Forest #####
    if randomforest_path:
        rf_model = joblib.load(randomforest_path)
        print("RandomForest found")
    else:
        rf_model = None

    ##### kNN #####
    if knn_path:
        knn_model = joblib.load(knn_path)
        print("KNN found")
    else:
        knn_model = None
    
    ##### Inference LOOP ##### 
    for batch in dataloader:
        # Convert to numpy immediately and handle the NaNs here
        x = batch.x.detach().cpu().numpy().astype(np.float32)
        mask = batch.train_mask.detach().cpu().numpy().astype(bool)
        x[~mask] = np.nan
        x_all.append(x)
    print("data loaded")
    if rf_model:
        results = rf_inf(x_all, rf_model)
        #recall, precision,f1 = basic_evaluation_metric(x_reconstructed_rf, all_training_samples['x_input'], all_training_samples['trainmask'])
        #custom_metric = custom_evaluation_metric(x_reconstructed_rf)
        #custom_metric_strict = custom_evaluation_metric_strict(x_reconstructed_rf, (batch.x+1))
        
    if knn_model:
        with ProcessPoolExecutor(max_workers=num_cores//2) as executor:  
            results = list(executor.map(knn_inf, x_all, repeat(knn_model)))
        return results
        #recall, precision,f1 = basic_evaluation_metric(x_reconstructed_knn, x_input, batch.train_mask)
        #custom_metric = custom_evaluation_metric(x_reconstructed_knn)‚
        #custom_metric_strict = custom_evaluation_metric_strict(x_reconstructed_knn, (batch.x+1))
        #total_f1_knn+=f1
    if majority_predictor:
        x_reconstructed_maj = fast_majority_class_predictor(x_input)
        recall, precision,f1 = basic_evaluation_metric(x_reconstructed_maj, x_input, batch.train_mask)
        total_f1_maj+=f1
    return 1
    num_samples = len(all_training_samples['x_input'])
    print(total_f1_rf/num_samples)
    #return total_f1_rf/i,total_f1_knn/i,total_f1_mice/i,total_f1_maj/i
        


        