import torch
from torch_geometric.loader import DataLoader
from ..Datageneration.dataloader import MyGraphDataset
from ..Evaluation.evaluation_metrics import basic_evaluation_metric, custom_evaluation_metric, custom_evaluation_metric_strict
from ..Baselines.majority_class_predictor import fast_majority_class_predictor
from ..Baselines.simpleImputer import simpleImpute
from ..Baselines.svdImputer import svd_inf
import numpy as np

def inference_baselines(PROCESSED_GRAPH_PATH, simple_imputer, majority_predictor, svd, svd_path):
    batch_size = 2
    inference_data = MyGraphDataset(root_dir=PROCESSED_GRAPH_PATH)
    dataloader = DataLoader(inference_data, batch_size=batch_size)
    total_f1 = torch.tensor([0.0,0.0,0.0])
    counter = 0
    ##### Inference LOOP ##### 
    for batch in dataloader:
        x_org = batch.x.detach().cpu().numpy().astype(np.float32)
        x = batch.x.detach().cpu().numpy().astype(np.float32)
        mask = batch.train_mask.detach().cpu().numpy().astype(bool)
        x[~mask] = np.nan
        x_all = x.reshape((batch_size,10000,90))
        if simple_imputer:
            results = simpleImpute(x_all, batch_size)
            _,_,f1 = basic_evaluation_metric(results, x_org, mask)
            total_f1+=f1
        
        if svd:
            results = svd_inf(x_all, svd_path)
            _,_,f1 = basic_evaluation_metric(results, x_org, mask)
            total_f1+=f1


        if majority_predictor:
            res = fast_majority_class_predictor(x_all)
            _,_,f1 = basic_evaluation_metric(res, x, batch.train_mask)
            total_f1+=f1
        counter+=1
    print(total_f1/counter)
    return
        


        