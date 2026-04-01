import torch
from torch_geometric.loader import DataLoader
from ..Datageneration.dataloader import MyGraphDataset
from ..Evaluation.evaluation_metrics import basic_evaluation_metric, custom_evaluation_metric, matth_coeff, confusion_matrix
from ..Baselines.majority_class_predictor import fast_majority_class_predictor
from ..Baselines.simpleImputer import simpleImpute
from ..Baselines.svdImputer import svd_inf
import numpy as np

def inference_baselines(PROCESSED_GRAPH_PATH, simple_imputer, majority_predictor, svd, svd_path):
    batch_size = 1
    inference_data = MyGraphDataset(root_dir=PROCESSED_GRAPH_PATH)
    dataloader = DataLoader(inference_data, batch_size=batch_size)
    total_f1 = torch.tensor([0.0,0.0,0.0])
    total_mc = 0
    total_cf_matrix = torch.tensor([[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]])
    custom_metric_tracker = 0
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
            m_coef = matth_coeff(results, x_org, mask)
            cf_matrix = confusion_matrix(results, x_org, mask)
            custom_metric = custom_evaluation_metric(results, batch_size)
            custom_metric_tracker+=custom_metric
            total_f1+=f1
            total_mc+=m_coef
            total_cf_matrix+=cf_matrix
        
        if svd:
            results = svd_inf(x_all, svd_path)
            _,_,f1 = basic_evaluation_metric(results, x_org, mask)
            m_coef = matth_coeff(results, x_org, mask)
            cf_matrix = confusion_matrix(results, x_org, mask)
            custom_metric = custom_evaluation_metric(results, batch_size)
            custom_metric_tracker+=custom_metric
            total_f1+=f1
            total_mc+=m_coef
            total_cf_matrix+=cf_matrix


        if majority_predictor:
            results = fast_majority_class_predictor(x_all)
            _,_,f1 = basic_evaluation_metric(results, x_org, mask)
            m_coef = matth_coeff(results, x_org, mask)
            cf_matrix = confusion_matrix(results, x_org, mask)
            custom_metric = custom_evaluation_metric(results, batch_size)
            custom_metric_tracker+=custom_metric
            total_f1+=f1
            total_mc+=m_coef
            total_cf_matrix+=cf_matrix
        counter+=1
    print(f"Average F1: {total_f1/counter}")
    print(f"Average MC Coefficient: {total_mc/counter}")
    print(f"Average Confusion Matrix: ")
    print(total_cf_matrix/counter)
    print(f"Average Custom Metric: {custom_metric_tracker/counter}")
    return      