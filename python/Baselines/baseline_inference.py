import torch
from torch_geometric.loader import DataLoader
from ..Datageneration.dataloader import MyGraphDataset
from ..Evaluation.evaluation_metrics import basic_evaluation_metric, custom_evaluation_metric, custom_evaluation_metric_strict
from ..Baselines.random_forest import rf_inf
from ..Baselines.knn import knn_inf
from ..Baselines.mice import mice_inf
from ..Baselines.majority_class_predictor import fast_majority_class_predictor
from tqdm import tqdm
import joblib

def inference_baselines(PROCESSED_GRAPH_PATH, randomforest_path, knn_path, mice_path, majority_predictor, gpu):
    inference_data = MyGraphDataset(root_dir=PROCESSED_GRAPH_PATH)
    dataloader = DataLoader(inference_data, batch_size=1)
    loop = tqdm(dataloader, total = len(dataloader))
    total_f1_rf = torch.tensor([0.0,0.0,0.0])
    total_f1_knn = torch.tensor([0.0,0.0,0.0])
    total_f1_mice = torch.tensor([0.0,0.0,0.0])
    total_f1_maj = torch.tensor([0.0,0.0,0.0])
    all_training_samples = {
        'x_input':[],
        'trainmask':[]
    }
    ##### Random Forest #####
    if randomforest_path:
        rf_model = joblib.load(randomforest_path)
        if gpu:
            device = torch_directml.device()
            rf_model = convert(rf_model, "torch", device=device)
    else:
        rf_model = None

    ##### kNN #####
    if knn_path:
        knn_model = joblib.load(knn_path)
        if gpu:
            device = torch_directml.device()
            rf_model = convert(knn_model, "torch", device=device)
    else:
        knn_model = None
    
    ##### MICE #####
    if mice_path:
        mice_model = joblib.load(mice_path)
        if gpu:
            device = torch_directml.device()
            rf_model = convert(mice_model, "torch", device=device)
    else:
        mice_model = None

    ##### Inference LOOP #####
    for batch in loop:
        x_input = batch.x*batch.train_mask
        all_training_samples['x_input'].append(x_input)
        all_training_samples['trainmask'].append(batch.train_mask)
    if rf_model:
        x_reconstructed_rf = rf_inf(rf_model, all_training_samples['x_input'], all_training_samples['mask'], gpu=gpu)
        #recall, precision,f1 = basic_evaluation_metric(x_reconstructed_rf, all_training_samples['x_input'], all_training_samples['trainmask'])
        #custom_metric = custom_evaluation_metric(x_reconstructed_rf)
        #custom_metric_strict = custom_evaluation_metric_strict(x_reconstructed_rf, (batch.x+1))
        
    if knn_model:
        x_reconstructed_knn = knn_inf(knn_model, all_training_samples['x_input'], all_training_samples['mask'], gpu=gpu)
        #recall, precision,f1 = basic_evaluation_metric(x_reconstructed_knn, x_input, batch.train_mask)
        #custom_metric = custom_evaluation_metric(x_reconstructed_knn)
        #custom_metric_strict = custom_evaluation_metric_strict(x_reconstructed_knn, (batch.x+1))
        total_f1_knn+=f1
    if mice_model:
        ###################################### This has to be fixed, wrong input ######################################
        x_reconstructed_mice = mice_inf(x_input)
        recall, precision,f1 = basic_evaluation_metric(x_reconstructed_mice, x_input, batch.train_mask)
        #custom_metric = custom_evaluation_metric(x_reconstructed_knn)
        #custom_metric_strict = custom_evaluation_metric_strict(x_reconstructed_knn, (batch.x+1))
        total_f1_mice+=f1
    if majority_predictor:
        x_reconstructed_maj = fast_majority_class_predictor(x_input)
        recall, precision,f1 = basic_evaluation_metric(x_reconstructed_maj, x_input, batch.train_mask)
        total_f1_maj+=f1
    num_samples = len(all_training_samples['x_input'])
    print(total_f1_rf/num_samples)
    #return total_f1_rf/i,total_f1_knn/i,total_f1_mice/i,total_f1_maj/i
        


        