import torch
from python.VGAE.VGAE_model import VariationalGraphAutoEncoder
from torch_geometric.loader import DataLoader
from ..Datageneration.dataloader import MyGraphDataset
from ..Evaluation.prob_to_states import transform_to_states
from ..Evaluation.evaluation_metrics import basic_evaluation_metric, custom_evaluation_metric, custom_evaluation_metric_strict
from ..Baselines.random_forest import randomforest_impute
from ..Baselines.knn import knn_impute
from ..Baselines.mice import mice
from ..Baselines.majority_class_predictor import fast_majority_class_predictor
from tqdm import tqdm
import torch.nn.functional as F


def inference(INPUT_DIM, HIDDEN_DIM, Z_DIM, NUM_HIDDEN_LAYERS, NUM_CLASSES,PROCESSED_GRAPH_PATH, WEIGHTS_PATH):
    vgae = VariationalGraphAutoEncoder(INPUT_DIM,HIDDEN_DIM,Z_DIM,NUM_HIDDEN_LAYERS, NUM_CLASSES)
    state_dict = torch.load(WEIGHTS_PATH, map_location=torch.device('cpu'))

    # 3. Load the weights into the model
    vgae.load_state_dict(state_dict['model_state_dict'])
    vgae.eval()

    inference_data = MyGraphDataset(root_dir=PROCESSED_GRAPH_PATH)
    dataloader = DataLoader(inference_data, batch_size=1)
    loop = tqdm(dataloader, total = len(dataloader))
    with torch.no_grad():
        for batch in loop:
            x_input_vgae = batch.x * batch.train_mask # this is in range 1 - 3
            x_input_eval = (batch.x-1)*batch.train_mask  # this is in range 0 - 2 (as classes have to start at 0 for evaluation)

            ### VGAE
            X_raw_logits, _ , _ = vgae(x_input_vgae, batch.edge_index)
            X_reconstructed = F.softmax(X_raw_logits, dim=1)
            x_states = transform_to_states(X_reconstructed)


            #### Evaluation 
            recall, precision,f1 = basic_evaluation_metric(x_states*batch.train_mask, x_input_eval,batch.train_mask)
            custom_metric = custom_evaluation_metric(x_states)
            custom_metric_strict = custom_evaluation_metric_strict(x_states+1, batch.x)
            print(f"Recall: {recall}")
            print(f"Precision: {precision}")
            print(f"F1: {f1}")
            print(f"Custom Metric: {custom_metric}")
            print(f"Custom Metric strict: {custom_metric_strict}")
            return
                                    
def inference_baselines(PROCESSED_GRAPH_PATH, randomforest, knn, mice, majority_predictor):
    inference_data = MyGraphDataset(root_dir=PROCESSED_GRAPH_PATH)
    dataloader = DataLoader(inference_data, batch_size=1)
    loop = tqdm(dataloader, total = len(dataloader))
    total_f1_rf = torch.tensor([0.0,0.0,0.0])
    total_f1_knn = torch.tensor([0.0,0.0,0.0])
    total_f1_mice = torch.tensor([0.0,0.0,0.0])
    total_f1_maj = torch.tensor([0.0,0.0,0.0])
    i=1
    for batch in loop:
        x_input = batch.x*batch.train_mask
        if randomforest:
            x_reconstructed_rf = randomforest_impute(x_input)
            recall, precision,f1 = basic_evaluation_metric(x_reconstructed_rf, x_input, batch.train_mask)
            #custom_metric = custom_evaluation_metric(x_reconstructed_rf)
            #custom_metric_strict = custom_evaluation_metric_strict(x_reconstructed_rf, (batch.x+1))
            
        if knn:
            x_reconstructed_knn = knn_impute(x_input)
            recall, precision,f1 = basic_evaluation_metric(x_reconstructed_knn, x_input, batch.train_mask)
            #custom_metric = custom_evaluation_metric(x_reconstructed_knn)
            #custom_metric_strict = custom_evaluation_metric_strict(x_reconstructed_knn, (batch.x+1))
            total_f1_knn+=f1
        if mice:
            x_reconstructed_mice = knn_impute(x_input)
            recall, precision,f1 = basic_evaluation_metric(x_reconstructed_mice, x_input, batch.train_mask)
            #custom_metric = custom_evaluation_metric(x_reconstructed_knn)
            #custom_metric_strict = custom_evaluation_metric_strict(x_reconstructed_knn, (batch.x+1))
            total_f1_mice+=f1
        if majority_predictor:
            x_reconstructed_maj = fast_majority_class_predictor(x_input)
            recall, precision,f1 = basic_evaluation_metric(x_reconstructed_maj, x_input, batch.train_mask)
            total_f1_maj+=f1
        i+=1
        print(i)
    return total_f1_rf/i,total_f1_knn/i,total_f1_mice/i,total_f1_maj/i
        


        