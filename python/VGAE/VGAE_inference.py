import torch
from python.VGAE.VGAE_model import VariationalGraphAutoEncoder
from torch_geometric.loader import DataLoader
from ..Datageneration.dataloader import MyGraphDataset
from ..Evaluation.prob_to_states import transform_to_states
from ..Evaluation.evaluation_metrics import basic_evaluation_metric, custom_evaluation_metric, custom_evaluation_metric_strict
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np


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
            x_input_eval = (batch.x)  # this is in range 0 - 2 (as classes have to start at 0 for evaluation)

            ### VGAE
            X_raw_logits, _ , _ = vgae(x_input_vgae, batch.edge_index)
            X_reconstructed = F.softmax(X_raw_logits, dim=1)
            x_states = transform_to_states(X_reconstructed)+1

            #### Evaluation 
            recall, precision,f1 = basic_evaluation_metric(x_states, x_input_eval,batch.train_mask)
            custom_metric = custom_evaluation_metric(x_states)
            custom_metric_strict = custom_evaluation_metric_strict(x_states+1, batch.x)
            print(f"Recall: {recall}")
            print(f"Precision: {precision}")
            print(f"F1: {f1}")
            print(f"Custom Metric: {custom_metric}")
            print(f"Custom Metric strict: {custom_metric_strict}")
            return
                                    