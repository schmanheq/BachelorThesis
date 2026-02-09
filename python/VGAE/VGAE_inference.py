import torch
from python.VGAE.VGAE_model import VariationalGraphAutoEncoder
from torch_geometric.loader import DataLoader
from ..Datageneration.dataloader import MyGraphDataset
from ..Evaluation.prob_to_states import transform_to_states
from ..Evaluation.evaluation_metrics import basic_evaluation_metric, custom_evaluation_metric, custom_evaluation_metric_strict, matth_coeff, confusion_matrix
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np


def inference(INPUT_DIM, HIDDEN_DIM, Z_DIM, NUM_HIDDEN_LAYERS_ENC, NUM_HIDDEN_LAYERS_DEC, NUM_CLASSES,PROCESSED_GRAPH_PATH, WEIGHTS_PATH):
    vgae = VariationalGraphAutoEncoder(INPUT_DIM,HIDDEN_DIM,Z_DIM,NUM_HIDDEN_LAYERS_ENC, NUM_HIDDEN_LAYERS_DEC, NUM_CLASSES)
    state_dict = torch.load(WEIGHTS_PATH, map_location=torch.device('cpu'))

    # 3. Load the weights into the model
    vgae.load_state_dict(state_dict['model_state_dict'])
    vgae.eval()
    batchsize = 1
    inference_data = MyGraphDataset(root_dir=PROCESSED_GRAPH_PATH)
    dataloader = DataLoader(inference_data, batch_size=batchsize)
    loop = tqdm(dataloader, total = len(dataloader))
    total_f1 = torch.tensor([0.0,0.0,0.0])
    total_mc = 0
    total_cf_matrix = torch.tensor([[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]])
    custom_metric_tracker = 0
    counter = 0
    with torch.no_grad():
        for batch in loop:
            x_input_vgae = batch.x * batch.train_mask # this is in range 1 - 3
            x_input_eval = (batch.x)  # this is in range 0 - 2 (as classes have to start at 0 for evaluation)

            ### VGAE
            X_raw_logits, _ , _ = vgae(x_input_vgae, batch.edge_index)
            X_reconstructed = F.softmax(X_raw_logits, dim=1)
            x_states = transform_to_states(X_reconstructed)+1

            #### Evaluation 
            _,_,f1 = basic_evaluation_metric(x_states, x_input_eval,batch.train_mask)
            m_coef = matth_coeff(x_states, x_input_eval,batch.train_mask)
            c_matrix= confusion_matrix(x_states, x_input_eval, batch.train_mask)
            custom_metric = custom_evaluation_metric(x_states, batchsize)
            
            counter+=1
            total_f1+=f1
            total_mc+=m_coef
            total_cf_matrix+=c_matrix

            custom_metric_tracker+=custom_metric
    print(f"Average F1: {total_f1/counter}")
    print(f"Average MC Coefficient: {total_mc/counter}")
    print(f"Average Confusion Matrix: ")
    print(total_cf_matrix/counter)
    print(f"Average Custom Metric: {custom_metric_tracker/counter}")
    return
                                    