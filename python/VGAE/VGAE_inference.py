import torch
from python.VGAE.VGAE_model import VariationalGraphAutoEncoder
from torch_geometric.loader import DataLoader
from ..Datageneration.dataloader import MyGraphDataset
from ..Evaluation.prob_to_states import transform_to_states
from ..Evaluation.evaluation_metrics import basic_evaluation_metric
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
            batch.x = batch.x-1
            x_input = batch.x * batch.train_mask
            X_raw_logits, _ , _ = vgae(x_input, batch.edge_index)
            X_reconstructed = F.softmax(X_raw_logits, dim=1)
            x_states = transform_to_states(X_reconstructed)
            basic_evaluation_metric(x_states*batch.train_mask, batch.x*batch.train_mask)



        