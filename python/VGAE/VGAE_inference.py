import torch
import numpy as np
from python.VGAE.VGAE_training import fetch_data, create_mask
from python.VGAE.VGAE_model import VariationalGraphAutoEncoder
INPUT_DIM = 11
HIDDEN_DIM = 50
Z_DIM = 10
NUM_HIDDEN_LAYERS = 2
NUM_CLASSES = 3

vgae = VariationalGraphAutoEncoder(INPUT_DIM,HIDDEN_DIM,Z_DIM,NUM_HIDDEN_LAYERS, NUM_CLASSES)
torch.load('../vgae_model_weights.pt')
vgae.eval()

edge_index, snapshots = fetch_data('../data/inference_network.csv', '../data/inference_snapshots.csv')
snapshots = np.rot90(snapshots, axes=(1, 2)).astype(int)
masked_snapshots = [create_mask(snapshot)*snapshot for snapshot in snapshots]
reconstructed = []
with torch.no_grad():
    for i in range(len(masked_snapshots)):
        X_reconstructed, _ , _ = vgae(torch.tensor(masked_snapshots[i], dtype=torch.float), torch.tensor(edge_index[i], dtype=torch.int))
        print(X_reconstructed)
        break
        