import torch
import numpy as np
from python.VGAE.VGAE_training import fetch_data_optimized, create_mask_optimize
from python.VGAE.VGAE_model import VariationalGraphAutoEncoder

def inference(INPUT_DIM, HIDDEN_DIM,PATH_NETWORK, PATH_SNAPSHOTS, Z_DIM, NUM_HIDDEN_LAYERS, NUM_CLASSES, MISSING_FRACTION):
    vgae = VariationalGraphAutoEncoder(INPUT_DIM,HIDDEN_DIM,Z_DIM,NUM_HIDDEN_LAYERS, NUM_CLASSES)
    torch.load('vgae_model_weights.pt')
    vgae.eval()

    edge_index, snapshots = fetch_data_optimized(PATH_NETWORK, PATH_SNAPSHOTS)
    snapshots = np.rot90(snapshots, axes=(1, 2)).astype(int)
    masked_snapshots = [create_mask_optimize(snapshot, MISSING_FRACTION)*snapshot for snapshot in snapshots]
    reconstructed = []
    with torch.no_grad():
        for i in range(len(masked_snapshots)):
            X_reconstructed, _ , _ = vgae(torch.tensor(masked_snapshots[i], dtype=torch.float), torch.tensor(edge_index[i], dtype=torch.int))
            print(X_reconstructed)
            break
        