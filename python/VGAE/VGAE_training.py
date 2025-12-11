from torch_geometric.data import Data
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn.utils as utils
from torch_geometric.loader import DataLoader
from .VGAE_model import VariationalGraphAutoEncoder
from torchinfo import summary

def create_mask(node_feature):
    mask = np.ones_like(node_feature, dtype=np.int8)
    rows, cols = node_feature.shape
    rows_idx = np.arange(rows)
    cols_idx = np.random.randint(0, cols, size=rows)
    mask[rows_idx, cols_idx]=0
    return mask

def fetch_data(adj_path, snapshots_path):
    edge_index = []
    sub_edge_index = []
    with open(adj_path, 'r') as file:
        for line in file:
            if '_' in line:
                edge_index.append(sub_edge_index)
                sub_edge_index=[]
            else:
                line = line.strip()
                line = line.split(',')
                sub_edge_index.append(line)
                
    node_features = []
    sub_node_features = []
    with open(snapshots_path,'r') as file2:
        for line in file2:
            if '_' in line:
                node_features.append(sub_node_features)
                sub_node_features = []
            else:
                line = line.strip()
                line = line.split(',')
                sub_node_features.append(line)
    return np.array(edge_index,dtype=int), np.array(node_features,dtype=int)

def loss_function(X_reconstructed, X, mu, si,mask, beta, gamma, N_total, timestamps):
    def turn_X_into_one_hot_encoded(X, numclasses):
        X_indices = X.long()-1
        X_hot_encoded = F.one_hot(X_indices, num_classes=numclasses)
        return X_hot_encoded.float()
    
    def categorical_cross_entropy(X_reconstructed, X, mask):
        X = X.flatten()
        X_reconstructed = X_reconstructed.flatten()
        categorical_ce = F.cross_entropy(X_reconstructed,X)
        return categorical_ce
    
    def kl_divergence(mu, si, beta):
        si_min_max = 2.0
        si_clipped = torch.clamp(si, min=-si_min_max, max=si_min_max)
        kl_div = -0.5 * torch.sum(1 + 2 * si_clipped - mu.pow(2) - torch.exp(2 * si_clipped))
        return beta*kl_div
    
    def temporal_smoothness_contraint(X_reconstructed, gamma,N_total, timestamps): 
        X_reshaped = X_reconstructed.view(N_total,timestamps,3)
        log_p = F.softmax(X_reshaped, dim=2)
        log_p_t = log_p[:, :-1, :]
        log_p_t_plus_1 = log_p[:, 1:, :]
        p_t = torch.exp(log_p_t)
        p_t_plus_1 = torch.exp(log_p_t_plus_1)
        kld_forward = F.kl_div(log_p_t, p_t_plus_1.detach(), reduction='sum')
        kld_backward = F.kl_div(log_p_t_plus_1, p_t.detach(), reduction='sum')
        total_kld = (kld_forward + kld_backward) / (N_total * (timestamps - 1))
        return gamma * total_kld
    
    X = turn_X_into_one_hot_encoded(X, numclasses=3)
    loss = categorical_cross_entropy(X_reconstructed, X, mask) + kl_divergence(mu, si, beta) + temporal_smoothness_contraint(X_reconstructed, gamma, N_total, timestamps)
    return loss

def training_loop(input_dim, hidden_dim, z_dim, epochs, num_hidden_layers, lr_rate, beta, gamma, num_classes, batchsize):
    DEVICE = torch.device("cuda" if  torch.cuda.is_available() else "cpu")
    edge_index ,node_features= fetch_data('data/training_network.csv', 'data/training_snapshots.csv')
    _, t_timestamps, n_nodes = node_features.shape
    node_features = np.rot90(node_features, axes=(1, 2)).astype(int)
    masks = [create_mask(node_feature) for node_feature in node_features]
    training_data = []

    for i in range(len(masks)):
        data = Data(x=torch.tensor(node_features[i],dtype=torch.float), edge_index=torch.tensor(edge_index[i], dtype=torch.int))
        data.mask = torch.tensor(masks[i], dtype=torch.int)
        training_data.append(data)
    dataloader = DataLoader(training_data, batch_size=batchsize, shuffle=True)


    vgae = VariationalGraphAutoEncoder(input_dim, hidden_dim, z_dim, num_hidden_layers,num_classes).to(DEVICE)
    optimizer = torch.optim.Adam(vgae.parameters(), lr=lr_rate)

    for epoch in range(epochs):
        vgae.train()
        loop = tqdm(dataloader, total=len(dataloader))
        total_loss = 0

        for batch in loop:
            batch = batch.to(DEVICE)
            x_reconstructed, mu, si = vgae(batch.x, batch.edge_index)
            loss = loss_function(x_reconstructed, batch.x, mu, si,batch.mask, beta, gamma,n_nodes, t_timestamps)

            optimizer.zero_grad()
            loss.backward()
            utils.clip_grad_norm_(vgae.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")


    PATH = "vgae_model_weights.pt"
    save_data = {
        'epoch': epoch, 
        'model_state_dict': vgae.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': total_loss/len(dataloader), 
    }
    torch.save(save_data, PATH)
    print(f"Model weights and metadata saved to {PATH}")