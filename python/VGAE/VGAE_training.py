from torch_geometric.data import Data
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn.utils as utils
from torch_geometric.loader import DataLoader
from .VGAE_model import VariationalGraphAutoEncoder
import csv

def create_mask_optimize(node_feature, fraction_missing_data):
    rows, cols = node_feature.shape
    num_to_mask = int(cols * fraction_missing_data)
    
    # This generates unique indices per row by "shuffling" via sorting
    # It's faster than loops and more accurate than randint
    cols_idx = np.argsort(np.random.rand(rows, cols), axis=1)[:, :num_to_mask]
    
    mask = np.ones_like(node_feature, dtype=np.int8)
    rows_idx = np.arange(rows)[:, None]
    mask[rows_idx, cols_idx] = 0
    
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

def fetch_data_optimized(adj_path, snapshots_path):
    def parse_file(path):
        data = []
        current_sub = []
        
        with open(path, 'r') as f:
            # csv.reader is significantly faster than manual string splitting
            reader = csv.reader(f)
            for row in reader:
                if not row or '_' in row[0]:
                    if current_sub:
                        data.append(current_sub)
                        current_sub = []
                else:
                    # Convert to int immediately to save memory and time
                    current_sub.append([int(x) for x in row])
            
            # Catch the last sub-index if the file doesn't end with '_'
            if current_sub:
                data.append(current_sub)
                
        return np.array(data, dtype=int)

    edge_index = parse_file(adj_path)
    node_features = parse_file(snapshots_path)
    
    return edge_index, node_features

def loss_function(X_reconstructed, X, mu, si,mask, beta, gamma, N_total, timestamps, batch_size):
    X_reconstructed = X_reconstructed.view(-1,90, 3)

    def turn_X_into_one_hot_encoded(X, numclasses):
        X_indices = X.long()-1
        X_hot_encoded = F.one_hot(X_indices, num_classes=numclasses)
        return X_hot_encoded.float()
    
    def categorical_cross_entropy(X_res, X_orig, mask):
        logits = X_res.reshape(-1, 3)
        targets = (X_orig.long() - 1).reshape(-1)
        
        ce = F.cross_entropy(logits, targets, reduction='none')
        return ce.mean()
    
    def kl_divergence(mu, si, beta):
        si_min_max = 2.0
        si_clipped = torch.clamp(si, min=-si_min_max, max=si_min_max)
        kl_div = -0.5 * torch.sum(1 + 2 * si_clipped - mu.pow(2) - torch.exp(2 * si_clipped))
        return beta*kl_div/batch_size
    
    def temporal_smoothness_contraint(X_reconstructed, gamma,N_total, timestamps): 
        log_p = F.softmax(X_reconstructed, dim=-1)
        log_p_t = log_p[:, :-1, :]
        log_p_t_plus_1 = log_p[:, :, 1:, :]
        p_t = torch.exp(log_p_t)
        p_t_plus_1 = torch.exp(log_p_t_plus_1)
        kld_forward = F.kl_div(log_p_t, p_t_plus_1.detach(), reduction='batchmean')
        kld_backward = F.kl_div(log_p_t_plus_1, p_t.detach(), reduction='batchmean')
        total_kld = (kld_forward + kld_backward) / (N_total * (timestamps - 1))
        return gamma * total_kld
    
    #X = turn_X_into_one_hot_encoded(X, numclasses=3) # apparently it work with just the raw classes better performance wise than with hot encoded labels
    loss = categorical_cross_entropy(X_reconstructed, X, mask) + kl_divergence(mu, si, beta) #+ temporal_smoothness_contraint(X_reconstructed, gamma, N_total, timestamps)
    return loss

def training_loop(input_dim, hidden_dim, z_dim, epochs, num_hidden_layers, lr_rate, beta, gamma, num_classes, batchsize, data_pathway, fraction_missing_data, weights_path):
    DEVICE = torch.device("cuda" if  torch.cuda.is_available() else "cpu")
    print(f"device: {DEVICE}")
    edge_index ,node_features= fetch_data_optimized(f'{data_pathway}/training_network.csv', f'{data_pathway}/training_snapshots.csv')
    print(f"Node features shape: {node_features.shape}")
    _, t_timestamps, n_nodes = node_features.shape
    node_features = np.rot90(node_features, axes=(1, 2)).astype(int)
    masks = [create_mask_optimize(node_feature, fraction_missing_data) for node_feature in node_features]
    masks = np.array(masks)
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
            batch = batch.to(DEVICE) #batch.x has shape batch_size*10000,90
            x_input = batch.x * batch.mask #apply the mask to the input
            x_reconstructed, mu, si = vgae(x_input, batch.edge_index) #x_reconstructed has shape batch_size*10000*90,3
            loss = loss_function(x_reconstructed, batch.x, mu, si,batch.mask, beta, gamma,n_nodes, t_timestamps, batchsize)#calc the loss using the reconstructed and original nodefeatures
            optimizer.zero_grad()
            loss.backward()
            utils.clip_grad_norm_(vgae.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")



    PATH = weights_path
    save_data = {
        'epoch': epoch, 
        'model_state_dict': vgae.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': total_loss/len(dataloader), 
    }
    torch.save(save_data, PATH)
    print(f"Model weights and metadata saved to {PATH}")