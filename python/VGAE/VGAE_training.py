from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn.utils as utils
from torch_geometric.loader import DataLoader
from .VGAE_model import VariationalGraphAutoEncoder
from ..Datageneration.dataloader import MyGraphDataset

def handle_class_imbalance(x_input, class_imbalance_count):
    susceptibles = (x_input==1).sum()
    infected = (x_input==2).sum()
    recovered = (x_input==3).sum()
    new_class_count=class_imbalance_count+torch.tensor([susceptibles, infected, recovered])
    imbalance_weights = 1/new_class_count
    imbalance_weights = imbalance_weights/imbalance_weights.sum()
    return new_class_count, imbalance_weights

def loss_function(X_reconstructed, X, mu, si, beta, gamma, N_total, timestamps, batch_size, imbalance_weights, edge_index):
    X_reconstructed = X_reconstructed.view(-1,90, 3)

    def categorical_cross_entropy(X_res, X_orig, imbalance_weights):
        logits = X_res.reshape(-1, 3)
        imbalance_weights = imbalance_weights.to(logits.device)
        targets = (X_orig.long() - 1).reshape(-1)
        ce = F.cross_entropy(logits, targets,weight=imbalance_weights, reduction='mean')
        return ce
    
    def kl_divergence(mu, si, beta):
        si_min_max = 2.0
        si_clipped = torch.clamp(si, min=-si_min_max, max=si_min_max)
        kl_div = -0.5 * torch.sum(1 + 2 * si_clipped - mu.pow(2) - torch.exp(2 * si_clipped))
        return beta*kl_div/batch_size
    
    def smoothness_constraint_optimized(X_reconstructed, gamma, edge_index):
        node_sums = X_reconstructed.sum(dim=1)
        row, col = edge_index
        diffs = (node_sums[row] - node_sums[col])**2
        return gamma * diffs.sum()
    
    def temporal_smoothness_contraint(X_reconstructed, gamma,N_total, timestamps): 
        log_p = F.softmax(X_reconstructed, dim=-1)
        log_p_t = log_p[:, :-1, :]
        log_p_t_plus_1 = log_p[:, 1:, :]
        p_t = torch.exp(log_p_t)
        p_t_plus_1 = torch.exp(log_p_t_plus_1)
        kld_forward = F.kl_div(log_p_t, p_t_plus_1.detach(), reduction='batchmean')
        kld_backward = F.kl_div(log_p_t_plus_1, p_t.detach(), reduction='batchmean')
        total_kld = (kld_forward + kld_backward) / (N_total * (timestamps - 1))
        return gamma * total_kld
    
    #X = turn_X_into_one_hot_encoded(X, numclasses=3) # apparently it work with just the raw classes better performance wise than with hot encoded labels
    loss = categorical_cross_entropy(X_reconstructed, X, imbalance_weights) + kl_divergence(mu, si, beta) + smoothness_constraint_optimized(X_reconstructed, gamma, edge_index)# +temporal_smoothness_contraint(X_reconstructed, gamma, N_total, timestamps)
    return loss

def training_loop(input_dim, hidden_dim, z_dim, epochs, num_hidden_layers, lr_rate, beta, gamma, num_classes, batchsize, path_processed_graphs, weights_path):
    n_nodes = 10000
    t_timestamps = 90
    DEVICE = torch.device("cuda" if  torch.cuda.is_available() else "cpu")
    print(f"device: {DEVICE}")
    training_data = MyGraphDataset(root_dir=path_processed_graphs)
    dataloader = DataLoader(training_data, batch_size=batchsize, shuffle=True)
    vgae = VariationalGraphAutoEncoder(input_dim, hidden_dim, z_dim, num_hidden_layers,num_classes).to(DEVICE)
    optimizer = torch.optim.Adam(vgae.parameters(), lr=lr_rate)
    class_imbalance_count = torch.tensor([0,0,0])
    imbalance_weights = torch.tensor([0.08, 0.9, 0.02])

    for epoch in range(epochs):
        vgae.train()
        loop = tqdm(dataloader, total=len(dataloader))
        total_loss = 0

        for batch in loop:
            class_imbalance_count, imbalance_weights = handle_class_imbalance(batch.x, class_imbalance_count)
            batch = batch.to(DEVICE) #batch.x has shape batch_size*10000,90
            x_input = batch.x * batch.train_mask #apply the mask to the input
            x_reconstructed, mu, si = vgae(x_input, batch.edge_index) #x_reconstructed has shape batch_size*10000*90,3
            loss = loss_function(x_reconstructed, batch.x, mu, si, beta, gamma,n_nodes, t_timestamps, batchsize, imbalance_weights, batch.edge_index)#calc the loss using the reconstructed and original nodefeatures
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