import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class VariationalGraphAutoEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, z_channels, num_hidden_layers_enc, num_hidden_layers_dec, num_classes):
        super().__init__()
        self.num_classes = num_classes
        # encoder
        self.input_to_hidden = GCNConv(in_channels, hidden_channels)

        self.hidden_to_hidden_mu = nn.ModuleList([GCNConv(hidden_channels, hidden_channels) for _ in range(num_hidden_layers_enc)])
        self.hidden_to_mu = GCNConv(hidden_channels, z_channels)

        self.hidden_to_hidden_si = nn.ModuleList([GCNConv(hidden_channels, hidden_channels) for _ in range(num_hidden_layers_enc)])
        self.hidden_to_si = GCNConv(hidden_channels, z_channels)

        #decoder 
        self.z_to_hidden = GCNConv(z_channels, hidden_channels)
        self.hidden_to_hidden_input_reconstructed = nn.ModuleList(GCNConv(hidden_channels, hidden_channels) for _ in range(num_hidden_layers_dec))
        self.hidden_to_input_reconstructed = GCNConv(hidden_channels, in_channels*num_classes)

    def encode(self, x, edge_index):
        x = F.relu(self.input_to_hidden(x, edge_index))

        mu = x
        for layer in self.hidden_to_hidden_mu:
            mu = F.relu(layer(mu, edge_index))
        mu = F.relu(self.hidden_to_mu(mu, edge_index))

        si = x
        for layer in self.hidden_to_hidden_si:
            si = F.relu(layer(si, edge_index))
        si = F.sigmoid(self.hidden_to_si(si, edge_index))
        return mu, si
    
    def decode(self, z, edge_index):
        x_reconstructed = self.z_to_hidden(z, edge_index)
        for layer in self.hidden_to_hidden_input_reconstructed:
            x_reconstructed = F.relu(layer(x_reconstructed, edge_index))
        x_raw_logits = self.hidden_to_input_reconstructed(x_reconstructed, edge_index)
        x_raw_logits = x_raw_logits.view(-1, self.num_classes)
        #x_probabilities = F.softmax(x_raw_logits, dim=1) # apparently its crucial to not use softmax as later in the CE loss does require raw logits
        return x_raw_logits
    
    def forward(self, x, edge_index):
        mu,sigma = self.encode(x, edge_index)
        e = torch.randn_like(sigma)
        z_reparameterized = mu + sigma*e
        x_reconstructed = self.decode(z_reparameterized, edge_index)
        return x_reconstructed, mu, sigma


