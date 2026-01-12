import os
import torch
from torch_geometric.data import Dataset

class MyGraphDataset(Dataset):
    def __init__(self, root_dir):
        super().__init__(root_dir)
        self.root_dir = root_dir
        # Only store the count/names, not the data
        self.file_names = [f for f in os.listdir(root_dir) if f.endswith('.pt')]

    def len(self):
        return len(self.file_names)

    def get(self, idx):
        # This is called by the DataLoader only when a batch needs this graph
        return torch.load(os.path.join(self.root_dir, f'graph_{idx}.pt'), weights_only=False)