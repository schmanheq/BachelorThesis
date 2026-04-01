from ..Datageneration.dataloader import MyGraphDataset
from ..Baselines.svdImputer import svd_training
import numpy as np

def baseline_training(data_path, model_path):
    training_data = MyGraphDataset(root_dir=data_path)
    all_x = []
    for data in training_data:
        all_x.append(data.x)
    print("Data loaded.")
    svd = svd_training(all_x,model_path)
    return svd



