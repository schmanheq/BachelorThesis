from ..Datageneration.dataloader import MyGraphDataset
from ..Baselines.random_forest import rf_train
from ..Baselines.knn import knn_train
from ..Baselines.mice import mice_train
import numpy as np

def baseline_training(rf, knn,mice, data_path):
    training_data = MyGraphDataset(root_dir=data_path)
    all_x = []
    all_masks = []
    for data in training_data:
        all_x.append(data.x)
        all_masks.append(data.train_mask)
    all_X = np.vstack(all_x)
    all_Masks = np.vstack(all_masks)
    print("Data loaded.")
    if rf:
        rf_train(all_X,all_Masks)
    if knn:
        knn_train(all_X, all_masks)
    if mice:
        mice_train(all_X, all_masks)

        


