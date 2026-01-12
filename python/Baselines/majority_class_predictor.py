import numpy as np
from python.VGAE.VGAE_training import fetch_data_optimized, create_mask_optimize
from python.Datageneration.datageneration import save_snapshots_fast



def majority_class_predictor(adj_path, snapshots_path, fraction_missing_data):
    _, node_features = fetch_data_optimized(adj_path, snapshots_path)
    node_features = np.rot90(node_features, axes=(1, 2)).astype(int)
    masks = [create_mask_optimize(node_feature, fraction_missing_data) for node_feature in node_features]
    masks = np.array(masks)
    node_features = node_features*masks
    node_features = node_features[0]
    for i in range(len(node_features)):
        try:
            first_infected = np.where(node_features[i]==2)[0][0]
        except IndexError:
            first_infected=None
        try:
            first_recovered = np.where(node_features[i]==3)[0][0]
        except IndexError:
            first_recovered=None

        if first_recovered:
            node_features[i][:first_infected]=1
            node_features[i][first_infected:first_recovered]=2
            node_features[i][first_recovered:]=3
        elif first_infected:
            node_features[i][:first_infected]=1
            node_features[i][first_infected:]=2
        else:
            node_features[i][...]=1
    return node_features