from python.VGAE.VGAE_training import training_loop
from python.Baselines.baseline_training import baseline_training
import numpy as np

def start_training(PATH_PROCESSED_GRAPHS, WEIGHTS_PATH):
    INPUT_DIM = 90
    HIDDEN_DIMS = 128
    Z_DIM = 10
    EPOCHS = 10
    NUM_HIDDEN_LAYERS = 2
    LR_RATE = 1e-4
    BETA = 0.1
    GAMMA = 0.8
    NUM_CLASSES = 3
    BATCH_SIZE = 16
    training_loop(INPUT_DIM, HIDDEN_DIMS, Z_DIM, EPOCHS, NUM_HIDDEN_LAYERS, LR_RATE, BETA, GAMMA, NUM_CLASSES, BATCH_SIZE, PATH_PROCESSED_GRAPHS, WEIGHTS_PATH)
    return 1

###### Training VGAE######
dataset = "dataset0"
PATH_PROCESSED_GRAPHS = 'processed_graphs_dataset0_low'
WEIGHTS_PATH= 'dataset2_graphs_low_imbalance_08_9_02.pt'
#start_training(PATH_PROCESSED_GRAPHS, WEIGHTS_PATH)
###### Training VGAE END ######

###### Training baselines ######
rf = False
knn = True
mice = True
process_graph_path = "training_data/processed_graphs_dataset0_high"
baseline_training(rf,knn,mice,process_graph_path)
###### Training baselines END ######

