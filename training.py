from python.VGAE.VGAE_training import training_loop
from python.Baselines.baseline_training import baseline_training
import numpy as np
import time

def start_training(PATH_PROCESSED_GRAPHS, WEIGHTS_PATH):
    INPUT_DIM = 90
    HIDDEN_DIMS = 128
    Z_DIM = 10
    EPOCHS = 10
    NUM_HIDDEN_LAYERS_ENC = 2
    NUM_HIDDEN_LAYERS_DEC = 2
    LR_RATE = 1e-4
    BETA = 0.1
    GAMMA = 0.8
    NUM_CLASSES = 3
    BATCH_SIZE = 16
    training_loop(INPUT_DIM, HIDDEN_DIMS, Z_DIM, EPOCHS, NUM_HIDDEN_LAYERS_ENC, NUM_HIDDEN_LAYERS_DEC, LR_RATE, BETA, GAMMA, NUM_CLASSES, BATCH_SIZE, PATH_PROCESSED_GRAPHS, WEIGHTS_PATH)
    return 1

###### Training VGAE######
dataset = "dataset0"
PATH_PROCESSED_GRAPHS = 'dataset0/training_processed_data/processed_graphs_dataset0_high'
WEIGHTS_PATH= 'dataset2_graphs_high_imbalance_111_166_02.pt'
start_training(PATH_PROCESSED_GRAPHS, WEIGHTS_PATH)
print("Finished VGAE Training, saved to: " + WEIGHTS_PATH)
###### Training VGAE END ######

###### Training baselines ######
start = time.time()
process_graph_path = "dataset0/inference_processed_data/processed_graphs_dataset0_high"
#baseline_training(process_graph_path)
end = time.time()
print(end-start)
###### Training baselines END ######

