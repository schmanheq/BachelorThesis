from python.VGAE.VGAE_training import training_loop
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
    BATCH_SIZE = 2
    training_loop(INPUT_DIM, HIDDEN_DIMS, Z_DIM, EPOCHS, NUM_HIDDEN_LAYERS, LR_RATE, BETA, GAMMA, NUM_CLASSES, BATCH_SIZE, PATH_PROCESSED_GRAPHS, WEIGHTS_PATH)
    return 1

###### Training ######
PATH_PROCESSED_GRAPHS = 'processed_graphs_high'
WEIGHTS_PATH = 'test_weights.pt'
start_training(PATH_PROCESSED_GRAPHS, WEIGHTS_PATH)
###### Training END ######
