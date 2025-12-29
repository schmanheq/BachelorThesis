from python.Datageneration.datageneration import training_data_generation
from python.VGAE.VGAE_training import training_loop
from python.VGAE.VGAE_inference import inference
import numpy as np

def start_data_generation(NUM_SAMPLES,NUM_NODES, NUM_ITERATIONS,K_MEAN, path_network, path_snapshots, path_logfile):
    training_data_generation(num_samples=NUM_SAMPLES, 
                            num_nodes=NUM_NODES, 
                            k_mean=K_MEAN, 
                            recovery_rate=0.1,
                            num_iterations=NUM_ITERATIONS ,
                            path_network=path_network, 
                            path_snapshots=path_snapshots,
                            path_logfile=path_logfile,
                            show=False)
    print("data generated.")
    return 1

def start_training(dataset, missing_fraction, weights_path):
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
    DATASET = dataset
    training_loop(INPUT_DIM, HIDDEN_DIMS, Z_DIM, EPOCHS, NUM_HIDDEN_LAYERS, LR_RATE, BETA, GAMMA, NUM_CLASSES, BATCH_SIZE, DATASET, missing_fraction, weights_path)
    return 1


###### Datageneration ######
DIRECTORY = 'dataset2'
NUM_SAMPLES = 5000
NUM_NODES = 10000
NUM_ITERATIONS = 90
K_MEAN = 12
PATH_NETWORK = f'{DIRECTORY}/training_network.csv'
PATH_SNAPSHOTS = f'{DIRECTORY}/training_snapshots.csv'
PATH_LOGFILE = f'{DIRECTORY}/logfile.txt'
#start_data_generation(NUM_SAMPLES,NUM_NODES, NUM_ITERATIONS,K_MEAN, PATH_NETWORK, PATH_SNAPSHOTS, PATH_LOGFILE)
###### Datageneration END ######

###### Training ######
WEIGHTS_PATH = "vgae_mode_weights_dataset2_highest_missing.pt"
dataset = 'dataset2'
missing_fraction = 0.5
start_training(dataset, missing_fraction, WEIGHTS_PATH)
###### Training END ######

###### Inference ######
NUM_SAMPLES = 1
NUM_NODES = 10000
NUM_ITERATIONS = 90
K_MEAN = 12
PATH_NETWORK = 'dataset_inference/inference_network.csv'
PATH_SNAPSHOTS = 'dataset_inference/inference_snapshot.csv'
PATH_LOGFILE='dataset_inference/logfile.txt'
#start_data_generation(NUM_SAMPLES,NUM_NODES, NUM_ITERATIONS,K_MEAN, PATH_NETWORK, PATH_SNAPSHOTS,PATH_LOGFILE)
#inference(INPUT_DIM, HIDDEN_DIMS,PATH_NETWORK, PATH_SNAPSHOTS, Z_DIM=10, NUM_HIDDEN_LAYERS=2,NUM_CLASSES=3)
###### Inference END ######
