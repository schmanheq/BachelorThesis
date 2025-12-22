from python.Datageneration.datageneration import training_data_generation, get_gamma
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

def start_training():
    INPUT_DIM = 90
    HIDDEN_DIMS = 50
    Z_DIM = 10
    EPOCHS = 15
    NUM_HIDDEN_LAYERS = 2
    LR_RATE = 3e-4
    BETA = 0.1
    GAMMA = 0.8
    NUM_CLASSES = 3
    BATCH_SIZE = 1
    training_loop(INPUT_DIM, HIDDEN_DIMS, Z_DIM, EPOCHS, NUM_HIDDEN_LAYERS, LR_RATE, BETA, GAMMA, NUM_CLASSES, BATCH_SIZE)
    return 1


###### Datageneration ######
DIRECTORY = 'dataset_test'
NUM_SAMPLES = 10
NUM_NODES = 10000
NUM_ITERATIONS = 90
K_MEAN = 12
PATH_NETWORK = f'{DIRECTORY}/training_network.csv'
PATH_SNAPSHOTS = f'{DIRECTORY}/training_snapshots.csv'
PATH_LOGFILE = f'{DIRECTORY}/logfile.txt'
start_data_generation(NUM_SAMPLES,NUM_NODES, NUM_ITERATIONS,K_MEAN, PATH_NETWORK, PATH_SNAPSHOTS, PATH_LOGFILE)
###### Datageneration END ######

###### Training ######
INPUT_DIM = 90
HIDDEN_DIMS = 512
#start_training()
###### Training END ######

###### Inference ######
NUM_SAMPLES = 1
NUM_NODES = 10000
NUM_ITERATIONS = 90
K_MEAN = 12
PATH_NETWORK = 'data/inference_network.csv'
PATH_SNAPSHOTS = 'data/inference_snapshot.csv'
#start_data_generation(NUM_SAMPLES,NUM_NODES, NUM_ITERATIONS,K_MEAN, PATH_NETWORK, PATH_SNAPSHOTS)
#inference(INPUT_DIM, HIDDEN_DIMS, 10, 2,3)
###### Inference END ######
