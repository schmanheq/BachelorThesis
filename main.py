from python.Datageneration.datageneration import training_data_generation
from python.VGAE.VGAE_training import training_loop

def start_data_generation(NUM_SAMPLES,NUM_NODES, NUM_ITERATIONS,K_MEAN):
    training_data_generation(num_samples=NUM_SAMPLES, 
                            num_nodes=NUM_NODES, 
                            k_mean=K_MEAN, 
                            infection_rate=0.33, 
                            recovery_rate=0.1,
                            num_iterations=NUM_ITERATIONS ,
                            path_network='data/training_network.csv', 
                            path_snapshots='data/training_snapshots.csv')
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

NUM_SAMPLES = 1
NUM_NODES = 10000
NUM_ITERATIONS = 90
K_MEAN = 12
#start_data_generation(NUM_SAMPLES,NUM_NODES, NUM_ITERATIONS,K_MEAN)

INPUT_DIM = 90
HIDDEN_DIMS = 512
start_training()
