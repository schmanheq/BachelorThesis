from python.Datageneration.datageneration import training_data_generation, save_snapshots_fast
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

def create_mask_optimize(node_feature, fraction_missing_data):
    rows = 10000
    cols = 90
    node_feature = np.ones((10000,90))
    num_to_mask = int(cols * fraction_missing_data)
    cols_idx = np.argsort(np.random.rand(rows, cols), axis=1)[:, :num_to_mask]
    mask = np.ones_like(node_feature, dtype=np.int8)
    rows_idx = np.arange(rows)[:, None]
    mask[rows_idx, cols_idx] = 0
    return mask 

def create_all_masks(num_samples, path_highest, path_medium, path_lowest):
    all_fractions = [(0.25,path_lowest), (0.33,path_medium), (0.5, path_highest)]
    for fraction,path in all_fractions:
        for i in range(num_samples):
            save_snapshots_fast(np.array(create_mask_optimize(None,fraction), dtype=np.int8), path)
    print("Masks created")

##### create masks ######
NUMSAMPLES = 2
PATH_MASKS_HIGH = 'dataset_test/masks_high.csv'
PATH_MASKS_MED = 'dataset_test/masks_med.csv'
PATH_MASKS_LOW = 'dataset_test/masks_low.csv'
#create_all_masks(NUMSAMPLES, PATH_MASKS_HIGH, PATH_MASKS_MED, PATH_MASKS_LOW)

###### Datageneration ######
DIRECTORY = 'dataset_inference'
NUM_SAMPLES = 2
NUM_NODES = 10000
NUM_ITERATIONS = 90
K_MEAN = 12
PATH_NETWORK = f'{DIRECTORY}/training_network.csv'
PATH_SNAPSHOTS = f'{DIRECTORY}/training_snapshots.csv'
PATH_LOGFILE = f'{DIRECTORY}/logfile.txt'
#start_data_generation(NUM_SAMPLES,NUM_NODES, NUM_ITERATIONS,K_MEAN, PATH_NETWORK, PATH_SNAPSHOTS, PATH_LOGFILE)
###### Datageneration END ######
