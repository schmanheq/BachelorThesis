from python.VGAE.VGAE_inference import inference, inference_baselines
import os


def start_inference(processed_graph_path, weights_path):
    inference(INPUT_DIM=90,
              HIDDEN_DIM=128,
              Z_DIM=10,
              NUM_HIDDEN_LAYERS=2,
              NUM_CLASSES=3,
              PROCESSED_GRAPH_PATH=processed_graph_path,
              WEIGHTS_PATH = weights_path)
    
    
########## inference START ##########
dataset = 'dataset0'
processed_graph_path = f'processed_graphs_dataset_inference_high'
weights_path = f'dataset0_weights/dataset0_graphs_high_imbalance_08_9_02.pt'
directory = 'dataset0_weights'
#start_inference(processed_graph_path, weights_path)
#################### ATTENTION ####################
random_forest = True
knn = False
mice = False
maj = False
rf, knn, mice, maj = inference_baselines(processed_graph_path, random_forest,knn, mice, maj)
print(rf, knn,mice, maj)
########## inference END ##########