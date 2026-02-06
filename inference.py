from python.VGAE.VGAE_inference import inference
from python.Baselines.baseline_inference import inference_baselines
import os
from concurrent.futures import ProcessPoolExecutor


def start_inference(processed_graph_path, weights_path):
    inference(INPUT_DIM=90,
              HIDDEN_DIM=128,
              Z_DIM=10,
              NUM_HIDDEN_LAYERS=2,
              NUM_CLASSES=3,
              PROCESSED_GRAPH_PATH=processed_graph_path,
              WEIGHTS_PATH = weights_path)
    


if __name__ == '__main__':

    ########## inference START ##########
    processed_graph_path = f'dataset0/inference_processed_data/processed_graphs_dataset0_high'
    weights_path = f'dataset0_weights/dataset0_graphs_high_imbalance_08_9_02.pt'
    random_forest = False
    knn = "dataset0/dataset0_weights/baseline_knn_v1_high.pkl"
    mice = False
    maj = False

    #start_inference(processed_graph_path, weights_path)
    results = inference_baselines(processed_graph_path, random_forest,knn, mice, maj)
    print(results)
    ########## inference END ##########