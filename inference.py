from python.VGAE.VGAE_inference import inference
from python.Baselines.baseline_inference import inference_baselines
import time


def start_inference(processed_graph_path, weights_path):
    inference(INPUT_DIM=90,
              HIDDEN_DIM=128,
              Z_DIM=10,
              NUM_HIDDEN_LAYERS=2,
              NUM_CLASSES=3,
              PROCESSED_GRAPH_PATH=processed_graph_path,
              WEIGHTS_PATH = weights_path)
    


if __name__ == '__main__':


    ########## inference VGAE ##########
    def inf_VGAE():
        start = time.time()
        processed_graph_path = f'processed_graphs_dataset2_inference_high'
        weights_path = f'weights/dataset0_graphs_high_imbalance_08_9_02.pt'
        #start_VGAEinference(processed_graph_path, weights_path)
        end = time.time()
        print(f"Inference for VGAE {end-start}")
    ########## inference VGAE ##########


    ########## inference Baselines ##########
    def inf_baselines():
        data = "dataset0/inference_processed_data/processed_graphs_dataset0_high"
        simple_imputer = True
        maj = False
        svd = False
        svd_path = "baseline_svd_v1.pkl"
        start = time.time()
        inference_baselines(data, simple_imputer, maj, svd, svd_path)
        end = time.time()
        print(end-start)
    
    inf_baselines()
    
    ########## inference Baselines ##########