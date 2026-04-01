from python.VGAE.VGAE_inference import inference
from python.Baselines.baseline_inference import inference_baselines
import time


def start_VGAEinference(processed_graph_path, weights_path):
    inference(INPUT_DIM=90,
              HIDDEN_DIM=128,
              Z_DIM=10,
              NUM_HIDDEN_LAYERS_ENC=2,
              NUM_HIDDEN_LAYERS_DEC=2,
              NUM_CLASSES=3,
              PROCESSED_GRAPH_PATH=processed_graph_path,
              WEIGHTS_PATH = weights_path)
    


if __name__ == '__main__':


    ########## inference VGAE ##########
    def inf_VGAE():
        start = time.time()
        processed_graph_path = f'dataset0/inference_processed_data/processed_graphs_dataset0_med'
        weights_path = f'dataset0/dataset0_weights/dataset0_graphs_med_imbalance_08_9_02.pt'
        start_VGAEinference(processed_graph_path, weights_path)
        end = time.time()
        print(f"Seconds for VGAE Inference {end-start}")
    ########## inference VGAE ##########


    ########## inference Baselines ##########
    def inf_baselines():
        data = "dataset0/inference_processed_data/processed_graphs_dataset0_low"
        simple_imputer = False
        maj = False
        svd = True
        svd_path = "dataset0/dataset0_weights/baseline_svd_dataset0_low.pkl"
        start = time.time()
        inference_baselines(data, simple_imputer, maj, svd, svd_path)
        end = time.time()
        print("Seconds for Baseline Inference: " + str(end-start))
    
    inf_baselines()
    #inf_VGAE()

    ########## inference Baselines ##########