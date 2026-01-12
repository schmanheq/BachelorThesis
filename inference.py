from python.VGAE.VGAE_inference import inference


def start_inference(processed_graph_path, weights_path):
    inference(INPUT_DIM=90,
              HIDDEN_DIM=128,
              Z_DIM=10,
              NUM_HIDDEN_LAYERS=2,
              NUM_CLASSES=3,
              PROCESSED_GRAPH_PATH=processed_graph_path,
              WEIGHTS_PATH = weights_path)
    
########## inference START ##########
processed_graph_path = 'processed_graphs_high'
weights_directory = 'dataset2_weights'
weights_path = f'/Users/dev/Documents/project/BachelorThesis/{weights_directory}/dataset2_graphs_high_weights.pt'
start_inference(processed_graph_path, weights_path)
########## inference END ##########
