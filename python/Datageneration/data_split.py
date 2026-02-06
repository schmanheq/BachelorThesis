import torch
import csv
import os
from torch_geometric.data import Data
import numpy as np

def split_giant_files(adj_csv, feat_csv, mask_csv, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    
    # Open all files as iterators (Zero memory used here)
    with open(mask_csv, 'r') as f_mask, open(feat_csv, 'r') as f_feat, open(adj_csv, 'r') as f_adj:
        reader_mask = csv.reader(f_mask)
        reader_feat = csv.reader(f_feat)
        reader_adj = csv.reader(f_adj)

        graph_id = 0
        masks = []

        print("Starting split...")
        for row_m in reader_mask:
            # Check for separator in the MASK file
            if '_' in row_m[0]:
                # 1. Fetch features for THIS graph only
                features = []
                while True:
                    try:
                        row_f = next(reader_feat)
                        if '_' in row_f[0]: # Stop at feature separator
                            break
                        features.append([float(val) for val in row_f])
                    except StopIteration:
                        break
                
                # Process features
                features_np = np.rot90(np.array(features), k=1)
                x = torch.tensor(np.ascontiguousarray(features_np), dtype=torch.float)

                # 2. Fetch network (Adjacency) - exactly 2 lines
                try:
                    row_edge1 = next(reader_adj)
                    row_edge2 = next(reader_adj)
                    # Skip the separator line in the adjacency file
                    _ = next(reader_adj) 
                    
                    edges = [
                        [int(v) for v in row_edge1],
                        [int(v) for v in row_edge2]
                    ]
                    edge_index = torch.tensor(edges, dtype=torch.long).contiguous()
                    print(edge_index.shape)
                except StopIteration:
                    print(f"Warning: Adjacency file ended early at graph {graph_id}")
                    break
                # 3. Process Mask (which we collected in the 'else' block)
                mask_tensor = torch.tensor(masks, dtype=torch.bool)

                # 4. Save Data Object
                data = Data(x=x, edge_index=edge_index, train_mask=mask_tensor)
                torch.save(data, os.path.join(target_dir, f'graph_{graph_id}.pt'))
                
                # Reset for next graph
                graph_id += 1
                masks = []
                if graph_id % 100 == 0:
                    print(f"Saved {graph_id} graphs...")
            else:
                # Still reading mask rows for the current graph
                masks.append([int(val) for val in row_m])

    print(f"Done! Saved {graph_id} graphs to {target_dir}")

# Call the function
split_giant_files('inference/training_network.csv',
                  'inference/training_snapshots.csv', 
                  'masks/masks_low.csv', 
                  'processed_graphs_dataset0_low')