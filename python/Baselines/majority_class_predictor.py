
import torch

def fast_majority_class_predictor(x_input):
    # Ensure we are working with a torch tensor
    # 1. Create masks for where the states '2' and '3' first appear
    is_2 = (x_input == 2)
    is_3 = (x_input == 3)
    
    # 2. Use cummax to "spread" the state to the right
    # Once a '2' or '3' is seen, the mask remains True for the rest of the row
    has_seen_2 = is_2.cummax(dim=1)[0]
    has_seen_3 = is_3.cummax(dim=1)[0]
    
    # 3. Apply logic using torch.where
    # Default state is 1 (Susceptible)
    out = torch.ones_like(x_input)
    
    # If we've seen a 2, it becomes 2
    out = torch.where(has_seen_2, torch.tensor(2, device=x_input.device), out)
    
    # If we've seen a 3, it becomes 3 (this overrides the 2 because 3 comes later)
    out = torch.where(has_seen_3, torch.tensor(3, device=x_input.device), out)
    
    return out