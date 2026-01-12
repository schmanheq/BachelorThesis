import torch
def transform_to_states(x_reconstructed):
    x_states = torch.argmax(x_reconstructed, dim=1)
    x_states = x_states.view((10000,90))
    return x_states