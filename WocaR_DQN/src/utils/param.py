### Store the parameters
import os
import torch

class Param:
    dtype = None
    device = None
    # get absolute dir
    root_dir = os.getcwd()
    release_model_dir = os.path.join(root_dir, r'released_models')
    model_dir = os.path.join(root_dir, r'learned_models')
    adv_dir = os.path.join(root_dir, r'learned_adv')
    data_dir = os.path.join(root_dir, r'data')
    plot_dir = os.path.join(root_dir, r'plot')

    def __init__(self, dtype=None, device=None, 
                 input_size=None, output_size=None,
                 n_layers=None, size=None,
                 learning_rate_schedule=None,
                 exploration_schedule=None):
        if dtype is not None:
            Param.dtype = dtype
        if device is not None:
            Param.device = device
        self.input_size = input_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.size = size
        self.learning_rate_schedule = learning_rate_schedule
        self.exploration_schedule = exploration_schedule
    def get():
        return (Param.dtype, Param.device)
    
def from_numpy(n_array, dtype=None):
    if dtype is None:
        return torch.from_numpy(n_array).to(Param.device).type(Param.dtype)
    else:
        return torch.from_numpy(n_array).to(Param.device).type(dtype)
    
    