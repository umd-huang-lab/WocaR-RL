import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from auto_LiRPA import BoundedModule, BoundedTensor, BoundedParameter
from auto_LiRPA.perturbations import *

import random
import numpy as np
from policy_gradients.ibp import network_bounds
from policy_gradients.models import activation_with_name
forward_one = True
from policy_gradients.models import CtsPolicy

def main():
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    random.seed(1234)
    np.random.seed(123)
    input_size = 17
    action_size = 6

    model = CtsPolicy(state_dim=input_size, action_dim=action_size, init="orthogonal")

    x = torch.randn(1, input_size)
    means, _ = model(x)
    # relaxed_means = relaxed_model(x)
    print('prediction', means)
    #print('relaxed pred', relaxed_means)
    
    with torch.no_grad():
        ub, lb = network_bounds(model, x, 0.01)
        # rub, rlb = network_bounds(relaxed_model, x, 0.01)

    print('upper bound', ub)
    print('lower bound', lb)

if __name__ == "__main__":
    main()