import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import random
import numpy as np
from policy_gradients.models import activation_with_name
forward_one = True
from policy_gradients.models import CtsPolicy, ValueDenseNet

def initial_bounds(x0, epsilon):
    '''
    x0 = input, b x c x h x w
    '''
    upper = x0+epsilon
    lower = x0-epsilon
    return upper, lower

def weighted_bound(layer, prev_upper, prev_lower):
    prev_mu = (prev_upper + prev_lower)/2
    prev_r = (prev_upper - prev_lower)/2
    mu = layer(prev_mu)
    if type(layer)==nn.Linear:
        r = F.linear(prev_r, torch.abs(layer.weight))
    elif type(layer)==nn.Conv2d:
        r = F.conv2d(prev_r, torch.abs(layer.weight), stride=layer.stride, padding=layer.padding)
    
    upper = mu + r
    lower = mu - r
    return upper, lower

def activation_bound(layer, prev_upper, prev_lower):
    upper = layer(prev_upper)
    lower = layer(prev_lower)
    return upper, lower

def network_bounds(model, x0, epsilon):
    '''
    get inteval bound progation upper and lower bounds for the actiavtion of a model
    
    model: a nn.Sequential module
    x0: input, b x input_shape
    epsilon: float, the linf distance bound is calculated over
    '''
    upper, lower = initial_bounds(x0, epsilon)
    for layer in model.modules():
        if type(layer) in (nn.Sequential,) or type(layer) in (nn.ModuleList, ):
            pass
        elif type(layer) in (nn.ReLU, nn.Sigmoid, nn.Tanh, nn.MaxPool2d, nn.Flatten):
            # print('layer:', type(layer))
            upper, lower = activation_bound(layer, upper, lower)
        elif type(layer) in (nn.Linear, nn.Conv2d):
            # print('layer:', type(layer))
            upper, lower = weighted_bound(layer, upper, lower)
        else:
            # print('Unsupported layer:', type(layer))
            pass
    return upper, lower

def main():
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    random.seed(1234)
    np.random.seed(123)
    input_size = 17
    action_size = 6

    policy = CtsPolicy(state_dim=input_size, action_dim=action_size, init="orthogonal")
    x =  torch.randn(3, input_size)
    action_ub, action_lb = network_bounds(policy, x, epsilon=0.0005)
    print(action_ub, action_lb)
    mean, std = policy(x)

    print(mean, std)


if __name__ == "__main__":
    main()
