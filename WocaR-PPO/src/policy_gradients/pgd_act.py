import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
# from auto_LiRPA import BoundedModule, BoundedTensor, BoundedParameter
# from auto_LiRPA.perturbations import *

import random
import numpy as np
from policy_gradients.ibp import network_bounds
from policy_gradients.models import activation_with_name
forward_one = True
from policy_gradients.models import CtsPolicy, ValueDenseNet

'''
Target action: using gradient descent to find the worst action with worst worst-case q value
'''
def worst_action_pgd(q_net, policy_net, states, eps=0.0005, maxiter=100):
    with torch.no_grad():
        action_ub, action_lb = network_bounds(policy_net, states, eps)
        action_means, _ = policy_net(states)
    # print(action_means)
    # var_actions = Variable(action_means.clone().to(device), requires_grad=True)
    var_actions = action_means.requires_grad_()
    step_eps = (action_ub - action_lb) / maxiter

    for i in range(maxiter):
        worst_q = q_net(torch.cat((states, var_actions), dim=1))
        worst_q.backward(torch.ones_like(worst_q))
        grad = var_actions.grad.data  
        var_actions.data -= step_eps * torch.sign(grad)
        var_actions = torch.max(var_actions, action_lb)
        var_actions = torch.min(var_actions, action_ub)
        var_actions = var_actions.detach().requires_grad_() 
    q_net.zero_grad()
    return var_actions.detach()

def main():
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    random.seed(1234)
    np.random.seed(123)
    input_size = 17
    action_size = 6

    policy = CtsPolicy(state_dim=input_size, action_dim=action_size, init="orthogonal")
    q_model = ValueDenseNet(input_size+action_size, init="orthogonal")
    x =  torch.randn(3, input_size)
    start_time = time.time()
    worst_action, mean, ub, lb = worst_action_pgd(q_model, policy, x)
    print(ub, lb)
    print('time', time.time() - start_time)
    with torch.no_grad():
        worst_q = q_model(torch.cat((x, worst_action), dim=1))
        worst_q_mean = q_model(torch.cat((x, mean), dim=1))
        worst_q_ub = q_model(torch.cat((x, ub), dim=1))
        worst_q_lb = q_model(torch.cat((x, lb), dim=1))
    print('worst_action', worst_action)
    print('worst_q', worst_q)
    print('worst_q_mean', worst_q_mean)
    print('worst_q_ub', worst_q_ub)
    print('worst_q_lb', worst_q_lb)

if __name__ == "__main__":
    main()
    


    
    
        
