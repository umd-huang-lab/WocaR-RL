import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# from auto_LiRPA import BoundedModule, BoundedTensor, BoundedParameter
# from auto_LiRPA.perturbations import *

import random
import numpy as np
from policy_gradients.ibp import network_bounds
from policy_gradients.models import activation_with_name
forward_one = True
from policy_gradients.models import CtsPolicy, ValueDenseNet
from policy_gradients.pgd_act import worst_action_pgd
from policy_gradients.convex_relaxation import RelaxedCtsPolicyForState

def worst_action_pgd(q_net, policy_net, states, eps=0.01, maxiter=100, lr=1e-4, device='cpu'):
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

def q_step(q_net, target_q_net, policy_net, states, actions, not_dones, next_states, rewards, gamma=0.99):
    curr_q = q_net(torch.cat((states, actions), dim=1))
    worst_actions = worst_action_pgd(q_net, policy_net, states)
    expected_q = rewards + gamma * not_dones * target_q_net(torch.cat((next_states, worst_actions), dim=1))
    '''
    print('curr_q', curr_q)
    print('expected_q', expected_q)
    '''
    
    q_loss = F.mse_loss(curr_q, expected_q)

    return q_loss

def main():
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    random.seed(1234)
    np.random.seed(123)
    input_size = 17
    action_size = 6

    x =  torch.randn(3, input_size)
    next_x = torch.randn(3, input_size)
    action = torch.randn(3, action_size)
    not_done = torch.ones(3, 1)
    reward = torch.randn(3, 1)

    policy = CtsPolicy(state_dim=input_size, action_dim=action_size, init="orthogonal")
    # relaxed_model = RelaxedCtsPolicyForState(state_dim=input_size, action_dim=action_size, policy_model = policy)
    q_model = ValueDenseNet(input_size+action_size, init="orthogonal")
    q_loss = q_step(q_model, q_model, policy, x, action, not_done, next_x, reward)
    print('q_loss', q_loss)


if __name__ == "__main__":
    main()
    
