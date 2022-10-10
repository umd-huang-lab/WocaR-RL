import random
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import os 
import copy
from collections import namedtuple, deque
from itertools import count
from VaR_DQN.utils.param import Param
from VaR_DQN.utils.worst_action import *
from VaR_DQN.utils.torch_utils import soft_update, hard_update
from VaR_DQN.utils.ibp import network_bounds, worst_action_select


### Rollout the Q-learning Agent in the given environment
def rollout(agent, env, num_trajectories=3, num_steps=1000):
    rews = []
    for i in range(num_trajectories):
        o = env.reset()
        total_rew = 0
        for t in range(num_steps):
            a = int(agent.step(o))
            (o, reward, done, _info) = env.step(a)
            total_rew += reward
            if done: break
        rews.append(total_rew)
    return sum(rews)/len(rews)

### Rollout the Q-learning Agent in the given Atari environment
### The only difference here is how the reward is tracked.
### Unlike previously we treat environment as terminated if 
### done is Ture, here we only end the loop if 'episode' appears
### in the info provided by the wrapper. In addition, we use 
### reward provided in the info when 'episode' is True, not the
### one accumulated from the reward provided by the environment at
### every step.
def roll_out_atari(agent, env, num_trajectories=3, max_steps=15000):
    rews = []
    for episode in range(num_trajectories):
        obs = env.reset()
        r = 0
        for t in count():
            action = agent.select_epilson_greedy_action(obs/255., 0.01)
            obs, reward, done, info = env.step(action)
            r+=reward
            if t>max_steps:
                print('Maximum {} Steps Reached'.format(max_steps))
                print(r)
                break
            if 'episode' in info.keys():
                rews.append(info['episode']['r'])
                break
    return sum(rews)/len(rews) if len(rews)>0 else np.nan

def build_mlp(
        input_size,
        output_size,
        n_layers,
        size,
        activation = nn.ReLU(),
        output_activation = nn.Identity(),
        init_method=None,
):

    layers = []
    in_size = input_size
    for _ in range(n_layers):
        curr_layer = nn.Linear(in_size, size)
        if init_method is not None:
            curr_layer.apply(init_method)
        layers.append(curr_layer)
        layers.append(activation)
        in_size = size

    last_layer = nn.Linear(in_size, output_size)
    if init_method is not None:
        last_layer.apply(init_method)

    layers.append(last_layer)
    layers.append(output_activation)
        
    return nn.Sequential(*layers)


### Architecture is the same as the one published in the original nature paper
### see https://www.nature.com/articles/nature14236
class Q_Atari(nn.Module):
    def __init__(self, in_channels=1, num_actions=18, radial=False):
        super(Q_Atari, self).__init__()
        self.num_actions = num_actions
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(64*6*6, 512) if radial else nn.Linear(64*7*7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
    def forward(self, x):
        if len(x.shape)==3:
            x = x.unsqueeze(0)
        return self.model(x)

### Dueling network architecture proposed in https://arxiv.org/pdf/1511.06581.pdf
class Q_Atari_Duel(nn.Module):
    def __init__(self, in_channels=1, num_actions=18):
        super(Q_Atari_Duel, self).__init__()
        self.num_actions = num_actions
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.flatten = nn.Flatten()
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc5 = nn.Linear(512, num_actions)
        self.fc6 = nn.Linear(7 * 7 * 64, 512)
        self.fc7 = nn.Linear(512, 1)

    def forward(self, x):
        if len(x.shape)==3:
            x = x.unsqueeze(0)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        a = F.relu(self.fc4(self.flatten(x)))
        a = self.fc5(a)
        v = F.relu(self.fc6(self.flatten(x)))
        v = self.fc7(v)
        a_mean = torch.mean(a, dim=1, keepdim=True)
        return (a-a_mean+v)


def model_get(name, num_actions, duel=None):
    if name == 'Test':
        return lambda: Q_Test(num_actions=num_actions)
    elif name == 'Lunar':
        return lambda: Q_Lunar(num_actions=num_actions)
    elif name == 'Atari':
        if duel:
            return lambda: Q_Atari_Duel(num_actions=num_actions)
        else:
            return lambda: Q_Atari(num_actions=num_actions)
    else:
        raise Exception("Environments not supported")
    
### Used in SA-DQN regularizer
def logits_margin(logits, y):
    comp_logits = logits - torch.zeros_like(logits).scatter(1, torch.unsqueeze(y, 1), 1e10)
    sec_logits, _ = torch.max(comp_logits, dim=1)
    margin = sec_logits - torch.gather(logits, 1, torch.unsqueeze(y, 1)).squeeze(1)
    margin = margin.sum()
    return margin

class DQN_Agent(nn.Module):
    def __init__(self, q_func, learning_rate= 0.00025, 
                 doubleQ=False, update_freq=None,
                 max_grad_norm=10.0, robust=False, 
                 alpha_schedule=None, eps_schedule=None,
                 kappa=None, pgd_param=None, reg_solver=None,
                 input_shape=(4,84,84)):
        super(DQN_Agent, self).__init__()
        self.Q = q_func().to(Param.device).type(Param.dtype)
        self.target_Q = copy.deepcopy(self.Q)
        self.robust = robust
        # robust training mode
        if self.robust: 
            self.robust_Q = q_func().to(Param.device).type(Param.dtype)
            self.worst_Q  = q_func().to(Param.device).type(Param.dtype)
            self.target_worst_Q = copy.deepcopy(self.worst_Q)
            self.robust_q_optimizer = torch.optim.Adam(self.robust_Q.parameters(), lr=learning_rate)
            self.worst_q_optimizer = torch.optim.Adam(self.worst_Q.parameters(), lr=learning_rate)
        self.q_optimizer = torch.optim.Adam(self.Q.parameters(), lr=learning_rate)
        self.doubleQ = doubleQ
        self.update_freq = update_freq
        self.counter = 0
        self.max_grad_norm = max_grad_norm
        self.num_actions = self.Q.num_actions
        self.alpha_schedule = alpha_schedule ### alpha schedule in robust-Q training
        self.eps_schedule = eps_schedule ### epsilon schedule in robust-Q training
        self.kappa = kappa
        self.pgd_param = pgd_param 
        self.reg_solver = reg_solver
        
        #if self.reg_solver == "cov":
            #dummy_input = torch.empty_like(torch.randn((1,) + input_shape))
            #if self.robust:
                #self.robust_Q = BoundedModule(self.robust_Q, dummy_input, device=Param.device)
            #else:
                #self.Q = BoundedModule(self.Q, dummy_input, device=Param.device)
        
    ### Q learning update
    def update(self, obs_batch, act_batch, rew_batch,\
               next_obs_batch, not_done_mask, gamma, tau, 
               which_q='Q', iters=1, weights=None):
        curr_Q, optimizer = None, None ### the q network that we are updating
        if which_q=='Q':
            curr_Q, optimizer = self.Q, self.q_optimizer
            current_Q_values = curr_Q(obs_batch).gather(1, act_batch)
            if self.doubleQ:
                alpha = self.alpha_schedule.value(self.counter)
                indices = curr_Q(next_obs_batch).max(1)[-1].unsqueeze(1)
                next_max_q = self.target_Q(next_obs_batch)
                next_max_q = next_max_q.gather(1, indices)
            else:
                next_max_q = (self.target_Q(next_obs_batch).max(1)[0]).unsqueeze(1)
        elif which_q == 'worst_Q':
            curr_Q, optimizer = self.worst_Q, self.worst_q_optimizer
            current_Q_values = curr_Q(obs_batch).gather(1, act_batch)
            alpha = self.alpha_schedule.value(self.counter)
            upper_q, lower_q = network_bounds(self.robust_Q, next_obs_batch, self.eps_schedule.value(self.counter))
            next_worst_q = self.target_worst_Q(next_obs_batch)
            if self.doubleQ:
                worst_actions = worst_action_select(curr_Q(next_obs_batch), upper_q, lower_q, Param.device)
            else:
                worst_actions = worst_action_select(next_worst_q, upper_q, lower_q, Param.device)
            next_worst_q = next_worst_q.gather(1, worst_actions)

        elif which_q == 'robust_Q':
            ### Update Robust Q values
            curr_Q, optimizer = self.robust_Q, self.robust_q_optimizer
            current_Q_values = curr_Q(obs_batch).gather(1, act_batch)
            alpha = self.alpha_schedule.value(self.counter)
            if self.doubleQ:
                indices = ((1-alpha) * self.Q(next_obs_batch) + alpha * self.worst_Q(next_obs_batch)).max(1)[-1].unsqueeze(1)
                ### Compute the target values based on weighted sum of worst Q and target Q
                next_max_q_1 = self.target_Q(next_obs_batch)
                next_max_q_1 = next_max_q_1.gather(1, indices)
                next_max_q_2 = self.target_worst_Q(next_obs_batch)
                next_max_q_2 = next_max_q_2.gather(1, indices)
                next_max_q   = (1-alpha) * next_max_q_1 + alpha * next_max_q_2
            else:
                next_max_q = ((1-alpha) * self.target_Q(next_obs_batch).max(1)[0]+alpha * self.worst_Q(next_obs_batch).max (1)[0]).unsqueeze(1)
        else:
            raise Exception('Q Type Mismatch')

        if which_q == 'worst_Q':
            next_Q_values = not_done_mask * next_worst_q
        else:
            next_Q_values = not_done_mask * next_max_q

        if self.reg_solver == 'none' or which_q == 'worst_Q' or (self.robust and which_q == 'Q'):
            reg_loss = 0.
        elif self.reg_solver == "pgd":
            labels = indices.clone().squeeze(-1)
            hinge_c = self.pgd_param["hinge_c"]
            loss_fn = lambda adv_state: logits_margin(curr_Q(adv_state), labels)
            state = obs_batch
            adv_state = pgd(state, self.eps_schedule.value(self.counter), loss_fn, 
                            Param.device, pgd_steps=self.pgd_param["pgd_steps"], 
                            lr=self.pgd_param["lr"], rand_init=True)
            adv_margin = logits_margin(curr_Q.forward(adv_state), labels)
            ori_margin = logits_margin(curr_Q.forward(state), labels)
            reg_loss = torch.clamp(adv_margin, min=-hinge_c)
        else:
            hinge_c = self.pgd_param["hinge_c"]
            _, lb   = network_bounds(curr_Q, obs_batch, self.eps_schedule.value(self.counter))
            reg_loss, _ = torch.min(lb, dim=1)
            reg_loss = torch.clamp(reg_loss, max=hinge_c)
            reg_loss = - reg_loss

        target_Q_values = rew_batch + (gamma * next_Q_values) 
        assert(next_Q_values.shape==target_Q_values.shape)

        ### Compute td error
        ### If weights is not None (used in priortized replay),
        ### we compute a weighted loss
        if weights is None:
            loss = F.smooth_l1_loss(current_Q_values, target_Q_values)
        else:
            loss = F.smooth_l1_loss(current_Q_values, target_Q_values, reduce=False)*(weights.unsqueeze(1))
            priority = loss + 1e-5
            loss = torch.mean(loss)
        loss += self.kappa*reg_loss
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(curr_Q.parameters(), self.max_grad_norm)
        optimizer.step()
        
        ### Update the Target network using vanilla Q
        ### two methods: 1. soft update tau*\theta+(1-tau)\theta' if tau is not None
        ### 2. hard update: after a certain period, update the target network completely
        if which_q=='Q':
            if (self.update_freq=='none'):
                soft_update(self.target_Q, self.Q, tau)
            else:
                self.counter += 1
                if (self.counter%self.update_freq==0):
                    hard_update(self.target_Q, self.Q)
        if which_q == 'worst_Q':
            if (self.update_freq=='none'):
                soft_update(self.target_worst_Q, self.worst_Q, tau)
            else:
                self.counter += 1
                if (self.counter%self.update_freq==0):
                    hard_update(self.target_worst_Q, self.worst_Q)

        
        ### Return the new priority
        if weights is not None:
            return priority.detach()
    
    
    ### Update the target Q network
    def update_target(self):
        self.target_Q.load_state_dict(self.Q.state_dict())
    
    def select_epilson_greedy_action(self, obs, eps):
        sample = random.random()
        if sample > eps:
            obs = torch.from_numpy(obs).to(Param.device).type(Param.dtype)
            with torch.no_grad():
                if self.robust:
                    return int(self.robust_Q(obs).data.max(1)[1].cpu())
                else:
                    return int(self.Q(obs).data.max(1)[1].cpu())
        else:
            with torch.no_grad():
                return random.randrange(self.num_actions)
    
    
    ### Choose the best action, happened during test time
    def step(self,obs):
        with torch.no_grad():
            if (len(obs.shape) == 1):
                obs = torch.from_numpy(obs).to(Param.device).type(Param.dtype).unsqueeze(0)
            else:
                obs = torch.from_numpy(obs).to(Param.device).type(Param.dtype)
            if self.robust:
                return self.robust_Q(obs).data.max(1)[1].cpu()[0]
            else:   
                return self.Q(obs).data.max(1)[1].cpu()[0]
    
    ### Choose the best action, happened during test time
    ### Input obs here is a torch tensor of shape (batch_size, 4, 84 84)
    ### Output is a tensor of shape (n,1)
    def step_torch_batch(self,obs):
        if self.robust:
            return self.robust_Q(obs).data.max(1)[1].unsqueeze(1)
        else:
            return self.Q(obs).data.max(1)[1].unsqueeze(1)
    
    ### The only difference between this function and select_epilson_greedy_action
    ### above is that obs here is a torch tensor, not a numpy array
    def step_torch_epsilon_greedy(self, obs, eps):
        if (len(obs.shape)==1):
            obs = obs.unsqueeze(0)
        eps_mask = torch.rand(obs.shape[0])
        random_act = torch.randint(high=self.num_actions,size=(obs.shape[0],))
        if self.robust:
            return torch.where(eps_mask < eps, random_act, self.robust_Q(obs).data.max(1)[1].cpu()).unsqueeze(1)
        else:
            return torch.where(eps_mask < eps, random_act, self.Q(obs).data.max(1)[1].cpu()).unsqueeze(1)
    
    def save(self, log_dir=os.path.join(Param.model_dir,'dqn/'), exp_name='dqn'):
        torch.save(
            {
                'q_model': self.target_Q.state_dict(),
                'q_optim': self.q_optimizer.state_dict()
            }, os.path.join(log_dir, exp_name, 'q_checkpoint.pth'))
        if self.robust:
            torch.save(
                {
                    'q_model': self.target_worst_Q.state_dict(),
                    'q_optim': self.worst_q_optimizer.state_dict()
                }, os.path.join(log_dir, exp_name, 'worst_q_checkpoint.pth'))
            torch.save(
                {
                    'q_model': self.robust_Q.state_dict(),
                    'q_optim': self.robust_q_optimizer.state_dict()
                }, os.path.join(log_dir, exp_name, 'robust_q_checkpoint.pth'))
    
    def load(self, load_dir):
        state_dict = torch.load(load_dir, map_location=Param.device)
        mapping = {"Q.conv1.weight":"model.0.weight",
           "Q.conv1.bias": "model.0.bias",
           "Q.conv2.weight": "model.2.weight",
           "Q.conv2.bias":  "model.2.bias",
           "Q.conv3.weight": "model.4.weight",
           "Q.conv3.bias":   "model.4.bias",
           "Q.fc4.weight":  "model.7.weight",
           "Q.fc4.bias":    "model.7.bias",
           "Q.fc5.weight":   "model.9.weight",
           "Q.fc5.bias":     "model.9.bias"
          }
        new_state_dict = {}
        for key in state_dict.keys():
            if key in mapping.keys():
                new_state_dict[mapping[key]] = state_dict[key]
        self.Q.load_state_dict(new_state_dict)
        hard_update(self.target_Q, self.Q)

def pgd(obs, epsilon, loss_fn, device, pgd_steps, lr, rand_init=False):
    """
    Implementation of PGD attacks, we want to maximize the loss function
    """
    obs_adv = obs.clone().to(device)
    if rand_init:
        obs_adv += (2 * epsilon * torch.rand_like(obs).to(device) - epsilon)
    for i in range(pgd_steps):
        with torch.no_grad():
            _obs_adv = obs_adv.clone().requires_grad_(True)
        loss = loss_fn(_obs_adv)
        loss.backward(torch.ones_like(loss))
        with torch.no_grad():
            gradients = _obs_adv.grad.sign()
        obs_adv += gradients * lr
        obs_adv = torch.max(torch.min(obs_adv, obs + epsilon), obs - epsilon)

    return obs_adv

