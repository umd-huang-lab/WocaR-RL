# Models used by RADIAL-RL:
# Tuomas Oikarinen, Tsui-Wei Weng, and Luca Daniel. Robust deep reinforcement learning through adversarial loss. arXiv preprint arXiv:2008.01976, 2020.

from __future__ import division
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from VaR_DQN.utils.param import Param

class CnnDQN(nn.Module):
    def __init__(self, num_channels, action_space):
        super(CnnDQN, self).__init__()
        
        self.num_actions = action_space.n
        self.model = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(64*6*6, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )
        self.train()
        
    def forward(self, x):
        x = self.model(x)
        return x

    def act(self, state, epsilon):
        with torch.no_grad():
            if random.random() > epsilon:
                q_value = self.forward(state)
                #print(q_value)
                action  = torch.argmax(q_value, dim=1)[0]
            else:
                action = random.randrange(self.num_actions)
        return action
    
    def select_epilson_greedy_action(self, obs, eps):
        sample = random.random()
        if sample > eps:
            obs = torch.from_numpy(obs).to(Param.device).type(Param.dtype)
            if not self.atari:
                obs = obs.unsqueeze(0)
            with torch.no_grad():
                return np.array(self.model(obs).data.max(1)[1].cpu().unsqueeze(1))
        else:
            with torch.no_grad():
                return random.randrange(self.num_actions)
    
    def step_torch(self,obs):
        if (len(obs.shape)==1):
            obs = obs.unsqueeze(0)
        return self.model(obs).data.max(1)[1].cpu()[0]
    
    def step_torch_epsilon_greedy(self, obs, eps):
        if (len(obs.shape)==3):
            obs = obs.unsqueeze(0)
        eps_mask = torch.rand(obs.shape[0])
        random_act = torch.randint(high=self.num_actions,size=(obs.shape[0],))
        return torch.where(eps_mask < eps, random_act, self.model(obs).data.max(1)[1].cpu()).unsqueeze(1)
    
    
class A3Cff(torch.nn.Module):
    def __init__(self, num_inputs, action_space):
        super(A3Cff, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(num_inputs, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(64*6*6, 512),
            nn.ReLU(),
            nn.Linear(512, action_space.n + 1)
        )
        self.train()

    def forward(self, inputs):
        x = self.model(inputs)
        value = x[:, 0:1]
        actions = x[:, 1:]

        return value, actions