import torch
import torch.nn as nn
import torch.autograd as autograd
import random
import numpy as np
import sys
import torch.nn.functional as F
sys.path.append("./auto_LiRPA")
from auto_LiRPA import BoundedModule
import math


USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
    
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

class QNetwork(nn.Module):
    def __init__(self, name, env, input_shape, num_actions, robust=False, width=1):
        super(QNetwork, self).__init__()
        self.env = env
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.robust = robust
        self.name = name
        if name == 'DQN':
            self.features = nn.Sequential(
                nn.Linear(input_shape[0], 128*width),
                nn.ReLU(),
                nn.Linear(128*width, 128*width),
                nn.ReLU(),
                nn.Linear(128*width, self.env.action_space.n)
            )
        elif name == 'CnnDQN':
            self.features = nn.Sequential(
                nn.Conv2d(input_shape[0], 32*width, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32*width, 64*width, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64*width, 64*width, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(3136*width, 512*width),
                nn.ReLU(),
                nn.Linear(512*width, self.num_actions)
            )
        elif name == 'DuelingCnnDQN':
            self.features = DuelingCnnDQN(input_shape, num_actions, width)
        else:
            raise NotImplementedError('{} network structure not implemented.'.format(name))
        
        if self.robust:
            dummy_input = torch.empty_like(torch.randn((1,) + input_shape))
            self.features = BoundedModule(self.features, dummy_input, device="cuda" if USE_CUDA else "cpu")
        
    def forward(self, *args, **kwargs):
        return self.features(*args, **kwargs)

    def act(self, state, epsilon=0):
        #state   = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0))
        if self.robust:
            q_value = self.forward(state, method_opt='forward')
        else:
            q_value = self.forward(state)
        action  = q_value.max(1)[1].data.cpu().numpy()
        mask = np.random.choice(np.arange(0, 2), p=[1-epsilon, epsilon])
        action = (1-mask) * action + mask * np.random.randint(self.env.action_space.n, size=state.size()[0])
        return action
    
    def bound_forward(self, x, eps):
        upper = x + eps
        lower = x - eps
        if self.name == 'DuelingCnnDQN':
            for layer in self.features.cnn.modules():
                if type(layer) in (nn.Sequential,):
                    pass
                elif type(layer) in (nn.ReLU, nn.Flatten):
                    upper, lower = activation_bound(layer, upper, lower)
                elif type(layer) in (nn.Conv2d, nn.Linear):
                    upper, lower = weighted_bound(layer, upper, lower)
            cnn_ub, cnn_lb = upper, lower
            ub, lb = cnn_ub, cnn_lb
            for layer in self.features.advantage.modules():
                if type(layer) in (nn.Sequential,):
                    pass
                elif type(layer) in (nn.ReLU, nn.Flatten):
                    ub, lb = activation_bound(layer, ub, lb)
                elif type(layer) in (nn.Conv2d, nn.Linear):
                    ub, lb = weighted_bound(layer, ub, lb)
            adv_ub, adv_lb = ub, lb
            ub, lb = cnn_ub, cnn_lb
            for layer in self.features.value.modules():
                if type(layer) in (nn.Sequential,):
                    pass
                elif type(layer) in (nn.ReLU, nn.Flatten):
                    ub, lb = activation_bound(layer, ub, lb)
                elif type(layer) in (nn.Conv2d, nn.Linear):
                    ub, lb = weighted_bound(layer, ub, lb)
            value_ub, value_lb = ub, lb
            logits_ub = value_ub + adv_ub - torch.sum(adv_ub, dim=1, keepdim=True) / self.num_actions
            logits_lb = value_lb + adv_lb - torch.sum(adv_lb, dim=1, keepdim=True) / self.num_actions
            return logits_ub, logits_lb
        elif self.name == 'CnnDQN':
            for layer in self.features.modules():
                if type(layer) in (nn.Sequential,):
                    pass
                elif type(layer) in (nn.ReLU, nn.Sigmoid, nn.Tanh, nn.MaxPool2d, nn.Flatten):
                    upper, lower = activation_bound(layer, upper, lower)
                elif type(layer) in (nn.Conv2d, nn.Linear):
                    upper, lower = weighted_bound(layer, upper, lower)
            return upper, lower


class DuelingCnnDQN(nn.Module):
    def __init__(self, input_shape, num_actions, width=1):
        super(DuelingCnnDQN, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.cnn = nn.Sequential(
            nn.Conv2d(input_shape[0], 32*width, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32*width, 64*width, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64*width, 64*width, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.advantage = nn.Sequential(
            nn.Linear(3136*width, 512*width),
            nn.ReLU(),
            nn.Linear(512*width, self.num_actions)
        )

        self.value = nn.Sequential(
            nn.Linear(3136*width, 512*width),
            nn.ReLU(),
            nn.Linear(512*width, 1)
        )

    def forward(self, x):
        cnn = self.cnn(x)
        advantage = self.advantage(cnn)
        value = self.value(cnn)
        return value + advantage - torch.sum(advantage, dim=1, keepdim=True) / self.num_actions


class CnnDQN(nn.Module):
    def __init__(self, input_shape, num_actions, robust=False):
        super(CnnDQN, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.robust = robust
        print("num_actions", self.num_actions)
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(64*7*7, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )
        self.train()
        
    def forward(self, x):
        x = self.features(x)
        return x

    def act(self, state, epsilon=0):
        with torch.no_grad():
            if random.random() > epsilon:
                q_value = self.forward(state)
                action  = torch.argmax(q_value, dim=1)[0]
            else:
                action = random.randrange(self.num_actions)
        return action
    
    def bound_forward(self, x, eps):
        upper = x + eps
        lower = x - eps
        if self.name == 'DuelingCnnDQN':
            for layer in self.features.cnn.modules():
                if type(layer) in (nn.Sequential,):
                    pass
                elif type(layer) in (nn.ReLU, nn.Flatten):
                    upper, lower = activation_bound(layer, upper, lower)
                elif type(layer) in (nn.Conv2d, nn.Linear):
                    upper, lower = weighted_bound(layer, upper, lower)
            cnn_ub, cnn_lb = upper, lower
            ub, lb = cnn_ub, cnn_lb
            for layer in self.features.advantage.modules():
                if type(layer) in (nn.Sequential,):
                    pass
                elif type(layer) in (nn.ReLU, nn.Flatten):
                    ub, lb = activation_bound(layer, ub, lb)
                elif type(layer) in (nn.Conv2d, nn.Linear):
                    ub, lb = weighted_bound(layer, ub, lb)
            adv_ub, adv_lb = ub, lb
            ub, lb = cnn_ub, cnn_lb
            for layer in self.features.value.modules():
                if type(layer) in (nn.Sequential,):
                    pass
                elif type(layer) in (nn.ReLU, nn.Flatten):
                    ub, lb = activation_bound(layer, ub, lb)
                elif type(layer) in (nn.Conv2d, nn.Linear):
                    ub, lb = weighted_bound(layer, ub, lb)
            value_ub, value_lb = ub, lb
            logits_ub = value_ub + adv_ub - torch.sum(adv_ub, dim=1, keepdim=True) / self.num_actions
            logits_lb = value_lb + adv_lb - torch.sum(adv_lb, dim=1, keepdim=True) / self.num_actions
            return logits_ub, logits_lb
        elif self.name == 'CnnDQN':
            for layer in self.features.modules():
                if type(layer) in (nn.Sequential,):
                    pass
                elif type(layer) in (nn.ReLU, nn.Sigmoid, nn.Tanh, nn.MaxPool2d, nn.Flatten):
                    upper, lower = activation_bound(layer, upper, lower)
                elif type(layer) in (nn.Conv2d, nn.Linear):
                    upper, lower = weighted_bound(layer, upper, lower)
            return upper, lower
    

def model_setup(env_id, env, robust_model, logger, use_cuda, dueling=True, model_width=1):
    if "NoFrameskip" not in env_id:
        net_name = 'DQN'
    else:
        if not dueling:
            net_name = 'CnnDQN'
        else:
            net_name = 'DuelingCnnDQN'
    # model = CnnDQN(env.observation_space.shape, env.action_space.n)
    model = QNetwork(net_name, env, env.observation_space.shape, env.action_space.n, robust_model, model_width)
    if use_cuda:
        model = model.cuda()
    return model
