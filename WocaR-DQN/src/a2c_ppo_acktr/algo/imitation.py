import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque
from itertools import count
import numpy as np
import random
import pickle
from WocaR_DQN.utils.param import Param

def imitate_learning(imitate_steps, expert_attacker, victim, 
                     attacker, envs, attacker_args, cont,
                     seed, batch_size=64, update_freq = 1000,
                     radial=False):
    
    buffer = DataBuffer(buffer_size = int(5e+4), batch_size = batch_size, seed=seed)
    obs, count, step, clean_action = envs.reset(), 0, 0, None
    if type(obs)==np.ndarray:
        obs = torch.from_numpy(obs).to(Param.device).type(Param.dtype).unsqueeze(0)
    while True:
        with torch.no_grad():
            action = victim.step_torch_epsilon_greedy(obs, 0.01)
        step += 1
        if clean_action is not None:
            mask = (action!=clean_action).squeeze(-1)
            obs_attacked, action_attacked = old_obs[mask], action[mask]
            buffer.add(obs_attacked, action_attacked)
            count += obs_attacked.shape[0]
            ### Perform an imitation update
            if step % update_freq == 0 and count > batch_size:
                expert_dataset = buffer.sample()
                imitate_update(attacker, expert_dataset, cont=cont)
            if count > imitate_steps:
                break
        # Obser reward and next obs
        obs, reward, done, infos = envs.step(action)
        if type(obs)==np.ndarray:
            obs = torch.from_numpy(obs).to(Param.device).type(Param.dtype).unsqueeze(0)
        clean_action = victim.step_torch_epsilon_greedy(obs, 0.01)
        ## Attack obs (if any)
        old_obs = obs.clone()
        obs = expert_attacker.attack_torch(q_network=victim.Q if not radial else victim.model, 
                                               obs=obs, epsilon=attacker_args['epsilon'], 
                                                fgsm=attacker_args['fgsm'], 
                                                lr=attacker_args['attack_lr'], 
                                                pgd_steps=attacker_args['attack_steps'],
                                                rand_init=attacker_args['rand_init'], 
                                                momentum=attacker_args['momentum'])

def imitate_update(agent, expert_data, cont=False):
    expert_obs, expert_action = expert_data
    ce_loss = nn.CrossEntropyLoss()
    
    recurrent_hidden_state_size = agent.actor_critic.base.recurrent_hidden_state_size
    recurrent = torch.zeros(expert_action.shape[0], recurrent_hidden_state_size, device=Param.device)
    masks = torch.ones(expert_action.shape[0], 1, device=Param.device)
    action = agent.actor_critic.act_mean(expert_obs, recurrent, masks)
    if not cont:
        action = torch.cat((action, -torch.sum(action, dim=1, keepdim=True)), 1)
    loss = ce_loss(action, expert_action.squeeze(1))
    print("Current Cross Entropy Loss:{}".format(loss))
    agent.optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(agent.actor_critic.parameters(), agent.max_grad_norm)
    agent.optimizer.step()

    
class DataBuffer(nn.Module):

    def __init__(self, buffer_size, batch_size, seed):
        super(DataBuffer, self).__init__()
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action"])
        self.seed = random.seed(seed)
        
    def add(self, state, action):
        for i in range(state.shape[0]):
            e = self.experience(state[i], action[i])
            self.memory.append(e)
    
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
 
        states = torch.vstack([e.state.unsqueeze(0) for e in experiences if e is not None]).type(Param.dtype).to(Param.device)
        actions = torch.vstack([e.action for e in experiences if e is not None]).to(Param.device)
        return (states, actions)
    
    def save(self, save_dir):
        states = torch.vstack([e.state.unsqueeze(0) for e in self.memory if e is not None])
        actions = torch.vstack([e.action for e in self.memory if e is not None])
        with open(save_dir, 'wb') as handle:
            pickle.dump([states, actions], handle)
    
    def load(self, save_dir):
         with open(save_dir, 'rb') as handle:
            self.memory = pickle.load(handle)
        
    def __len__(self):
        return len(self.memory)