import copy
import glob
import os
import time
import sys
from collections import deque
import json

import gym
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from WocaR_DQN.a2c_ppo_acktr import algo, utils
from WocaR_DQN.a2c_ppo_acktr.algo import gail
from WocaR_DQN.a2c_ppo_acktr.arguments import get_args
from WocaR_DQN.a2c_ppo_acktr.envs import make_vec_envs
from WocaR_DQN.a2c_ppo_acktr.model import Policy
from WocaR_DQN.a2c_ppo_acktr.storage import RolloutStorage
# from evaluation import evaluate
from WocaR_DQN.attacker.attacker import common_fgsm, common_pgd, common_momentum_fgm, noisy_pgd
from WocaR_DQN.utils.dqn_core import *
from WocaR_DQN.utils.param import Param
from WocaR_DQN.a2c_ppo_acktr.algo.kfac import KFACOptimizer

from radial_utils import CnnDQN, A3Cff
from radial_wrapper import atari_env
# from sa_utils import Logger

from gym.spaces.box import Box
from torch.autograd import Variable
from torch.distributions.categorical import Categorical
from torch.distributions.kl import kl_divergence
from torch.distributions import Beta
COEFF = 1

def state2tensor(state, device):
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    return state_tensor

def reward2tensor(reward, device):
    reward_tensor = torch.Tensor([reward]).unsqueeze(0).to(device).to(torch.float32)
    return reward_tensor

def get_action(victim, obs):
    output = victim.forward(obs)
    if victim_type == "a3c":
        output = output[1]
    action = torch.argmax(output, dim=1)
    return action.detach()

def get_q(victim, obs):
    output = victim.forward(obs)
    if victim_type == "a3c":
        output = output[1]
    return output

def get_policy(victim, obs):
    if victim_type == "dqn":
        return torch.distributions.categorical.Categorical(logits=victim(obs).squeeze()).probs.unsqueeze(0)
    elif victim_type == "a3c":
        return torch.distributions.categorical.Categorical(logits=victim(obs)[1].squeeze()).probs.unsqueeze(0)

def dqn(victim, attacker, obs, epsilon, device):
    q = victim(obs)
    ce_loss = torch.nn.CrossEntropyLoss()
    perturb_a = attacker.Q(obs).data.max(1)[1]
    def loss_fn(perturbed_obs):
        q = victim(perturbed_obs)
        # we want to maximize this loss, i.e. the dissimilarity between the perturbed policy and the best policy
        loss = - ce_loss(q, perturb_a)
        return loss
    return common_fgsm(obs, epsilon, loss_fn, device)

def random(obs, epsilon, device):
    perturb = (2 * epsilon * torch.rand_like(obs) - epsilon * torch.ones_like(obs)).to(device)
    perturb = epsilon * torch.sign(perturb)
    return (perturb+obs).detach()

def huang(victim, obs, epsilon, pgd_steps, pgd_lr, fgsm, device):
    q = get_q(victim, obs)
    ce_loss = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        optimal_act = q.data.max(1)[1]
    def loss_fn(perturbed_obs):
        q = get_q(victim, perturbed_obs)
        # we want to maximize this loss, i.e. the dissimilarity between the perturbed policy and the best policy
        loss = ce_loss(q, optimal_act)
        return loss
    if fgsm:
        return common_fgsm(obs, epsilon, loss_fn, device)
    else:
        return common_pgd(obs, epsilon, loss_fn, device, pgd_steps, pgd_lr)

def momentum(victim, obs, epsilon, device, pgd_steps, pgd_lr):
    q = get_q(victim, obs)
    ce_loss = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        optimal_act = q.data.max(1)[1]
    def loss_fn(perturbed_obs):
        q = get_q(victim, perturbed_obs)
        # we want to maximize this loss, i.e. the dissimilarity between the perturbed policy and the best policy
        loss = ce_loss(q, optimal_act)
        return loss
    return common_momentum_fgm(obs, epsilon, loss_fn, device, pgd_steps)

def patt(victim, obs, epsilon, device, pgd_steps, pgd_lr):
    q = get_q(victim, obs)
    with torch.no_grad():
        optimal_q = q.data.max(1)[0]
        worst_act = q.data.min(1)[1]

    beta_dist = Beta(torch.FloatTensor([2]).to(device), torch.FloatTensor([2]).to(device))
    ce_loss = torch.nn.CrossEntropyLoss()

    obs_var = obs.clone().detach().to(device).requires_grad_(True)
    loss = ce_loss(get_q(victim, obs_var), worst_act)
    loss.backward()

    grad_dir = obs_var.grad/torch.norm(obs_var.grad)

    with torch.no_grad():
        s_adv = obs.clone()
        for i in range(pgd_steps):
            noise_factor = beta_dist.sample()
            s = obs - noise_factor * grad_dir
            s = torch.max(torch.min(s, obs + epsilon), obs - epsilon)
            # print("noise", noise_factor)
            new_q, new_act = get_q(victim, s).data.max(1)
            # print("new", new_q, new_act)
            update_idx = new_q < optimal_q
            # print("update", update_idx)
            s_adv[update_idx] = s[update_idx].clone()
    # print("final", s_adv)
    return s_adv.detach()

def kl(victim, obs, epsilon, device, pgd_steps, pgd_lr):
    init = 2 * epsilon * torch.rand_like(obs) - epsilon * torch.ones_like(obs)
    perturb = Variable(init.to(device), requires_grad=True)
    ### Compute pi(a|s), and pi(a|s')
    with torch.no_grad():
        q_original = get_q(victim, obs)
        original_dist  = Categorical(logits=q_original.squeeze())

    def loss_fn(perturbed_obs):
        q_perturbed = get_q(victim, perturbed_obs)
        perturbed_dist = Categorical(logits=q_perturbed.squeeze())
        # we want to maximize the distance between the original policy and the pertorbed policy
        loss = kl_divergence(original_dist, perturbed_dist)
        return loss
    return noisy_pgd(obs, epsilon, loss_fn, device, pgd_steps, pgd_lr, True)
    
def obs_dir_perturb_fgsm(victim, obs, direction, epsilon, device, cont=False):
    """
    Targeted attack: find the best obs attack in order to perturb the policy 
    to a target direction as much as possible
    """
    init = torch.zeros_like(obs).to(device)
    perturb = Variable(init, requires_grad=True)

    ce_loss = torch.nn.CrossEntropyLoss()

    perturbed_policy = victim(obs+perturb)
    if victim_type == "a3c":
        perturbed_policy = perturbed_policy[1]

    loss = ce_loss(perturbed_policy, direction)
#     print("before loss", loss)
    loss.backward()
    grad = perturb.grad.data
    perturb.data -= epsilon * torch.sign(grad)
    
    return (obs+perturb).detach()

def obs_dir_perturb_pgd(victim, obs, direction, epsilon, device, pgd_steps, pgd_lr):
    """
    Targeted attack: find the best obs attack in order to perturb the policy 
    to a target direction as much as possible
    """
    init = torch.zeros_like(obs).to(device)
    perturb = Variable(init, requires_grad=True)

    loss_func = torch.nn.CrossEntropyLoss()
#     loss_func = torch.nn.MultiMarginLoss()
    
    def loss_fn(perturbed_obs):
        perturbed_policy = victim(perturbed_obs)
        if victim_type == "a3c":
            perturbed_policy = perturbed_policy[1]

        loss = - loss_func(perturbed_policy, direction)
        return loss
    
    return common_pgd(obs, epsilon, loss_fn, device, pgd_steps, lr=pgd_lr, rand_init=False)

def obs_dir_perturb_momentum(victim, obs, direction, epsilon, device, pgd_steps, pgd_lr):
    ce_loss = torch.nn.CrossEntropyLoss()
        
    def loss_fn(perturbed_obs):
        loss = ce_loss(victim(perturbed_obs), direction)
        return loss
    
    mu = 0.5
    v = torch.zeros_like(obs).to(device)
    lr = pgd_lr

    obs_adv = obs.clone().detach().to(device)
    for i in range(pgd_steps):
        _obs_adv = obs_adv.clone().detach().requires_grad_(True)
        loss = loss_fn(_obs_adv + mu * v)
        loss.backward(torch.ones_like(loss))
        gradients = _obs_adv.grad

        v = mu * v + gradients/torch.norm(gradients, p=1)
        obs_adv -= v.sign().detach() * lr
        # print(obs_adv)
        obs_adv = torch.max(torch.min(obs_adv, obs + epsilon), obs - epsilon)
#         print("i", i, "adv", obs_adv[0]-obs[0])
        
    return obs_adv.detach()

def eval_clean(curr_model, env, device):
    episode_reward = 0
    state = env.reset()

    with torch.no_grad():
        while True:
            input_x = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = get_action(curr_model, input_x)
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            state = next_state
            if done and not info:
                state = env.reset()
            elif info:
                state = env.reset()
                print('Reward:{}'.format(episode_reward))
                return episode_reward

def eval_attack(model, env, args, device):
    episode_reward = 0
    state = env.reset()
    fools = 0
    steps = 0
    if args.attacker == "paad":
        mask = torch.ones(1, 1, device=device)
        recurrent = torch.zeros(1, attacker_agent.recurrent_hidden_state_size, device=device)
    while True:
        steps += 1
        obs = torch.FloatTensor(state).unsqueeze(0).to(device)
        old_obs = obs.clone()
        output = model.forward(obs)
        old_action = get_action(model, obs)
        # perturb
        if args.attacker == "random":
            obs = random(obs, args.epsilon, device=device)
        elif args.attacker == "minbest":
            obs = huang(model, obs, args.epsilon, pgd_steps=args.attack_steps, pgd_lr=args.attack_lr, fgsm=args.fgsm, device=device)
        elif args.attacker == "momentum":
            obs = momentum(model, obs, args.epsilon, pgd_steps=args.attack_steps, pgd_lr=args.attack_lr, device=device)
        elif args.attacker == "minq":
            obs = patt(model, obs, args.epsilon, pgd_steps=args.attack_steps, pgd_lr=args.attack_lr, device=device)
        elif args.attacker == "maxdiff":
            obs = kl(model, obs, args.epsilon, pgd_steps=args.attack_steps, pgd_lr=args.attack_lr, device=device)
        elif args.attacker == "paad":
            with torch.no_grad():
                _, action, _, recurrent = attacker_agent.act(
                    obs, recurrent, mask, deterministic=args.det)
            perturb_direction = action[0]
            if args.momentum:
                obs = obs_dir_perturb_momentum(model, obs, perturb_direction, args.epsilon, pgd_steps=args.attack_steps, pgd_lr=args.attack_lr, device=device)
            elif args.fgsm:
                obs = obs_dir_perturb_fgsm(model, obs, perturb_direction, args.epsilon, device=device)
            else:
                obs = obs_dir_perturb_pgd(model, obs, perturb_direction, args.epsilon, pgd_steps=args.attack_steps, pgd_lr=args.attack_lr, device=device)

        action = get_action(model, obs)
        if old_action != action:
            fools += 1
        next_state, reward, done, info = env.step(action)
        episode_reward += reward
        state = next_state
#         mask = torch.FloatTensor([[0.0] if done else [1.0]])
        if done and not info:
            state = env.reset()
        elif info:
            state = env.reset()
            print('Reward: {}'.format(episode_reward))
            print('Fools: {}/{}, fooling rate: {}'.format(fools, steps, fools/steps))
            return episode_reward

def main():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

#     torch.set_num_threads(1)
    device = torch.device("cuda:{}".format(args.cuda_id) if args.cuda else "cpu")

    with open("released_models/radial_models/env_config.json") as f:
        config = json.load(f)
    env_conf = config['Default']
    for i in config.keys():
        if i in args.env_name:
            env_conf = config[i]
    print("env config", env_conf)
    envs = atari_env(args.env_name, env_conf, 10000, 4)
    print("The observation space is", envs.observation_space)
    envs.seed(args.seed) # use the same seed for the environment to have a fair comparison
    
    action_space = envs.action_space # Box(-1.0, 1.0, (envs.action_space.n-1,))
    print("The action space is", action_space)
            
    # load sa model
    global victim_type
    if args.v_algo == "a3c":
        victim_type = "a3c"
    else:
        victim_type = "dqn"
    
    if victim_type == "dqn":
        model = CnnDQN(envs.observation_space.shape[0], envs.action_space)

    elif victim_type == "a3c":
        model = A3Cff(envs.observation_space.shape[0], envs.action_space)
    
    weights = torch.load("released_models/radial_models/{}/{}_robust.pt".format(victim_type, args.env_name), map_location=device)
    model.load_state_dict(weights)
    model.to(device)
    model.eval()
    
    print("loaded radial model")
    
    if args.attack_lr == 0.01:
        args.attack_lr = args.epsilon / 10
    print("attack steps", args.attack_steps, "attack lr", args.attack_lr)
    
    if args.attacker == "paad":
        global attacker_agent
        attacker_agent = Policy(
            envs.observation_space.shape,
            action_space,
            beta=False,
            epsilon=args.epsilon,
            base_kwargs={'recurrent': False, 'dim': 80})
        attack_model = "released_models/robust_attacker/radial/{}_{}_pgd_{}_attacker".format(args.env_name, np.round(args.epsilon*255), victim_type)
        try:
            attacker_agent.load_state_dict(torch.load(attack_model, map_location=device))
        except:
            KFACOptimizer(attacker_agent)
            attacker_agent.load_state_dict(torch.load(attack_model, map_location=device))
        attacker_agent = attacker_agent.to(device)
        print("loading pa attacker from", attack_model)
        mask = torch.ones(1, 1, device=device)
        recurrent = torch.zeros(1, attacker_agent.recurrent_hidden_state_size, device=device)
        
        
    obs = envs.reset()
    obs = state2tensor(obs, device)
    episode_rewards = deque(maxlen=10)
    total_episode_rewards = []
    total_fool = deque(maxlen=10)
    fools = 0

    start = time.time()
    
    rewards = 0
    num_episodes = 0
    total_steps = 0
    epi_steps = 0
    
    if args.attacker is None:
        for i in range(args.test_episodes):
            reward = eval_clean(model, envs, device)
            total_episode_rewards.append(reward)
        print("mean reward", np.array(total_episode_rewards).mean().round(2), "+-",
          np.array(total_episode_rewards).std().round(2))
    else:
        for i in range(args.test_episodes):
            reward = eval_attack(model, envs, args, device)
            total_episode_rewards.append(reward)
        print("mean reward", np.array(total_episode_rewards).mean().round(2), "+-",
          np.array(total_episode_rewards).std().round(2))
    

if __name__ == "__main__":
    main()