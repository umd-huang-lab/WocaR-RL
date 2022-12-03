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
from WocaR_DQN.attacker.attacker import common_fgsm, common_pgd, common_momentum_fgm, noisy_pgd
from WocaR_DQN.utils.dqn_core import *
from WocaR_DQN.utils.param import Param

from sa_wrappers import make_atari, wrap_deepmind, wrap_pytorch, make_atari_cart
from sa_utils import Logger, QNetwork, model_setup
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from torch.autograd import Variable
from torch.distributions.categorical import Categorical
from torch.distributions.kl import kl_divergence
from torch.distributions import Beta

COEFF = 1

def state2tensor(state, device):
    state_tensor = torch.from_numpy(np.ascontiguousarray(state)).unsqueeze(0).to(device).to(torch.float32)
    state_tensor /= 255
    return state_tensor

def reward2tensor(reward, device):
    reward_tensor = torch.Tensor([reward]).unsqueeze(0).to(device).to(torch.float32)
    return reward_tensor

def get_policy(victim, obs):
    return torch.distributions.categorical.Categorical(logits=victim(obs).squeeze()).probs.unsqueeze(0)

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

def huang(victim, obs, epsilon, fgsm, pgd_steps, pgd_lr, device):
    q = victim(obs)
    ce_loss = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        optimal_act = q.data.max(1)[1]
    def loss_fn(perturbed_obs):
        q = victim(perturbed_obs)
        # we want to maximize this loss, i.e. the dissimilarity between the perturbed policy and the best policy
        loss = ce_loss(q, optimal_act)
        return loss
    if fgsm:
        return common_fgsm(obs, epsilon, loss_fn, device)
    else:
        return common_pgd(obs, epsilon, loss_fn, device, pgd_steps, pgd_lr)

def momentum(victim, obs, epsilon, pgd_steps, pgd_lr, device):
    q = victim(obs)
    ce_loss = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        optimal_act = q.data.max(1)[1]
    def loss_fn(perturbed_obs):
        q = victim(perturbed_obs)
        # we want to maximize this loss, i.e. the dissimilarity between the perturbed policy and the best policy
        loss = ce_loss(q, optimal_act)
        return loss
    return common_momentum_fgm(obs, epsilon, loss_fn, device, pgd_steps)

def maxworst(victim, obs, epsilon, pgd_steps, pgd_lr, device):
    q = victim(obs)
    ce_loss = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        optimal_act = q.data.min(1)[1]
    def loss_fn(perturbed_obs):
        q = victim(perturbed_obs)
        # we want to maximize this loss, i.e. the dissimilarity between the perturbed policy and the best policy
        loss = - ce_loss(q, optimal_act)
        return loss
    return common_pgd(obs, epsilon, loss_fn, device, pgd_steps, pgd_lr, False)

def patt(victim, obs, epsilon, pgd_steps, pgd_lr, device):
    q = victim(obs)
    with torch.no_grad():
        optimal_q = q.data.max(1)[0]
        worst_act = q.data.min(1)[1]

    beta_dist = Beta(torch.FloatTensor([2]).to(device), torch.FloatTensor([2]).to(device))
    ce_loss = torch.nn.CrossEntropyLoss()

    obs_var = obs.clone().detach().to(device).requires_grad_(True)
    loss = ce_loss(victim(obs_var), worst_act)
    loss.backward()

    grad_dir = obs_var.grad/torch.norm(obs_var.grad)

    with torch.no_grad():
        s_adv = obs.clone()
        for i in range(pgd_steps):
            noise_factor = beta_dist.sample()
            s = obs - noise_factor * grad_dir
            s = torch.max(torch.min(s, obs + epsilon), obs - epsilon)
            # print("noise", noise_factor)
            new_q, new_act = victim(s).data.max(1)
            # print("new", new_q, new_act)
            update_idx = new_q < optimal_q
            # print("update", update_idx)
            s_adv[update_idx] = s[update_idx].clone()
    # print("final", s_adv)
    return s_adv.detach()

def kl(victim, obs, epsilon, pgd_steps, pgd_lr, device):
    init = 2 * epsilon * torch.rand_like(obs) - epsilon * torch.ones_like(obs)
    perturb = Variable(init.to(device), requires_grad=True)
    ### Compute pi(a|s), and pi(a|s')
    with torch.no_grad():
        q_original = victim(obs)
        original_dist  = Categorical(logits=q_original.squeeze())

    def loss_fn(perturbed_obs):
        q_perturbed = victim(perturbed_obs)
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
#     print("want to perturb to action", direction)
#     if direction == old_action:
#         loss = - ce_loss(victim(obs+perturb), direction)
#     else:
    loss = ce_loss(victim(obs+perturb), direction)
#     print("before loss", loss)
    loss.backward()
    grad = perturb.grad.data
    perturb.data -= epsilon * torch.sign(grad)
    
#     loss = ce_loss(victim(obs+perturb), direction)
#     print("after loss", loss)
    
    return perturb.detach() + obs.detach()

def obs_dir_perturb_pgd(victim, obs, direction, epsilon, device, pgd_steps, pgd_lr, cont=False):
    """
    Targeted attack: find the best obs attack in order to perturb the policy 
    to a target direction as much as possible
    """
    init = torch.zeros_like(obs).to(device)
    perturb = Variable(init, requires_grad=True)

    ce_loss = torch.nn.CrossEntropyLoss()
    
    def loss_fn(perturbed_obs):
        perturbed_policy = victim(perturbed_obs)
        loss = - ce_loss(perturbed_policy, direction)
        return loss
    
    return common_pgd(obs, epsilon, loss_fn, device, pgd_steps, pgd_lr, rand_init=False) 

def obs_dir_perturb_momentum(victim, obs, direction, epsilon, pgd_steps, pgd_lr, device):
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

def main():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


    torch.set_num_threads(1)
    device = torch.device("cuda:{}".format(args.cuda_id) if args.cuda else "cpu")


    config_name = args.env_name[:args.env_name.find("NoFrameskip")]
    with open("released_models/sa_models/configs/" + config_name+".json") as f:
        config = json.load(f)
    
    if args.attack_lr == 0.01:
        args.attack_lr = args.epsilon / 10
    print("attack steps", args.attack_steps, "attack lr", args.attack_lr)
        
    envs = make_atari(args.env_name)
    envs = wrap_deepmind(envs, clip_rewards=False, episode_life=False, central_crop=True, 
                         restrict_actions=config['restrict_actions'], crop_shift=config['crop_shift'])
    envs = wrap_pytorch(envs)
    print("The observation space is", envs.observation_space)
    
    action_space = Discrete(envs.action_space.n) #Box(-1.0, 1.0, (envs.action_space.n-1,))
    print("The action space is", action_space)
    envs.seed(args.seed)
    
    logger = Logger(open("data/log/log_sa_{}.txt".format(args.env_name), "w"))
    
    model = model_setup(args.env_name, envs, robust_model=True, logger=logger, use_cuda=True, 
                        dueling=True, model_width=1, device=device)
    model.features.load_state_dict(torch.load("released_models/sa_models/"+config_name+"-convex.model"))
    print("loaded sa model")
    
    if args.attacker == "paad":
        global attacker_agent
        attacker_agent = Policy(
            envs.observation_space.shape,
            action_space,
            base_kwargs={'recurrent': args.recurrent_policy})
        attack_model = "released_models/robust_attacker/sa/{}_{}_pgd_sa_attacker".format(args.env_name, np.round(args.epsilon*255))
        attacker_agent.load_state_dict(torch.load(attack_model, map_location=device))
        attacker_agent = attacker_agent.to(device)
        print("loading pa attacker from", attack_model)
        mask = torch.ones(args.num_processes, 1, device=device)
        recurrent = torch.zeros(args.num_processes, attacker_agent.recurrent_hidden_state_size, device=device)

    obs = envs.reset()
    obs = state2tensor(obs, device)
    total_episode_rewards = []
    episode_rewards = deque(maxlen=10)
    total_fool = deque(maxlen=10)
    fools = 0
    
    start = time.time()
    
    rewards = 0
    num_episodes = 0
    total_steps = 0
    epi_steps = 0
    
    while num_episodes < args.test_episodes:
        total_steps += 1
        epi_steps += 1
        old_action = model.act(obs)[0]
        old_obs = obs.clone()

        if args.attacker == "random":
            obs = random(obs, args.epsilon, device)
        elif args.attacker == "minbest":
            obs = huang(model, obs, args.epsilon, args.fgsm, args.attack_steps, args.attack_lr, device)
        elif args.attacker == "momentum":
            obs = momentum(model, obs, args.epsilon, args.attack_steps, args.attack_lr, device)
        elif args.attacker == "minq":
            obs = patt(model, obs, args.epsilon, args.attack_steps, args.attack_lr, device)
        elif args.attacker == "maxdiff":
            obs = kl(model, obs, args.epsilon, args.attack_steps, args.attack_lr, device)
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

        attacked_action = model.act(obs)[0]
        if old_action != attacked_action:
            fools += 1

        obs, reward, done, info = envs.step(attacked_action)
        obs = state2tensor(obs, device)

        rewards += reward

        if done:
            episode_rewards.append(rewards)
            print("num_episodes", num_episodes, "rewards", rewards)
            total_episode_rewards.append(rewards)
            total_fool.append(fools)
            rewards = 0
            epi_steps = 0
            fools = 0
            num_episodes += 1
            obs = envs.reset()
            obs = state2tensor(obs, device)

        if total_steps % args.log_interval == 0 and len(episode_rewards) > 1:
            end = time.time()
            print(
                "num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, mean fool {:.1f} \n"
                .format(total_steps,
                        int(total_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), np.mean(total_fool)))
            


    total_episode_rewards = np.array(total_episode_rewards)
    print("mean reward", total_episode_rewards.mean().round(2), "+-",
          total_episode_rewards.std().round(2))


if __name__ == "__main__":
    main()