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
from WocaR_DQN.utils.param import Param
# from evaluation import evaluate
from WocaR_DQN.attacker.attacker import common_fgsm, common_pgd, common_momentum_fgm, noisy_pgd

from radial_utils import CnnDQN, A3Cff
from radial_wrapper import atari_env
from sa_utils import Logger
from gym.spaces.box import Box
from torch.autograd import Variable

COEFF = 1

def state2tensor(state, device):
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    return state_tensor

def reward2tensor(reward, device):
    reward_tensor = torch.Tensor([reward]).unsqueeze(0).to(device).to(torch.float32)
    return reward_tensor

def get_action(victim, obs):
#     if torch.isnan(obs).any() or torch.isinf(obs).any():
#         print("has nan", obs)
    output = victim.forward(obs)
    if victim_type == "a3c":
        output = output[1]
    action = torch.argmax(output, dim=1)
    return action.detach()

def get_policy(victim, obs):
    if victim_type == "dqn":
        return torch.distributions.categorical.Categorical(logits=victim(obs).squeeze()).probs.unsqueeze(0)
    elif victim_type == "a3c":
        return torch.distributions.categorical.Categorical(logits=victim(obs)[1].squeeze()).probs.unsqueeze(0)

def obs_dir_perturb_fgsm(victim, obs, direction, epsilon, device, cont=False):
    """
    Targeted attack: find the best obs attack in order to perturb the policy 
    to a target direction as much as possible
    """
    init = torch.zeros_like(obs).to(device)
    perturb = Variable(init, requires_grad=True)

    ce_loss = torch.nn.CrossEntropyLoss()
#     print("want to perturb to action", direction)
#     print("old action", old_action)

    perturbed_policy = victim(obs+perturb)
    if victim_type == "a3c":
        perturbed_policy = perturbed_policy[1]
#     if direction == old_action:
#         loss = - ce_loss(perturbed_policy, direction)
#     else:
    loss = ce_loss(perturbed_policy, direction)
#     print("before loss", loss)
    loss.backward()
    grad = perturb.grad.data
    perturb.data -= epsilon * torch.sign(grad)
    
#     loss = ce_loss(victim(obs+perturb), direction)
#     print("after loss", loss)
    
    return perturb.detach()

def obs_dir_perturb_pgd(victim, obs, direction, epsilon, pgd_steps, pgd_lr, device, cont=False):
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
    
    return common_pgd(obs, epsilon, loss_fn, device, pgd_steps, lr=pgd_lr, rand_init=False) - obs

def obs_dir_perturb_momentum(victim, obs, direction, epsilon, device, cont=False, maxiter=10):
    ce_loss = torch.nn.CrossEntropyLoss()
        
    def loss_fn(perturbed_obs):
        loss = ce_loss(victim(perturbed_obs), direction)
        return loss
    
    mu = 0.5
    v = torch.zeros_like(obs).to(device)
    lr = epsilon / maxiter

    obs_adv = obs.clone().detach().to(device)
    for i in range(maxiter):
        _obs_adv = obs_adv.clone().detach().requires_grad_(True)
        loss = loss_fn(_obs_adv + mu * v)
        loss.backward(torch.ones_like(loss))
        gradients = _obs_adv.grad

        v = mu * v + gradients/torch.norm(gradients, p=1)
        obs_adv -= v.sign().detach() * lr
        # print(obs_adv)
        obs_adv = torch.max(torch.min(obs_adv, obs + epsilon), obs - epsilon)
#         print("i", i, "adv", obs_adv[0]-obs[0])
        
    return obs_adv.detach() - obs

def main():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    
    torch.set_num_threads(1)
    device = torch.device("cuda:{}".format(args.cuda_id) if args.cuda else "cpu")
    Param(torch.cuda.FloatTensor, device)
    
    with open("released_models/radial_models/env_config.json") as f:
        config = json.load(f)
    env_conf = config['Default']
    for i in config.keys():
        if i in args.env_name:
            env_conf = config[i]
    print("env config", env_conf)
    envs = atari_env(args.env_name, env_conf, 10000, 4)
    print("The observation space is", envs.observation_space)
    env_maker = lambda: atari_env(args.env_name, env_conf, 10000, 4)
    
    action_space = envs.action_space # Box(-1.0, 1.0, (envs.action_space.n-1,))
    print("The action space is", action_space)
    cont = False
    
    args.num_processes = 1
    
    exp_name = args.env_name + "_" + str(np.round(args.epsilon*255))
    if args.fgsm:
        exp_name += "_fgsm"
    elif args.momentum:
        exp_name += "_momentum"
    else:
        exp_name += "_pgd"
        
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
    
    save_path = "./learned_adv/{}/".format(args.algo)
    model_save_path =  os.path.join(save_path, exp_name + "_" + victim_type + "_attacker")
    
    actor_critic = Policy(
        envs.observation_space.shape,
        action_space,
        base_kwargs={'recurrent': args.recurrent_policy, 'dim': 80})
    if args.load:
        print("load model from", model_save_path)
        actor_critic.load_state_dict(torch.load(model_save_path, map_location=device))
    actor_critic.to(device)
    
    logger = Logger(open("data/log/log_radial_{}_{}.txt".format(args.algo, exp_name, victim_type), "w"))
    
    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm,
            imitate=args.imitate)
    elif args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm,
            imitate=args.imitate)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, 
            args.entropy_coef, acktr=True, 
            beta=False, imitate=args.imitate,
            lr=args.lr,eps=args.eps,
            alpha=args.alpha, 
            max_grad_norm=args.max_grad_norm)   
    
    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, action_space,
                              actor_critic.recurrent_hidden_state_size)
    
    if args.attack_lr == 0.01:
        args.attack_lr = args.epsilon / 10
    print("attack steps", args.attack_steps, "attack lr", args.attack_lr)
    
    obs = envs.reset()
    rollouts.obs[0].copy_(state2tensor(obs, device))
    rollouts.to(device)
    episode_rewards = deque(maxlen=10)
    total_episode_rewards = []
    total_fool = deque(maxlen=10)
    rewards = torch.zeros(args.num_processes, 1, device=device)
    
    best_performance = np.inf
    performance_record = deque(maxlen=20)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    
    rewards = 0
    num_episodes = 0
    
    try:
        for j in range(num_updates):

            if args.use_linear_lr_decay:
                # decrease learning rate linearly
                utils.update_linear_schedule(
                    agent.optimizer, j, num_updates,
                    agent.optimizer.lr if args.algo == "acktr" else args.lr)
            fools = 0
            for step in range(args.num_steps):
                # Sample actions
                with torch.no_grad():
                    value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                        rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step], deterministic=args.det)
                perturb_direction = action[0]
                
                old_action = get_action(model, rollouts.obs[step])
                if args.fgsm:
                    obs_perturb = obs_dir_perturb_fgsm(model, rollouts.obs[step], perturb_direction, 
                                                       args.epsilon, device)
                elif args.momentum:
                    obs_perturb = obs_dir_perturb_momentum(model, rollouts.obs[step], perturb_direction, 
                                                       args.epsilon, device)
                else:
                    obs_perturb = obs_dir_perturb_pgd(model, rollouts.obs[step], perturb_direction, 
                                                       args.epsilon, args.attack_steps, args.attack_lr, device)

                extra_reward = 0
                if args.test and args.attacker is None:
                    obs, reward, done, info = envs.step(old_action)
                else:
                    attacked_action = get_action(model, rollouts.obs[step] + obs_perturb)
                    if old_action != attacked_action:
                        fools += 1
                        extra_reward += args.reward_bonus
                    obs, reward, done, info = envs.step(attacked_action)
                rewards += reward

                if done and not info:
                    obs = envs.reset()
                elif info:
                    obs = envs.reset()
                    episode_rewards.append(rewards)
                    performance_record.append(rewards)
                    rewards = 0
                    num_episodes += 1
                masks = torch.FloatTensor(
                    [[0.0] if done else [1.0]])
                bad_masks = torch.FloatTensor(
                    [[1.0]])
                rollouts.insert(state2tensor(obs, device), recurrent_hidden_states, action,
                                action_log_prob, value, reward2tensor(-reward+extra_reward, device), 
                                masks, bad_masks)

            with torch.no_grad():
                next_value = actor_critic.get_value(
                    rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                    rollouts.masks[-1]).detach()


            rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                     args.gae_lambda, args.use_proper_time_limits)

            value_loss, action_loss, dist_entropy = agent.update(rollouts)

            rollouts.after_update()

            total_fool.append(fools)

            if j % args.log_interval == 0 and len(episode_rewards) > 1:
                total_num_steps = (j + 1) * args.num_processes * args.num_steps
                end = time.time()
                print(
                    "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, mean fool {:.1f}, value loss {:.1f}\n"
                    .format(j, total_num_steps,
                            int(total_num_steps / (end - start)),
                            len(episode_rewards), np.mean(episode_rewards),
                            np.median(episode_rewards), np.min(episode_rewards),
                            np.max(episode_rewards), np.mean(total_fool), value_loss if not args.test else 0))
                logger.log(
                    "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, mean fool {:.1f}, value loss {:.1f}\n"
                    .format(j, total_num_steps,
                            int(total_num_steps / (end - start)),
                            len(episode_rewards), np.mean(episode_rewards),
                            np.median(episode_rewards), np.min(episode_rewards),
                            np.max(episode_rewards), np.mean(total_fool), value_loss if not args.test else 0))

            if (j % args.save_interval == 0
                    or j == num_updates - 1) and args.save_dir != "":
                if len(performance_record) > 1 and np.mean(performance_record) < best_performance:
                    best_performance = np.mean(performance_record)
                    try:
                        os.makedirs(save_path)
                    except OSError:
                        pass
                    print("saving to", model_save_path, "best performance", best_performance)
                    torch.save(actor_critic.state_dict(), model_save_path)
    except KeyboardInterrupt:
        pass
    finally:
        torch.save(actor_critic.state_dict(), os.path.join(save_path, exp_name + "_" + victim_type + "_attacker_final.pt"))


if __name__ == "__main__":
    main()