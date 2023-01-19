import copy
import glob
import os
import sys
#import time
from collections import deque

import gym
from gym.spaces.box import Box
import numpy as np
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
from WocaR_DQN.attacker.attacker import *
from WocaR_DQN.utils.dqn_core import DQN_Agent, Q_Atari,model_get
from WocaR_DQN.utils.param import Param
from WocaR_DQN.a2c_ppo_acktr.algo.kfac import KFACOptimizer

def main():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    try:
        utils.cleanup_log_dir(log_dir)
        utils.cleanup_log_dir(eval_log_dir)
    except:
        pass
    
    torch.set_num_threads(1)
    device = torch.device("cuda:{}".format(args.cuda_id) if args.cuda else "cpu")

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False)
    print("The observation space is", envs.observation_space)
    
    if args.cuda: 
        Param(torch.cuda.FloatTensor, device)
    else:
        Param(torch.FloatTensor, device)
    
    q_func = model_get('Atari', num_actions = envs.action_space.n, duel=False)
    agent = DQN_Agent(q_func)
    agent_dir = os.path.join(args.victim_dir, args.env_name)
    agent.load_state_dict(torch.load(agent_dir, map_location=Param.device))

    vec_norm = utils.get_vec_normalize(envs)
    if vec_norm is not None:
        vec_norm.eval()

    if args.attacker:
        optim_method = "momentum" if args.momentum else ("fgsm" if args.fgsm else "pgd" )
        rew_file = open(os.path.join(args.res_dir, r"dqn_{}_{}_e{}_{}.txt".format( 
                args.env_name, args.attacker, args.epsilon, optim_method)), "wt")
        if args.attacker == "minbest":
            attacker = Huang_Attack()
        elif args.attacker == "minq":
            attacker = Pattanaik_Attack()
        elif args.attacker == "maxdiff":
            attacker = KL_Attack()
        elif args.attacker == "random":
            attacker = Random_Attack()
        elif args.attacker == "sarl":
            print("load obs attacker")
            exp_name = "dqn_obs_attacker_{}_e{}_{}".format(args.env_name, args.epsilon,
                "fgsm" if args.fgsm else "pgd" )
            action_space = Box(-args.epsilon, args.epsilon, envs.observation_space.shape)
            obs_attacker = Policy(
                envs.observation_space.shape,
                action_space,
                beta=False,
                epsilon=args.epsilon,
                base_kwargs={'recurrent': False})
            old_steps, obs_attacker_state, _ = \
                    torch.load(os.path.join(args.adv_dir, args.algo, 
                        exp_name), map_location=device)
            obs_attacker.load_state_dict(obs_attacker_state)
            obs_attacker.to(device)
            attacker = Obs_Attack(obs_attacker, envs.observation_space.shape)
            recurrent = torch.zeros(
                args.num_processes, obs_attacker.recurrent_hidden_state_size, device=device)
            masks = torch.ones(args.num_processes, 1, device=device)
            print("training steps for this model:", old_steps)
        elif args.attacker == "paad":
            print("load observation-policy attacker")
            exp_name = "dqn_obspol_attacker_{}_e{}_{}".format(args.env_name, args.epsilon, "fgsm" if args.fgsm else "pgd" )
            action_space = Box(-1.0, 1.0, (envs.action_space.n-1,))
            pa_attacker = Policy(
                envs.observation_space.shape,
                action_space,
                beta=False,
                epsilon=args.epsilon,
                base_kwargs={'recurrent': False})
            if args.algo == "acktr":
                KFACOptimizer(pa_attacker) # the model structure for the acktr attacker is different
            old_steps, pa_attacker_state, _ = \
                    torch.load(os.path.join(args.adv_dir, args.algo,
                        exp_name), map_location=device)
            pa_attacker.load_state_dict(pa_attacker_state)
            pa_attacker.to(device)
            attacker = ObsPol_Attack(pa_attacker, det=args.det, cont=False)
            recurrent = torch.zeros(
                args.num_processes, pa_attacker.recurrent_hidden_state_size, device=device)
            masks = torch.ones(args.num_processes, 1, device=device)
            print("training steps for this model:", old_steps)
    else:
        rew_file = open(os.path.join(args.res_dir, r"{}_{}_noattack.txt".format(args.algo, args.env_name)), "wt")
    
    obs = envs.reset()
    
    ## Attack obs (if any)
    if args.attacker:
        if args.attacker == "sarl":
            obs, recurrent = attacker.attack_torch(obs, recurrent, masks, epsilon=args.epsilon, device=device)
        elif args.attacker == "paad":
            obs, recurrent = attacker.attack_torch(agent, obs, recurrent, masks, epsilon=args.epsilon,
                        fgsm=args.fgsm, lr=args.attack_lr, pgd_steps=args.attack_steps, device=device, 
                        rand_init=args.rand_init, momentum=args.momentum)
        elif args.attacker == "random":
            obs = attacker.attack_torch(obs, epsilon=args.epsilon, device=device)
        else:
            obs = attacker.attack_torch(agent.Q, obs, epsilon=args.epsilon, fgsm=args.fgsm, 
                        lr=args.attack_lr, pgd_steps=args.attack_steps, device=device, 
                        rand_init=args.rand_init, momentum=args.momentum)

    episode_rewards = deque(maxlen=10)
    # start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    
    num_episodes = 0
    all_rewards = []
    
    for j in range(num_updates):
        
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                action = agent.step_torch_epsilon_greedy(obs, 0.01)
            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)
            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
                    num_episodes += 1
                    rew_file.write("Episode: {}, Reward: {} \n".format(num_episodes, info['episode']['r']))
                    all_rewards.append(info['episode']['r'])
            
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])

            ## Attack obs (if any)
            softmax = torch.nn.Softmax(dim=-1)
            clean_policy = softmax(agent.Q(obs))
            old_obs = obs.clone()
            if args.attacker:
                if args.attacker == "sarl":
                    obs, recurrent = attacker.attack_torch(obs, recurrent, masks, epsilon=args.epsilon, device=device)
                elif args.attacker == "paad":
                    obs, recurrent = attacker.attack_torch(agent, obs, recurrent, masks, epsilon=args.epsilon,
                                fgsm=args.fgsm, lr=args.attack_lr, pgd_steps=args.attack_steps, device=device, 
                                rand_init=args.rand_init, momentum=args.momentum)
                elif args.attacker == "random":
                    obs = attacker.attack_torch(obs, epsilon=args.epsilon, device=device)
                else:
                    obs = attacker.attack_torch(agent.Q, obs, epsilon=args.epsilon, fgsm=args.fgsm, lr=args.attack_lr, pgd_steps=args.attack_steps, device=device, rand_init=args.rand_init, momentum=args.momentum)
        
        if num_episodes >= args.test_episodes:
            break

        if j % args.log_interval == 0:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            if len(episode_rewards) > 1:
                if not args.verbose:
                    print(
                        "Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}"
                        .format(len(episode_rewards), np.mean(episode_rewards),
                                np.median(episode_rewards), np.min(episode_rewards),
                                np.max(episode_rewards)))
            if args.attacker:
                if not args.verbose:
                    print("attack amount", torch.norm(obs-old_obs, p=np.inf))

    all_rewards = np.array(all_rewards)
    print("Average rewards", np.mean(all_rewards), "std", np.std(all_rewards))
    rew_file.write("Average rewards:" + str(np.mean(all_rewards).round(2)) + ", std:" + str(np.std(all_rewards).round(2)))
    rew_file.close()

if __name__ == "__main__":
    main()
