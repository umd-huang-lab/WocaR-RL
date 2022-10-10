import copy
import glob
import os
import sys
import time
from collections import deque
import argparse

import gym
from gym.spaces.box import Box
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from VaR_DQN.a2c_ppo_acktr import algo, utils
from VaR_DQN.a2c_ppo_acktr.algo import gail
from VaR_DQN.a2c_ppo_acktr.envs import make_vec_envs
from VaR_DQN.a2c_ppo_acktr.model import Policy
from VaR_DQN.a2c_ppo_acktr.storage import RolloutStorage
from VaR_DQN.attacker.attacker import *
from VaR_DQN.utils.dqn_core import DQN_Agent, Q_Atari, model_get
from VaR_DQN.a2c_ppo_acktr.algo.kfac import KFACOptimizer

ATT_MAP = {
    'minbest': 'huang',
    'momentum': 'huang',
    'minq': 'patt',
    'maxdiff': 'kl',
    'random': 'random',
#     'sa': 'obs',
    'paad': 'obspol'
}

def get_args():
    parser = argparse.ArgumentParser(description='Test Attack')
    parser.add_argument(
        '--v-path', 
        type=str, 
        help='path of the victim model')
    parser.add_argument(
        '--v-type', 
        type=str, 
        default='a2c',
        help='type of the victim policy (dqn, a2c, custom)')
    parser.add_argument(
        '--attacker', 
        type=str, 
        default=None,
        help='attack method')
    parser.add_argument(
        '--attack-model', 
        type=str, 
        default=None,
        help='path of the attack model (only for RL-based attackers)')
    parser.add_argument(
        '--det',
        action='store_true',
        default=False,
        help='whether to use deterministic policy')
    parser.add_argument(
        '--v-det',
        action='store_true',
        default=False,
        help='whether victim uses deterministic policy')
    parser.add_argument(
        '--epsilon',
        type=float,
        default=0.01,
        help='the attack budget')
    parser.add_argument(
        '--test-episodes',
        type=int,
        default=1000,
        help='number of episodes to test return (default: 1000)')
    parser.add_argument(
        '--env-name',
        default='PongNoFrameskip-v4',
        help='environment to train on (default: PongNoFrameskip-v4)')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='discount factor for rewards (default: 0.99)')
    parser.add_argument(
        '--log-dir',
        default='./data/log/',
        help='directory to save agent logs (default: ./data/log/)')
    parser.add_argument(
        '--num-processes',
        type=int,
        default=32,
        help='how many training CPU processes to use (default: 32)')
    parser.add_argument(
        '--seed', 
        type=int, 
        default=1, 
        help='random seed (default: 1)')
    parser.add_argument(
        '--attack-lr',
        type=float,
        default=0.01,
        help='PGD attack learning rate')
    parser.add_argument(
        '--attack-steps',
        type=int,
        default=10,
        help='PGD attack learning steps')
    parser.add_argument(
        '--rand-init',
        action='store_true',
        default=False,
        help='whether to use a random initialization for pgd attacks')
    parser.add_argument(
        '--res-dir',
        default='./data/test_results/',
        help='directory to save agent rewards (default: ./data/test_results/)')
    parser.add_argument(
        '--cuda-id',
        type=int,
        default=0)
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')
    
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    assert args.attacker in ['random', 'minbest', 'momentum', 'minq', 'maxdiff', 'paad', None]
    assert args.v_type in ['a2c', 'dqn']
    
    global ATT_METHOD
    global FGSM
    global MOMENTUM
    if args.attacker:
        ATT_METHOD = ATT_MAP[args.attacker]
    else:
        ATT_METHOD = None
    if ATT_METHOD == 'huang' or ATT_METHOD == 'obspol':
        FGSM = True
    else:
        FGSM = False
    if ATT_METHOD == 'momentum':
        MOMENTUM = True
    else:
        MOMENTUM = False
    print(ATT_METHOD, FGSM, MOMENTUM, args.epsilon)
    return args

def agent_act(agent, obs, masks, args, device):
    with torch.no_grad():
        if args.v_type == 'a2c':
            recurrent = torch.zeros(args.num_processes, agent.recurrent_hidden_state_size, device=device)
            _, action, _, _ = agent.act(obs, recurrent, masks, deterministic=args.v_det)
        elif args.v_type == 'dqn':
            action = agent.step_torch_batch(obs)
    return action

def attack(attacker, agent, obs, masks, args, device):
    
    recurrent = torch.zeros(args.num_processes, 1, device=device)
    
    if args.v_type == 'a2c':
        if ATT_METHOD == "obs":
            perturbed_obs = attacker.attack_stoc(obs, recurrent, masks, epsilon=args.epsilon, device=device)
        elif ATT_METHOD == "random":
            perturbed_obs = attacker.attack_stoc(obs, epsilon=args.epsilon, device=device)
        else:
            perturbed_obs = attacker.attack_stoc(agent, obs, recurrent, masks, epsilon=args.epsilon, 
                                       fgsm=FGSM, lr=args.attack_lr, pgd_steps=args.attack_steps, 
                                       device=device, rand_init=args.rand_init, momentum=MOMENTUM)
    elif args.v_type == 'dqn':
        if ATT_METHOD == "obs":
            perturbed_obs, _ = attacker.attack_torch(obs, recurrent, masks, epsilon=args.epsilon, device=device)
        elif ATT_METHOD == "obspol":
            perturbed_obs, _ = attacker.attack_torch(agent, obs, recurrent, masks, epsilon=args.epsilon,
                        fgsm=FGSM, lr=args.attack_lr, pgd_steps=args.attack_steps, device=device, 
                        rand_init=args.rand_init, momentum=MOMENTUM)
        elif ATT_METHOD == "random":
            perturbed_obs = attacker.attack_torch(obs, epsilon=args.epsilon, device=device)
        else:
            perturbed_obs = attacker.attack_torch(agent.Q, obs, epsilon=args.epsilon, fgsm=FGSM, 
                        lr=args.attack_lr, pgd_steps=args.attack_steps, device=device, 
                        rand_init=args.rand_init, momentum=MOMENTUM)
    return perturbed_obs

def main():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)
    
    if not os.path.exists(args.res_dir):
        os.makedirs(args.res_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:{}".format(args.cuda_id) if args.cuda else "cpu")

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False)
    if args.cuda: 
        Param(torch.cuda.FloatTensor, device)
    else:
        Param(torch.FloatTensor, device)
        
    if envs.action_space.__class__.__name__ == "Discrete":
        action_space = Box(-1.0, 1.0, (envs.action_space.n-1,))
        cont = False
    elif envs.action_space.__class__.__name__ == "Box":
        action_space = Box(-1.0, 1.0, (envs.action_space.shape[0],))
        cont = True

    ## Load the victim model
    if args.v_type == 'dqn':
        print("loading dqn victim agent from", args.v_path)
        q_func = model_get('Atari', num_actions = envs.action_space.n, duel=False)
        agent = DQN_Agent(q_func)
        agent_dir = os.path.join(args.v_path)
        agent.load_state_dict(torch.load(agent_dir, map_location=device))
    elif args.v_type == 'a2c':
        agent = Policy(
            envs.observation_space.shape,
            envs.action_space,
            base_kwargs={'recurrent': False})
        print("loading a2c victim agent from", args.v_path)
        agent_states, ob_rms = torch.load(args.v_path, map_location=device)
        agent.load_state_dict(agent_states)
    agent.to(device)

    vec_norm = utils.get_vec_normalize(envs)
    if vec_norm is not None:
        vec_norm.eval()

    if args.attacker:
        vtype = "_vdet" if args.v_det else ""
        rew_file = open(os.path.join(args.res_dir, r"{}_{}_{}_e{}{}.txt".format(args.v_type, 
                args.env_name, ATT_METHOD, args.epsilon, vtype)), "wt")
        if ATT_METHOD == "huang":
            attacker = Huang_Attack()
        elif ATT_METHOD == "patt":
            if args.v_type == 'a2c':
                raise Exception("Sorry, MinQ Attack is not applicable to the A2C victim.")
            attacker = Pattanaik_Attack()
        elif ATT_METHOD == "kl":
            attacker = KL_Attack()
        elif ATT_METHOD == "random":
            attacker = Random_Attack()
        elif ATT_METHOD == "obs":
            print("load obs attacker")
            old_steps, obs_attacker, _ = torch.load(args.attack_model, map_location=device)
            obs_attacker.to(device)
            attacker = Obs_Attack(obs_attacker, envs.observation_space.shape, det=args.det)
            print("training steps for this model:", old_steps)
        elif ATT_METHOD == "obspol":
            print("load paad attacker")
            pa_attacker = Policy(
                envs.observation_space.shape,
                action_space,
                beta=False,
                epsilon=args.epsilon,
                base_kwargs={'recurrent': False})
            KFACOptimizer(pa_attacker) # the model structure for the acktr attacker is different
            old_steps, pa_attacker_state, _ = torch.load(args.attack_model, map_location=device)
            pa_attacker.load_state_dict(pa_attacker_state)
            pa_attacker.to(device)
            attacker = ObsPol_Attack(pa_attacker, det=args.det, cont=cont)
            print("training steps for this model:", old_steps)
    
    else:
        rew_file = open(os.path.join(args.res_dir, r"{}_{}_noattack.txt".format(args.v_type, args.env_name)), "wt")
    
    obs = envs.reset()
    
    masks = torch.ones(args.num_processes, 1, device=device)
    ## Attack obs (if any)
    if args.attacker:
        obs = attack(attacker, agent, obs, masks, args, device)

    episode_rewards = deque(maxlen=10)
    start = time.time()
    
    num_episodes = 0
    all_rewards = []
    
    total_num_steps = 0 
    while True:
        total_num_steps += 1
        # Sample actions
        with torch.no_grad():
            v_action = agent_act(agent, obs, masks, args, device)

        # Obser reward and next obs
        obs, reward, done, infos = envs.step(v_action)

        for info in infos:
            if 'episode' in info.keys():
                episode_rewards.append(info['episode']['r'])
                num_episodes += 1
                rew_file.write("Episode: {}, Reward: {} \n".format(num_episodes, info['episode']['r']))
                all_rewards.append(info['episode']['r'])

        # If done then clean the history of observations.
        masks = torch.FloatTensor(
            [[0.0] if done_ else [1.0] for done_ in done])

        ## Attack obs (if any)
        old_obs = obs.clone()
        if args.attacker:
            obs = attack(attacker, agent, obs, masks, args, device)

        if num_episodes >= args.test_episodes:
            break
        
        if total_num_steps % 100 == 0:
            end = time.time()
            print(
                "num timesteps {}, FPS {}"
                .format(total_num_steps, int(total_num_steps / (end - start))))
            if len(episode_rewards) > 1:
                print(
                "Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}"
                .format(len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards)))
            if args.attacker:
                print("attack amount", torch.norm(obs-old_obs, p=np.inf).item())
        
    all_rewards = np.array(all_rewards)
    print("Average rewards", np.mean(all_rewards).round(2), "std", np.std(all_rewards).round(2))
    rew_file.write("Average rewards:" + str(np.mean(all_rewards).round(2)) + " std: " + str(np.std(all_rewards).round(2)))
    rew_file.close()

if __name__ == "__main__":
    main()