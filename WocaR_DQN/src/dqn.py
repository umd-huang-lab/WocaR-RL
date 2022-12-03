import sys
import os
sys.path.append("/Users/liangyongyuan/robust-rl/code_atari/WocaR_DQN")
sys.path.append("/Users/liangyongyuan/robust-rl/code_atari")
import pickle
import numpy as np
from collections import namedtuple, deque
import copy
import time
from WocaR_DQN.utils.monitor import Monitor
from WocaR_DQN.utils.dqn_core import *
from WocaR_DQN.utils.atari_utils import atari_env, make_env
from WocaR_DQN.utils.replay_buffer import *
from WocaR_DQN.utils.schedule import *
from WocaR_DQN.utils.load_config import load_config
from WocaR_DQN.utils.param import Param
import random
import gym
import gym.spaces
gym.logger.set_level(40)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

def dqn(config):

    dtype = Param.dtype
    device = Param.device
    exp_name = config['env_name'] if config['env_name'] == 'none' else config['exp_name']
    
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    env = gym.make(config['env_name'])
    env = Monitor(env)
    env = make_env(env, frame_stack=False, scale=False)
    #env = atari_env(config['env_name'])
    q_func = model_get('Atari', num_actions = env.action_space.n, duel=config["duel"])
        
    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space)      == gym.spaces.Discrete
    num_actions = env.action_space.n  
    
    
    ### Setup Exploration Schedule
    exploration = PiecewiseSchedule([(0, config["exploration"]["exp_initp"]), 
                                     (config["exploration"]["exploration_frames"],config["exploration"]["exp_finalp"])], 
                                    outside_value=config["exploration"]["exp_finalp"])
    if config["prioritized_replay"]:
        beta_schedule = PiecewiseSchedule([(0,config["beta_start"]), (config["frame_total"], 1.0)], 
                                                outside_value=1.0)
    else:
        beta_schedule = None
        
    alpha_schedule = PiecewiseSchedule([(0, config["alpha_schedule"]["alpha_init"]), 
                                     (config["alpha_schedule"]["steps"],config["alpha_schedule"]["alpha_final"])], 
                                     outside_value=config["alpha_schedule"]["alpha_final"])
    eps_schedule = PiecewiseSchedule([(0, config["eps_schedule"]["eps_init"]), 
                                     (config["eps_schedule"]["steps"],config["eps_schedule"]["eps_final"])], 
                                     outside_value=config["eps_schedule"]["eps_final"])
    
    dqn_agent = DQN_Agent(q_func, learning_rate=config["lr"],doubleQ=config["doubleQ"], 
                          update_freq=config["update_freq"], 
                          robust=config["robust"], alpha_schedule=alpha_schedule, 
                          eps_schedule=eps_schedule, kappa=config['kappa'], 
                          pgd_param=config['pgd_param'], reg_solver=config['reg_solver'])
    if config["trained_dir"] != 'none':
        dqn_agent.load(config["trained_dir"])
    
    if not config["prioritized_replay"]:
        replay_buffer = DQNReplayBuffer(num_actions, config["buffer_size"],
                                        config["batch_size"],config["seed"])
    else:
        replay_buffer = DQNPrioritizedBuffer(config["buffer_size"], 
                                             config["batch_size"],seed=config["seed"])
    
    ### Set up logger file
    logger_file = open(os.path.join(Param.data_dir, r"logger_{}.txt".format(exp_name)), "wt")
    
    
    ### Prepareing for running on the environment
    t, counter = 0, 0
    mean_episode_reward = -float('nan')
    best_mean_episode_reward = -float('inf')
    scores = []
    scores_window = deque(maxlen=config["log_steps"])
    ep_len_window = deque(maxlen=config["log_steps"])
    time_window = deque(maxlen=config["log_steps"])
    
    while (counter<config["frame_total"]):
        t += 1
        score, steps = 0, 0
        last_obs = env.reset()
        start = time.time()
        while (True):
            if (counter>config["learning_starts"] or config["trained_dir"] is not None):
                ### Epsilon Greedy Policy
                last_obs_normalized = last_obs/255. 
                eps = exploration.value(counter)
                action = dqn_agent.select_epilson_greedy_action(last_obs_normalized, eps)
            else: 
                ### Randomly Select an action before the learning starts
                action = random.randrange(num_actions)
            # Advance one step
            obs, reward, done, info = env.step(action)
            steps += 1
            counter+=1
            score += reward
            ### Add the experience into the buffer
            replay_buffer.add(last_obs, action, reward, obs, done)
            ### Update last observation
            last_obs = obs
            
            ### Q learning udpates
            if (counter > config["learning_starts"] and
                counter % config["learning_freq"] == 0):
                if not config["prioritized_replay"]:
                    obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = replay_buffer.sample()
                    not_done_mask = 1 - done_mask
                    obs_batch = obs_batch/255. 
                    next_obs_batch = next_obs_batch/255.
                    dqn_agent.update(obs_batch, act_batch, rew_batch, \
                                     next_obs_batch, not_done_mask, config["gamma"], config["tau"], which_q='Q')
                else:
                    obs_batch, act_batch, rew_batch, next_obs_batch, \
                    done_mask, indices, weights = replay_buffer.sample(beta=beta_schedule.value(counter))
                    obs_batch, next_obs_batch = obs_batch.squeeze(1), next_obs_batch.squeeze(1)
                    not_done_mask = (1 - done_mask).unsqueeze(1)
                    obs_batch = obs_batch/255. 
                    next_obs_batch = next_obs_batch/255. 
                    priority = dqn_agent.update(obs_batch, act_batch, rew_batch, \
                                     next_obs_batch, not_done_mask, config["gamma"], config["tau"], weights=weights, which_q='Q')   
                    replay_buffer.update_priorities(indices, priority.cpu().numpy())
                    
                if config["robust"]:
                    dqn_agent.update(obs_batch, act_batch, rew_batch, \
                                    next_obs_batch, not_done_mask, config["gamma"], config["tau"], which_q='worst_Q')
                    dqn_agent.update(obs_batch, act_batch, rew_batch, \
                                    next_obs_batch, not_done_mask, config["gamma"], config["tau"], which_q='robust_Q')
                    
            if done or steps>config["max_steps"]:
                ep_len_window.append(steps)
                scores_window.append(score)
                steps = 0
                break
    
        scores.append(score)
        time_window.append(time.time()-start)
        ### print and log the learning process
        if t % config["log_steps"] == 0 and counter>config["learning_starts"]:
            print("------------------------------Episode {}------------------------------------".format(t))
            logger_file.write("------------------------------Episode {}------------------------------------\n".format(t))
            print('Num of Interactions with Environment:{:.2f}k'.format(counter/1000))
            logger_file.write('Num of Interactions with Environment:{:.2f}k\n'.format(counter/1000))
            print('Mean Training Reward per episode: {:.2f}'.format(np.mean(scores_window)))
            logger_file.write('Mean Training Reward per episode: {:.2f}\n'.format(np.mean(scores_window)))
            print('Average Episode Length: {:.2f}'.format(np.mean(ep_len_window)))
            logger_file.write('Average Episode Length: {:.2f}\n'.format(np.mean(ep_len_window)))
            print('Average Time: {:.2f}'.format(np.mean(time_window)))
            logger_file.write('Average Time: {:.2f}\n'.format(np.mean(time_window)))
            eval_reward = roll_out_atari(dqn_agent, env)
            print('Eval Reward:{:.2f}'.format(eval_reward))
            logger_file.write('Eval Reward:{:.2f}\n'.format(eval_reward))
            logger_file.flush()
            dqn_agent.save(exp_name=exp_name)
    logger_file.close()
    dqn_agent.save(exp_name=exp_name)
    return 
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="./config/rqdqn.json")
    parser.add_argument('--cuda', type=int, default=None)
    args = parser.parse_args()
    config = load_config(args)
    
    if config["no_cuda"]:
        Param(torch.FloatTensor, torch.device("cpu"))
    else:
        Param(torch.cuda.FloatTensor, torch.device("cuda:{}".format(config["cuda"] 
                                                    if args.cuda is None else args.cuda)))

    dqn(config)
    
