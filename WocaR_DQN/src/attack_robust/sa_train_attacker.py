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
from WocaR_DQN.a2c_ppo_acktr.algo.imitation import DataBuffer
# from evaluation import evaluate
from WocaR_DQN.attacker.attacker import common_fgsm, common_pgd, common_momentum_fgm, noisy_pgd
from sa_wrappers import make_atari, wrap_deepmind, wrap_pytorch, make_atari_cart
from sa_utils import Logger, QNetwork, model_setup
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from torch.autograd import Variable

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
    
    return perturb.detach()

def obs_dir_perturb_pgd(victim, obs, direction, epsilon, pgd_steps, pgd_lr, device):
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
    
    return common_pgd(obs, epsilon, loss_fn, device, pgd_steps, lr=pgd_lr, rand_init=False) - obs.detach()

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
        
    return obs_adv.detach() - obs.detach()

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

def imitate_update(agent, expert_data, device):
    expert_obs, expert_action = expert_data
    ce_loss = nn.CrossEntropyLoss()
    
    recurrent_hidden_state_size = agent.actor_critic.base.recurrent_hidden_state_size
    recurrent = torch.zeros(expert_action.shape[0], recurrent_hidden_state_size, device=device)
    masks = torch.ones(expert_action.shape[0], 1, device=device)
    action_dist = agent.actor_critic.get_dist(expert_obs, recurrent, masks)
    print("action dist", action_dist)
    
    loss = ce_loss(action, expert_action.squeeze(1))
    print("Current Cross Entropy Loss:{}".format(loss))
    agent.optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(agent.actor_critic.parameters(), agent.max_grad_norm)
    agent.optimizer.step() 
    
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
        
    envs = make_atari(args.env_name)
    envs = wrap_deepmind(envs, clip_rewards=False, episode_life=False, central_crop=True, 
                         restrict_actions=config['restrict_actions'], crop_shift=config['crop_shift'])
    envs = wrap_pytorch(envs)
    print("The observation space is", envs.observation_space)
    
    action_space = Discrete(envs.action_space.n) #Box(-1.0, 1.0, (envs.action_space.n-1,))
    print("The action space is", action_space)
    
    args.num_processes = 1
    
    exp_name = args.env_name + "_" + str(np.round(args.epsilon*255))

    if args.fgsm:
        exp_name += "_fgsm"
    elif args.momentum:
        exp_name += "_momentum"
    else:
        exp_name += "_pgd"
        
    save_path = "./learned_adv/{}/".format(args.algo)
    model_save_path =  os.path.join(save_path, exp_name + "_sa_attacker")
         
    actor_critic = Policy(
            envs.observation_space.shape,
            action_space,
            base_kwargs={'recurrent': args.recurrent_policy})
    if args.load:
        actor_critic.load_state_dict(torch.load(model_save_path))
        print("load model from", model_save_path)

    actor_critic.to(device)
    
    # load sa model
    logger = Logger(open("data/log/log_sa_train_{}.txt".format(args.env_name), "w"))
    
    model = model_setup(args.env_name, envs, robust_model=True, logger=logger, use_cuda=True, 
                        dueling=True, model_width=1, device=device)
    model.features.load_state_dict(torch.load("released_models/sa_models/"+config_name+"-convex.model"))
    print("loaded sa model")
    
    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
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
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

    if args.gail:
        assert len(envs.observation_space.shape) == 1
        discr = gail.Discriminator(
            envs.observation_space.shape[0] + envs.action_space.shape[0], 100,
            device)
        file_name = os.path.join(
            args.gail_experts_dir, "trajs_{}.pt".format(
                args.env_name.split('-')[0].lower()))
        
        expert_dataset = gail.ExpertDataset(
            file_name, num_trajectories=4, subsample_frequency=20)
        drop_last = len(expert_dataset) > args.gail_batch_size
        gail_train_loader = torch.utils.data.DataLoader(
            dataset=expert_dataset,
            batch_size=args.gail_batch_size,
            shuffle=True,
            drop_last=drop_last)
        
    if args.attack_lr == 0.01:
        args.attack_lr = args.epsilon / 10
    print("attack steps", args.attack_steps, "attack lr", args.attack_lr)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(state2tensor(obs, device))
    rollouts.to(device)
    total_episode_rewards = []
    episode_rewards = deque(maxlen=10)
    total_fool = deque(maxlen=10)
    
    best_performance = np.inf
    performance_record = deque(maxlen=20)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    
    rewards = 0
    num_episodes = 0
    recurrent = torch.zeros(1, actor_critic.recurrent_hidden_state_size, device=device)
    masks = torch.ones(1, 1, device=device)
    
    try:
        for j in range(num_updates):

            if args.use_linear_lr_decay:
                # decrease learning rate linearly
                utils.update_linear_schedule(
                    agent.optimizer, j, num_updates,
                    agent.optimizer.lr if args.algo == "acktr" else args.lr)
            fools = 0
            for step in range(args.num_steps):
                old_action = model.act(rollouts.obs[step])[0]
                
                # Sample actions
                with torch.no_grad():
                    value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                        rollouts.obs[step], recurrent, masks, deterministic=args.det)
                    
                perturb_direction = action[0]
                
                if args.fgsm:
                    obs_perturb = obs_dir_perturb_fgsm(model, rollouts.obs[step], perturb_direction, 
                                                       args.epsilon, device=device)
                elif args.momentum:
                    obs_perturb = obs_dir_perturb_momentum(model, rollouts.obs[step], perturb_direction, 
                                                       args.epsilon, pgd_steps=args.attack_steps, 
                                                           pgd_lr=args.attack_lr, device=device)
                else:
                    obs_perturb = obs_dir_perturb_pgd(model, rollouts.obs[step], perturb_direction, 
                                                       args.epsilon, pgd_steps=args.attack_steps, 
                                                      pgd_lr=args.attack_lr, device=device)
#                 extra_reward = 0
                if args.test and args.attacker is None:
                    obs, reward, done, info = envs.step(old_action)
                else:
                    attacked_action = model.act(rollouts.obs[step] + obs_perturb)[0]
                    if old_action != attacked_action:
                        fools += 1
                    obs, reward, done, info = envs.step(attacked_action)

                rewards += reward
                

                # If done then clean the history of observations.
                if done:
                    episode_rewards.append(rewards)
                    if args.test:
                        total_episode_rewards.append(rewards)
                    performance_record.append(rewards)
                    rewards = 0
                    num_episodes += 1
                    obs = envs.reset()
                m = torch.FloatTensor(
                    [[0.0] if done else [1.0]])
                bm = torch.FloatTensor(
                    [[0.0] if 'bad_transition' in info.keys() else [1.0]])
                rollouts.insert(state2tensor(obs, device), recurrent_hidden_states, action,
                                action_log_prob, value, reward2tensor(-reward, device), 
                                m, bm)

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
                    "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, mean fool {:.1f}, value loss {:.3f}\n"
                    .format(j, total_num_steps,
                            int(total_num_steps / (end - start)),
                            len(episode_rewards), np.mean(episode_rewards),
                            np.median(episode_rewards), np.min(episode_rewards),
                            np.max(episode_rewards), np.mean(total_fool), value_loss if not args.test else 0))
                logger.log(
                    "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, mean fool {:.1f}, value loss {:.3f}\n"
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
        if not args.test:
            torch.save(actor_critic, os.path.join(save_path, exp_name + "_sa_attacker_final.pt"))


if __name__ == "__main__":
    main()