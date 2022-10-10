import copy
import glob
import os
import time
from collections import deque
import gym
from gym.spaces.box import Box
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.autograd import Variable
from VaR_DQN.a2c_ppo_acktr import algo, utils
from VaR_DQN.a2c_ppo_acktr.arguments import get_args
from VaR_DQN.a2c_ppo_acktr.envs import make_vec_envs
from VaR_DQN.a2c_ppo_acktr.model import Policy, BetaMLP
from VaR_DQN.a2c_ppo_acktr.storage import RolloutStorage
from VaR_DQN.utils.ppo_core import mlp
from VaR_DQN.utils.param import Param
from VaR_DQN.utils.dqn_core import DQN_Agent, model_get
from VaR_DQN.utils.schedule import *
from VaR_DQN.attacker.attacker import Huang_Attack, Pattanaik_Attack

COEFF = 1

def get_policy(victim, obs):
    return torch.distributions.categorical.Categorical(logits=victim.Q(obs).squeeze())

def dqn_dir_perturb_fgsm(victim, obs, direction, epsilon, device):
    """
    Targeted attack: find the best obs attack in order to perturb the policy 
    to a target direction as much as possible
    Here, it uses fast gradient sign method (FGSM) to calculate the perturbation.
    
    victim (DQN_Agent): victim Q-learning agent
    obs (np.array): observation of the agent, which has shape (n,4,84,84)
    direction (np.array): perturbation direction, which has shape (num_actions,)
    epsilon (float): perturbation radius
    device (torch.device)
    """
    init = torch.zeros_like(obs).to(device)
    perturb = Variable(init, requires_grad=True)

    ### Compute the gradient
    clean_policy = get_policy(victim, obs).probs.detach()
    policy = get_policy(victim, obs+perturb).probs
    diff = policy - clean_policy
    direction = direction.detach()
    cos_sim = nn.CosineSimilarity() 
    
    ### Update the perturbation
    loss = - torch.mean(cos_sim(diff, direction) + COEFF * torch.norm(diff, dim=1, p=2))
    loss.backward()
    grad = perturb.grad.data
    perturb.data -= epsilon * torch.sign(grad)
    return perturb.detach()

def dqn_dir_perturb_momentum(victim, obs, direction, epsilon, device, maxiter=10):
    """
    Targeted attack: find the best obs attack in order to perturb the policy 
    to a target direction as much as possible
    
    Here, instead of using FGSM to find the perturbation, we use Nesterov momentum 
    based method proposed in https://biases-invariances-generalization.github.io/pdf/big_33.pdf
    
    victim (DQN_Agent): victim Q-learning agent
    obs (np.array): observation of the agent, which has shape (n,4,84,84)
    direction (np.array): perturbation direction, which has shape (num_actions,)
    epsilon (float): perturbation radius
    device (torch.device)
    maxiter (int): maximum amount of iterations in Nesterov momentum methods
    """
    clean_policy = get_policy(victim, obs).probs.detach()
    direction = direction.detach()
    cos_sim = nn.CosineSimilarity() 

    def loss_fn(perturbed_obs):
        perturbed_policy = get_policy(victim, perturbed_obs).probs
        diff = perturbed_policy - clean_policy
        loss = torch.mean(cos_sim(diff, direction) + COEFF * torch.norm(diff, dim=1, p=2))
        return loss

    mu = 0.5
    v = torch.zeros_like(obs).to(device)
    lr = epsilon / maxiter ### The step-size that we use

    obs_adv = obs.clone().detach().to(device)
    for i in range(maxiter):
        ### Compute the gradient
        _obs_adv = obs_adv.clone().detach().requires_grad_(True)
        loss = loss_fn(_obs_adv + mu * v)
        loss.backward(torch.ones_like(loss))
        gradients = _obs_adv.grad

        ### Update the perturbation
        v = mu * v + gradients/torch.norm(gradients, p=1)
        obs_adv += v.sign().detach() * lr
        ### Clip the perturbed states within the l-infinity norm ball
        obs_adv = torch.max(torch.min(obs_adv, obs + epsilon), obs - epsilon)
        
    return obs_adv.detach() - obs.detach()

def dqn_dir_perturb_pgd(victim, obs, direction, epsilon, device,
    maxiter=30, lr=1e-4, etol=1e-7, rand_init=False):
    """
    Targeted attack: find the best obs attack in order to perturb the policy 
    to a target direction as much as possible
    Here it uses projected gradient sign method (PGD) to calculate the perturbation.
    
    victim (DQN_Agent): victim Q-learning agent
    obs (np.array): observation of the agent, which has shape (n,4,84,84)
    direction (np.array): perturbation direction, which has shape (num_actions,)
    epsilon (float): perturbation radius
    device (torch.device)
    maxiter (int): maximum amount of iterations in projected gradient descent
    """
    clean_policy = get_policy(victim, obs).probs.detach()
    direction = direction.detach()
    cos_sim = nn.CosineSimilarity() 
    obs_adv = obs.clone().detach().to(device)
    if rand_init:
        obs_adv += (2 * epsilon * torch.rand_like(obs).to(device) - epsilon)
    for i in range(maxiter):
        ### Compute the gradient
        _obs_adv = obs_adv.clone().detach().requires_grad_(True)
        policy = get_policy(victim, _obs_adv).probs
        diff = policy - clean_policy
        loss = - torch.mean(cos_sim(diff, direction) + COEFF * torch.norm(diff, dim=1, p=2))
        loss.backward()
        
        ### Update the perturbation
        gradients = _obs_adv.grad.sign().detach()
        obs_adv -= gradients * lr
        ### Clip the perturbed states within the l-infinity norm ball
        obs_adv = torch.max(torch.min(obs_adv, obs + epsilon), obs - epsilon)

    return obs_adv.detach() - obs.detach()

def main():
    args = get_args()
    
    ### Setup random seed, cuda device and logging directory
    
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

    ### Setting up the environment
    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False)
    action_space = Box(-1.0, 1.0, (envs.action_space.n-1,))
    obs_dim = envs.observation_space.shape
    act_dim = envs.action_space.n
    
    #### PATHs ####
    exp_name = "dqn_obspol_attacker_{}_e{}_{}".format(args.env_name, args.epsilon, "fgsm" if args.fgsm else "pgd" )
    model_dir = os.path.join(args.adv_dir, args.algo)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    victim_dir = args.victim_dir
    model_path = os.path.join(model_dir, exp_name)
    result_path = os.path.join(args.res_dir, exp_name + ".txt")

    ### Define the director of PA-AD
    ### We could use either a2c, ppo, or acktr as the RL algorithm
    ### to train the director.
    
    actor_critic = Policy(
        envs.observation_space.shape,
        action_space,
        beta=False,
        epsilon=args.epsilon,
        base_kwargs={'recurrent': args.recurrent_policy})
    if args.load:
        print("----load model----")
        old_steps, load_states, _ = torch.load(model_path)
        actor_critic.load_state_dict(load_states)
        print("load a model trained for", old_steps, "steps")
    actor_critic = actor_critic.to(device)
    
    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm,
            beta=False,
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
            beta=False,
            imitate=args.imitate)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, 
            args.entropy_coef, acktr=True, 
            beta=False, imitate=args.imitate, 
            lr=args.lr,eps=args.eps, alpha=args.alpha, 
            max_grad_norm=args.max_grad_norm)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, action_space,
                              actor_critic.recurrent_hidden_state_size)

    ## load pre-trained victim
    if args.cuda: 
        Param(torch.cuda.FloatTensor, device)
    else:
        Param(torch.FloatTensor, device)
    
    ### Load Victim Q agent
    q_func = model_get('Atari', num_actions = envs.action_space.n, duel=False)
    victim = DQN_Agent(q_func)
    agent_dir = os.path.join(victim_dir, args.env_name)
    victim.load_state_dict(torch.load(agent_dir, map_location=Param.device))
    victim.to(device)
    
    vec_norm = utils.get_vec_normalize(envs)
    if vec_norm is not None:
        vec_norm.eval()
    
    ### Here it's not using any recurrent neural network, but for the sake of time,
    ### we keep the recurrent as in the original repo. This wouldn't make any difference
    ### to the result of our code.
    default_recurrent = torch.zeros(
        args.num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    default_masks = torch.ones(args.num_processes, 1, device=device)
    
    ### Set-up a couple of logging items before rollout 
    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)
    episode_rewards = deque(maxlen=10)
    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    rew_file = open(result_path, "wt")
    best_performance = np.inf
    performance_record = deque(maxlen=100)
        
    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step], beta=False, deterministic=args.det)

            ### Compute the policy perturbation direction.
            ### Here the action space of the original environment is discrete, 
            ### o the action space of the director is only |A|-1. The last dimension of the
            ### policy perturbation is given by 1-\sum a_i
            perturb_direction = torch.cat((action, -torch.sum(action, dim=1, keepdim=True)), 1)
        
            ### Compute the perturbation in the state space
            if args.fgsm:
                obs_perturb = dqn_dir_perturb_fgsm(victim, rollouts.obs[step], perturb_direction, 
                        args.epsilon, device)
            elif args.momentum:
                obs_perturb = dqn_dir_perturb_momentum(victim, rollouts.obs[step], perturb_direction, 
                        args.epsilon, device, maxiter=args.attack_steps)
            else:
                obs_perturb = dqn_dir_perturb_pgd(victim, rollouts.obs[step], perturb_direction, 
                        args.epsilon, device, lr=args.attack_lr, maxiter=args.attack_steps, 
                        rand_init=args.rand_init)
            
            ### Compute the agent's action based on perturbed observation.
            v_action = victim.step_torch_batch(obs+obs_perturb)

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(v_action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
                    performance_record.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = Param.dtype(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = Param.dtype(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, -reward, masks, bad_masks)
        
        ### Update the director
        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()
        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)
        if step % args.train_freq == 0:
            value_loss, action_loss, dist_entropy = agent.update(rollouts)
        rollouts.after_update()

        ### Save the director after args.save_interval iterations
        if args.save_interval > 0 and (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            if len(performance_record) > 1 and np.mean(performance_record) < best_performance:
                total_num_steps = (j + 1) * args.num_processes * args.num_steps
                print("*** save for", np.mean(performance_record))
                best_performance = np.mean(performance_record)
                torch.save([
                    total_num_steps,
                    actor_critic.state_dict(),
                    getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
                ], model_path)

        if j % args.log_interval == 0:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            if not args.verbose:
                print(
                    "Iteration {}, num timesteps {}, FPS {}"
                    .format(j, total_num_steps, int(total_num_steps / (end - start))))
            if len(episode_rewards) > 1 and not args.train_nn:
                if not args.verbose:
                    print(
                    "Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, {:.1f}, {:.1f}, {:.1f}"
                    .format(len(episode_rewards), np.mean(episode_rewards),
                            np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), value_loss, action_loss, dist_entropy))
                rew_file.write("Step: {}, Reward: {} \n".format(total_num_steps, np.mean(episode_rewards)))
            if not args.verbose:
                print(obs_perturb[0])
                print("norm", torch.norm(obs_perturb, p=np.inf))
            

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            ob_rms = utils.get_vec_normalize(envs).ob_rms
            evaluate(actor_critic, ob_rms, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device)
    rew_file.close()

if __name__ == "__main__":
    main()
