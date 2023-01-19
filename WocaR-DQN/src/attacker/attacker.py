import torch
import gym
gym.logger.set_level(40)
from WocaR_DQN.utils.param import Param
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions.categorical import Categorical
from torch.distributions.kl import kl_divergence
from torch.distributions import Beta
from WocaR_DQN.utils.dqn_core import DQN_Agent, rollout
# from WocaR_DQN.hyperparameter.attack_lr import *
from WocaR_DQN.trainer_adv.a2c_pa_attacker import obs_dir_perturb_fgsm, obs_dir_perturb_pgd, obs_dir_perturb_momentum
import numpy as np
import os

COEFF = 1

class Attacker:
    def __init__(self):
        self.name = None
        return
    def attack(obs, epsilon):
        raise NotImplementedError()

def common_fgsm(obs, epsilon, loss_fn, device):
    """
    Implementation of FGSM attacks, we want to maximize the loss function
    """
    perturb = Variable(torch.zeros_like(obs).to(device), requires_grad=True)
    loss = loss_fn(obs+perturb)
    loss.backward(torch.ones_like(loss))
    perturb.data += epsilon * torch.sign(perturb.grad.data)
    return (obs + perturb).detach()

def common_momentum_fgm(obs, epsilon, loss_fn, device, steps):
    """
    Implementation of Nesterov Momentum-FGM attacks 
    https://biases-invariances-generalization.github.io/pdf/big_33.pdf
    """
    obs_adv = obs.clone().detach().to(device)

    mu = 0.5
    v = torch.zeros_like(obs).to(device)
    lr = epsilon / steps

    for i in range(steps):
        _obs_adv = obs_adv.clone().detach().requires_grad_(True)
        loss = loss_fn(_obs_adv + mu * v)
        loss.backward(torch.ones_like(loss))
        gradients = _obs_adv.grad

        v = mu * v + gradients/torch.norm(gradients, p=1)
        obs_adv += v.sign().detach() * lr
        # print(obs_adv)
        obs_adv = torch.max(torch.min(obs_adv, obs + epsilon), obs - epsilon)
#         print("i", i, "adv", obs_adv[0]-obs[0])
        
    return obs_adv.detach()

def common_pgd(obs, epsilon, loss_fn, device, pgd_steps, lr, rand_init=False):
    """
    Implementation of PGD attacks, we want to maximize the loss function
    """
    obs_adv = obs.clone().detach().to(device)
    if rand_init:
        obs_adv += (2 * epsilon * torch.rand_like(obs).to(device) - epsilon)
    for i in range(pgd_steps):
        _obs_adv = obs_adv.clone().detach().requires_grad_(True)
        loss = loss_fn(_obs_adv)
        loss.backward(torch.ones_like(loss))
        gradients = _obs_adv.grad.sign().detach()
        obs_adv += gradients * lr
        obs_adv = torch.max(torch.min(obs_adv, obs + epsilon), obs - epsilon)

    return obs_adv.detach()

def noisy_pgd(obs, epsilon, loss_fn, device, pgd_steps, lr, rand_init=False):
    """
    Implementation of PGD attacks with step noises, we want to maximize the loss function
    (As defined by the MAD Attack https://arxiv.org/abs/2003.08938)
    """
    obs_adv = obs.clone().detach().to(device)
    if rand_init:
        obs_adv += (2 * epsilon * torch.rand_like(obs).to(device) - epsilon)
    for i in range(pgd_steps):
        _obs_adv = obs_adv.clone().detach().requires_grad_(True)
        loss = loss_fn(_obs_adv)
        loss.backward(torch.ones_like(loss))
        noise_factor = np.sqrt(2 * lr) / (i+2)
        update = _obs_adv.grad + noise_factor * torch.randn_like(obs)
        obs_adv += update.sign().detach() * lr
        obs_adv = torch.max(torch.min(obs_adv, obs + epsilon), obs - epsilon)

    return obs_adv.detach()

def dqn_dir_perturb_fgsm(victim, obs, direction, epsilon, device, cont=False):
    """
    Targeted attack: find the best obs attack in order to perturb the policy 
    to a target direction as much as possible
    """
    init = torch.zeros_like(obs).to(device)
    perturb = Variable(init, requires_grad=True)

    clean_policy = get_policy(victim, obs).probs.detach()
    policy = get_policy(victim, obs+perturb).probs
    diff = policy - clean_policy
    direction = direction.detach()
    cos_sim = nn.CosineSimilarity() 

    # loss = - torch.mean(cos_sim(diff, direction))
    
    loss = - torch.mean(cos_sim(diff, direction) + COEFF * torch.norm(diff, dim=1, p=2))

    # print("loss", torch.mean(torch.norm(unit_diff - direction, dim=1, p=2)), 
    #         torch.mean(torch.norm(diff, dim=1, p=2)))

    loss.backward()
    grad = perturb.grad.data
    perturb.data -= epsilon * torch.sign(grad)

    # print("clean", clean_policy)
    # print("direction", direction)
    # print("perturbed policy", perturbed_policy)

    return perturb.detach()

def get_policy(victim, obs):
    return torch.distributions.categorical.Categorical(logits=victim.Q(obs).squeeze())

def dqn_dir_perturb_momentum(victim, obs, direction, epsilon, device, cont=False, maxiter=10):
    """
    Targeted attack: find the best obs attack in order to perturb the policy 
    to a target direction as much as possible
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
    lr = epsilon / maxiter

    obs_adv = obs.clone().detach().to(device)
    for i in range(maxiter):
        _obs_adv = obs_adv.clone().detach().requires_grad_(True)
        loss = loss_fn(_obs_adv + mu * v)
        loss.backward(torch.ones_like(loss))
        gradients = _obs_adv.grad

        v = mu * v + gradients/torch.norm(gradients, p=1)
        obs_adv += v.sign().detach() * lr
        # print(obs_adv)
        obs_adv = torch.max(torch.min(obs_adv, obs + epsilon), obs - epsilon)
#         print("i", i, "adv", obs_adv[0]-obs[0])
        
    return obs_adv.detach() - obs.detach()

def dqn_dir_perturb_pgd(victim, obs, direction, epsilon, device, cont=False,
    maxiter=30, lr=1e-4, etol=1e-7, rand_init=False):
    """
    Targeted attack: find the best obs attack in order to perturb the policy 
    to a target direction as much as possible
    """
    clean_policy = get_policy(victim, obs).probs.detach()
    direction = direction.detach()
    cos_sim = nn.CosineSimilarity() 
    # # init = 0.1 * (2 * epsilon * torch.rand_like(obs).to(device) - epsilon)
    # init = torch.zeros_like(obs).to(device)
    # perturb = Variable(init, requires_grad=True)
    # optimizer = torch.optim.Adam([perturb], lr=lr)
    
    # old_loss = np.inf
    # for i in range(maxiter):
    #     policy = get_policy(victim, obs+perturb).probs
    #     diff = policy - clean_policy
    #     # unit_diff = diff / torch.norm(diff, dim=1, p=2, keepdim=True)
    #     loss = - torch.mean(cos_sim(diff, direction) + COEFF * torch.norm(diff, dim=1, p=2))
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #     with torch.no_grad():
    #         perturb[:] = torch.clamp(perturb, -epsilon, epsilon)
    #     if abs(loss.item()-old_loss) < etol:
    #         break
    #     old_loss = loss.item()
        
    # return perturb.detach()

    obs_adv = obs.clone().detach().to(device)
    if rand_init:
        obs_adv += (2 * epsilon * torch.rand_like(obs).to(device) - epsilon)
    for i in range(maxiter):
        _obs_adv = obs_adv.clone().detach().requires_grad_(True)
        policy = get_policy(victim, _obs_adv).probs
        diff = policy - clean_policy
        loss = - torch.mean(cos_sim(diff, direction) + COEFF * torch.norm(diff, dim=1, p=2))
        loss.backward()

        gradients = _obs_adv.grad.sign().detach()
        obs_adv -= gradients * lr
        obs_adv = torch.max(torch.min(obs_adv, obs + epsilon), obs - epsilon)

    return obs_adv.detach() - obs.detach()

### Attack proposed by Huang et al. in https://arxiv.org/pdf/1702.02284.pdf
### The goal is to seek the perturbed state to minimize the probability for
### the agent to take the best action
class Huang_Attack(Attacker):
    def __init__(self):
        self.name = 'Huang'
        
    def attack(self, q_network, obs, epsilon, pgd_steps=100, lr=1e-2, fgsm=False, 
            device=Param.device, rand_init=False, momentum=False):
        obs = torch.from_numpy(obs).to(device).type(Param.dtype).unsqueeze(0)
        obs.requires_grad = False
        q = q_network(obs)
        ce_loss = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            optimal_act = q.data.max(1)[1]

        def loss_fn(perturbed_obs):
            q = q_network(perturbed_obs)
            # we want to maximize this loss, i.e. the dissimilarity between the perturbed policy and the best policy
            loss = ce_loss(q, optimal_act)
            return loss
        
        if fgsm:
            return common_fgsm(obs, epsilon, loss_fn, device).cpu().numpy()
        elif momentum:
            return common_momentum_fgm(obs, epsilon, loss_fn, device, pgd_steps).cpu().numpy()
        else:
            # PGD
            return common_pgd(obs, epsilon, loss_fn, device, pgd_steps, lr, rand_init).cpu().numpy()
    
    def attack_torch(self, q_network, obs, epsilon, pgd_steps=100, lr=1e-2, fgsm=False,
                norm=np.inf, device=Param.device, rand_init=False, momentum=False):
        q = q_network(obs)
        ce_loss = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            optimal_act = q.data.max(1)[1]
        def loss_fn(perturbed_obs):
            q = q_network(perturbed_obs)
            # we want to maximize this loss, i.e. the dissimilarity between the perturbed policy and the best policy
            loss = ce_loss(q, optimal_act)
            return loss
        if fgsm:
            return common_fgsm(obs, epsilon, loss_fn, device)
        elif momentum:
            return common_momentum_fgm(obs, epsilon, loss_fn, device, pgd_steps)
        else:
            # PGD
            return common_pgd(obs, epsilon, loss_fn, device, pgd_steps, lr, rand_init)
    
    def attack_stoc(self, v_policy, obs, recurrent, masks, epsilon, fgsm=False, 
                pgd_steps=20, lr=1e-2, norm=np.inf, device=Param.device, 
                rand_init=False, momentum=False):

        ce_loss = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            clean_policy = v_policy.get_dist(obs, recurrent, masks).probs
            max_ind = torch.argmax(clean_policy, dim=1)
        
        def loss_fn(perturbed_obs):
            perturbed_policy = v_policy.get_dist(perturbed_obs, recurrent, masks).probs
            # we want to maximize this loss, i.e. the dissimilarity between the perturbed policy and the best policy
            loss = ce_loss(perturbed_policy, max_ind) 
            return loss
        if fgsm:
            return common_fgsm(obs, epsilon, loss_fn, device)
        elif momentum:
            return common_momentum_fgm(obs, epsilon, loss_fn, device, pgd_steps)
        else:
            # PGD
            return common_pgd(obs, epsilon, loss_fn, device, pgd_steps, lr, rand_init)
        
### Attack proposed by Pattanaik et al. in https://arxiv.org/pdf/1712.03632.pdf
### instead of decreasing the probability of the agent taking the best action,
### it optimize the probability of the agent taking the worst action
class Pattanaik_Attack(Attacker):
    def __init__(self):
        self.name = 'Pattanaik'
    
    def attack_torch(self, q_network, obs, epsilon, pgd_steps=100, lr=1e-2, fgsm=False,
        norm=np.inf, device=Param.device, rand_init=False, momentum=False):
        q = q_network(obs)
        with torch.no_grad():
            optimal_q = q.data.max(1)[0]
            worst_act = q.data.min(1)[1]
        
        beta_dist = Beta(torch.FloatTensor([2]).to(device), torch.FloatTensor([2]).to(device))
        ce_loss = torch.nn.CrossEntropyLoss()
            
        obs_var = obs.clone().detach().to(device).requires_grad_(True)
        loss = ce_loss(q_network(obs_var), worst_act)
        loss.backward()

        grad_dir = obs_var.grad/torch.norm(obs_var.grad)
        
        with torch.no_grad():
            s_adv = obs.clone()
            for i in range(pgd_steps):
                noise_factor = beta_dist.sample()
                s = obs - noise_factor * grad_dir
                s = torch.max(torch.min(s, obs + epsilon), obs - epsilon)
                # print("noise", noise_factor)
                new_q, new_act = q_network(s).data.max(1)
                # print("new", new_q, new_act)
                update_idx = new_q < optimal_q
                # print("update", update_idx)
                s_adv[update_idx] = s[update_idx].clone()
        # print("final", s_adv)
        return s_adv.detach()
    

class MaxWorst_Attack(Attacker):
    def __init__(self):
        self.name = 'MaxWorst'
    
    def attack(self, q_network, obs, epsilon, pgd_steps=100, lr=1e-2, fgsm=False, 
            device=Param.device, rand_init=False, momentum=False):
        obs = torch.from_numpy(obs).to(device).type(Param.dtype).unsqueeze(0)
        obs.requires_grad = False
        ce_loss = torch.nn.CrossEntropyLoss()
        q = q_network(obs)
        with torch.no_grad():
            worst_act = q.data.min(1)[1]

        def loss_fn(perturbed_obs):
            q = q_network(perturbed_obs)
            # we want to maximize this loss, i.e. the similarity between the perturbed policy and the worst policy
            loss = - ce_loss(q, worst_act)
            return loss

        if fgsm:
            return common_fgsm(obs, epsilon, loss_fn, device).cpu().numpy()
        elif momentum:
            return common_momentum_fgm(obs, epsilon, loss_fn, device, pgd_steps).cpu().numpy()
        else:
            # PGD
            return common_pgd(obs, epsilon, loss_fn, device, pgd_steps, lr, rand_init).cpu().numpy()
    
    def attack_torch(self, q_network, obs, epsilon, pgd_steps=100, lr=1e-2, fgsm=False,
        norm=np.inf, device=Param.device, rand_init=False, momentum=False):
        q = q_network(obs)
        ce_loss = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            worst_act = q.data.min(1)[1]

        def loss_fn(perturbed_obs):
            q = q_network(perturbed_obs)
            # we want to maximize this loss, i.e. the similarity between the perturbed policy and the worst policy
            loss = - ce_loss(q, worst_act)
            return loss

        if fgsm:
            return common_fgsm(obs, epsilon, loss_fn, device)
        elif momentum:
            return common_momentum_fgm(obs, epsilon, loss_fn, device, pgd_steps)
        else:
            # PGD
            return common_pgd(obs, epsilon, loss_fn, device, pgd_steps, lr, rand_init)
        
    def attack_stoc(self, v_policy, obs, recurrent, masks, epsilon, fgsm=False, 
        pgd_steps=100, lr=1e-2, norm=np.inf, device=Param.device, rand_init=False, 
        momentum=False):

        ce_loss = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            clean_policy = v_policy.get_dist(obs, recurrent, masks).probs
            min_ind = torch.argmin(clean_policy, dim=1)

        def loss_fn(perturbed_obs):
            perturbed_policy = v_policy.get_dist(perturbed_obs, recurrent, masks).probs
            # we want to maximize this loss, i.e. the similarity between the perturbed policy and the worst policy
            loss = - ce_loss(perturbed_policy, min_ind)
            return loss

        if fgsm:
            return common_fgsm(obs, epsilon, loss_fn, device)
        elif momentum:
            return common_momentum_fgm(obs, epsilon, loss_fn, device, pgd_steps)
        else:
            # PGD
            return common_pgd(obs, epsilon, loss_fn, device, pgd_steps, lr, rand_init)

class RandomWorst_Attack(Attacker):
    def __init__(self):
        self.name = 'RandWorst'
    
    def attack_stoc(self, v_policy, obs, recurrent, masks, epsilon, fgsm=False, 
        pgd_steps=100, lr=1e-2, norm=np.inf, device=Param.device, rand_init=False, 
        momentum=False):

        ce_loss = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            clean_policy = v_policy.get_dist(obs, recurrent, masks).probs 
            targ_arr = np.array([np.random.choice(list(range(clean_policy.size()[1]))) for i in range(clean_policy.size()[0])])
            # print(targ_arr)
            targ_ind = torch.LongTensor(targ_arr).to(device)
            # print(targ_ind)

        def loss_fn(perturbed_obs):
            perturbed_policy = v_policy.get_dist(perturbed_obs, recurrent, masks).probs
            # we want to maximize this loss, i.e. the similarity between the perturbed policy and the worst policy
            loss = - ce_loss(perturbed_policy, targ_ind)
            return loss

        if fgsm:
            return common_fgsm(obs, epsilon, loss_fn, device)
        elif momentum:
            return common_momentum_fgm(obs, epsilon, loss_fn, device, pgd_steps)
        else:
            # PGD
            return common_pgd(obs, epsilon, loss_fn, device, pgd_steps, lr, rand_init)

class TargetWorst_Attack(Attacker):
    def __init__(self, target):
        self.name = 'TargetWorst'
        self.target_policy = target
    
    def attack_stoc(self, v_policy, obs, recurrent, masks, epsilon, fgsm=False, 
        pgd_steps=100, lr=1e-2, norm=np.inf, device=Param.device, rand_init=False, 
        momentum=False):

        ce_loss = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            _, targ_ind, _, _ = self.target_policy.act(
                obs, recurrent, masks, beta=False, deterministic=True) 
            targ_ind = targ_ind.flatten()
        def loss_fn(perturbed_obs):
            perturbed_policy = v_policy.get_dist(perturbed_obs, recurrent, masks).probs
            # we want to maximize this loss, i.e. the similarity between the perturbed policy and the worst policy
            loss = - ce_loss(perturbed_policy, targ_ind)
            return loss

        if fgsm:
            return common_fgsm(obs, epsilon, loss_fn, device)
        elif momentum:
            return common_momentum_fgm(obs, epsilon, loss_fn, device, pgd_steps)
        else:
            # PGD
            return common_pgd(obs, epsilon, loss_fn, device, pgd_steps, lr, rand_init)

### Attack mentioned by Zhang et al. in https://arxiv.org/pdf/2003.08938.pdf
### Instead of minimizing the best action probability or maximize the worst action 
### probability, it instead minimize Q(s,\pi(s')), which is supposed to be the optimal
### attack under the assumption that only one single state could be perturbed at a time
class SA_Attack(Attacker):
    def __init__(self):
        self.name = 'SA'
    
    '''
        To minimize Q(s,pi(s')), we use projected gradient descent, and also
        assume that pi(s') is a softmax policy in order to calculate the gradient.
        k here is the number of interations in PGD, default: k=10
    '''
    def attack(self, q_network, obs, epsilon, pgd_steps=100, lr=1e-2):
        obs = torch.from_numpy(obs).to(Param.device).type(Param.dtype).unsqueeze(0)
        obs.requires_grad = False
        init = torch.zeros_like(obs)
        # init = 2 * epsilon * torch.rand_like(obs) - epsilon * torch.ones_like(obs)
        perturb = Variable(init.to(device).type(Param.dtype), requires_grad=True)

        for i in range(pgd_steps):
            ### Compute Q function of perturbed state s'
            q = q_network(obs+perturb)
            act = torch.nn.Softmax(-1)(q)
            loss = torch.dot(q.squeeze(), act.squeeze())
            ### Calculate the gradient
            loss.backward(torch.ones_like(loss))
            grad = perturb.grad.data
            perturb.data -= lr*grad
            perturb.data = perturb.data.clamp(-epsilon, epsilon)
            grad.zero_()
        return (obs + perturb).detach().cpu().numpy()
    
    
### Implement the attack for maximizing the KL divergence between perturbed policy
### and the original policy
class KL_Attack(Attacker):
    def __init__(self):
        self.name = 'KL'
    
    '''
        To minimize KL(pi(a|s), pi(a|s')), we use projected gradient descent, and also
        assume that pi(s') is a softmax policy in order to calculate the gradient.
        k here is the number of interations in PGD, default: k=10
    '''
    def attack(self, q_network, obs, epsilon, pgd_steps=100, lr=5e-5, device=Param.device, 
            rand_init=True, momentum=False):
        obs = torch.from_numpy(obs).to(Param.device).type(Param.dtype).unsqueeze(0)
        obs.requires_grad = False
        # perturb = Variable(torch.zeros_like(obs).to(Param.device).type(Param.dtype), requires_grad=True)
        # init = 2 * epsilon * torch.rand_like(obs) - epsilon * torch.ones_like(obs)
        # perturb = Variable(init.to(device).type(Param.dtype), requires_grad=True)
        ### Compute pi(a|s), and pi(a|s')
        with torch.no_grad():
            q_original  = act = q_network(obs)
            original_dist  = Categorical(logits=q_original.squeeze())

        def loss_fn(perturbed_obs):
            q_perturbed = q_network(perturbed_obs)
            perturbed_dist = Categorical(logits=q_perturbed.squeeze())
            # we want to maximize the distance between the original policy and the pertorbed policy
            loss = kl_divergence(original_dist, perturbed_dist)
            return loss

        if fgsm:
            return common_fgsm(obs, epsilon, loss_fn, device).cpu().numpy()
        elif momentum:
            return common_momentum_fgm(obs, epsilon, loss_fn, device, pgd_steps).cpu().numpy()
        else:
            # PGD
            return common_pgd(obs, epsilon, loss_fn, device, pgd_steps, lr, rand_init).cpu().numpy()

    
    def attack_torch(self, q_network, obs, epsilon, fgsm=False, pgd_steps=100, lr=1e-2, 
            norm=np.inf, device=Param.device, rand_init=True, momentum=False):
        # init = torch.zeros_like(obs)
        init = 2 * epsilon * torch.rand_like(obs) - epsilon * torch.ones_like(obs)
        perturb = Variable(init.to(device), requires_grad=True)
        ### Compute pi(a|s), and pi(a|s')
        with torch.no_grad():
            q_original = q_network(obs)
            original_dist  = Categorical(logits=q_original.squeeze())
        
        def loss_fn(perturbed_obs):
            q_perturbed = q_network(perturbed_obs)
            perturbed_dist = Categorical(logits=q_perturbed.squeeze())
            # we want to maximize the distance between the original policy and the pertorbed policy
            loss = kl_divergence(original_dist, perturbed_dist)
            return loss

        if fgsm:
            return common_fgsm(obs, epsilon, loss_fn, device) 
        elif momentum:
            return common_momentum_fgm(obs, epsilon, loss_fn, device, pgd_steps)
        else:
            # PGD
            return common_pgd(obs, epsilon, loss_fn, device, pgd_steps, lr, rand_init) 

        # for i in range(pgd_steps):
        #     q_perturbed = q_network(obs+perturb)
        #     perturbed_dist = Categorical(logits=q_perturbed.squeeze())
        #     loss = kl_divergence(original_dist, perturbed_dist)
            
        #     #if (i%50==0):
        #          #print("KL divergence:{}".format(loss))
            
        #     ### Calculate the gradient, and maximize the kl divergence
        #     loss.backward(torch.ones_like(loss))
        #     grad = perturb.grad.data
        #     perturb.data += lr*grad
        #     perturb.data = perturb.data.clamp(-epsilon, epsilon)
        #     #if (i%50==0):
        #         #print("perturb.data:{}".format(perturb.data))
        #         #print("grad:{}".format(grad))
        #     grad.zero_()
        # # print("perturb", perturb.data)
        # return (obs + perturb).detach()
    
    def attack_stoc(self, v_policy, obs, recurrent, masks, epsilon, fgsm=False, 
        pgd_steps=100, lr=1e-2, norm=np.inf, device=Param.device, rand_init=True, 
        momentum=False):
        
        
        with torch.no_grad():
            clean_policy = v_policy.get_dist(obs, recurrent, masks)

        def loss_fn(perturbed_obs):
            perturbed_policy = v_policy.get_dist(perturbed_obs, recurrent, masks)
            # we want to maximize the distance between the original policy and the pertorbed policy
            loss = kl_divergence(clean_policy, perturbed_policy)
            return loss

        if fgsm:
            return common_fgsm(obs, epsilon, loss_fn, device) 
        elif momentum:
            return common_momentum_fgm(obs, epsilon, loss_fn, device, pgd_steps)
        else:
            # PGD
            return noisy_pgd(obs, epsilon, loss_fn, device, pgd_steps, lr, rand_init)

class RS_Attack(Attacker):
    def __init__(self, rs_qnet, action_space):
        self.name = 'RS'
        self.q_net = rs_qnet
        if action_space.__class__.__name__ == "Discrete":
            self.act_dim = action_space.n
        elif action_space.__class__.__name__ == "Box":
            self.act_dim = action_space.shape[0]
    
    def attack(self, q_network, obs, epsilon, pgd_steps=100, lr=1e-2, fgsm=False, 
            device=Param.device, rand_init=False, momentum=False):
        obs = torch.from_numpy(obs).to(device).type(Param.dtype).unsqueeze(0)
        obs.requires_grad = False

        def loss_fn(perturbed_obs):
            policy_action = q_network(perturbed_obs).max(1)[1]
            real_q = self.q_net(torch.cat((obs, policy_action), dim=-1))
            # we want to maximize this loss, i.e. the similarity between the perturbed policy and the worst policy
            loss = - real_q
            return loss

        if fgsm:
            return common_fgsm(obs, epsilon, loss_fn, device).cpu().numpy()
        elif momentum:
            return common_momentum_fgm(obs, epsilon, loss_fn, device, pgd_steps)
        else:
            # PGD
            return common_pgd(obs, epsilon, loss_fn, device, pgd_steps, lr, rand_init).cpu().numpy()
    
    def attack_torch(self, q_network, obs, epsilon, pgd_steps=100, lr=1e-2, fgsm=False,
        norm=np.inf, device=Param.device, rand_init=False, momentum=False):

        def loss_fn(perturbed_obs):
            policy_action = q_network(perturbed_obs).max(1)[1]
            print("policy action", policy_action)
            real_q = self.q_net(torch.cat((obs, policy_action), dim=-1))
            # we want to minimize the real Q value of the selected action
            loss = - real_q
            return loss

        if fgsm:
            return common_fgsm(obs, epsilon, loss_fn, device)
        elif momentum:
            return common_momentum_fgm(obs, epsilon, loss_fn, device, pgd_steps)
        else:
            # PGD
            return common_pgd(obs, epsilon, loss_fn, device, pgd_steps, lr, rand_init)
    
    def attack_stoc(self, v_policy, obs, recurrent, masks, epsilon, fgsm=False, 
        pgd_steps=100, lr=1e-2, norm=np.inf, device=Param.device, rand_init=False, 
        momentum=False):

        clean_policy = v_policy.get_dist(obs, recurrent, masks)

        def loss_fn(perturbed_obs):
            policy_action = v_policy.get_dist(perturbed_obs, recurrent, masks).mode()
            # print("policy action", policy_action)
            # onehot_action = torch.nn.functional.one_hot(policy_action.squeeze(), self.act_dim)
            # print("onehot action", onehot_action)
            real_q = self.q_net(torch.cat((obs, policy_action), dim=-1))
            # we want to minimize the real Q value of the selected action
            loss = - real_q
            return loss

        if fgsm:
            return common_fgsm(obs, epsilon, loss_fn, device)
        elif momentum:
            return common_momentum_fgm(obs, epsilon, loss_fn, device, pgd_steps)
        else:
            # PGD
            return common_pgd(obs, epsilon, loss_fn, device, pgd_steps, lr, rand_init)

class Obs_Attack(Attacker):
    def __init__(self, adv_policy, action_shape, det=False):
        self.name = 'Obs'
        self.adv_policy = adv_policy
        self.action_shape = action_shape
        self.det = det
    
    def attack(self, q_network, obs, epsilon):
        raise NotImplementedError
        
    def attack_torch(self, obs, recurrent, masks, epsilon, device=Param.device):
        with torch.no_grad():
            _, action, _, recurrent_hidden_states = self.adv_policy.act(
                obs, recurrent, masks, beta=False, deterministic=self.det)
        if len(self.action_shape) == 3:  # is atari input
            perturb = action.view(-1, self.action_shape[0], 
                    self.action_shape[1], self.action_shape[2])
        else:
            perturb = action
        perturb = perturb.clamp(-epsilon, epsilon)
        
        return (obs + perturb).detach(), recurrent_hidden_states
    
    def attack_stoc(self, obs, recurrent, masks, epsilon, device=Param.device):
        with torch.no_grad():
            _, action, _, recurrent_hidden_states = self.adv_policy.act(
                obs, recurrent, masks, beta=False, deterministic=self.det)
        if len(self.action_shape) == 3:  # is atari input
            perturb = action.view(-1, self.action_shape[0], 
                    self.action_shape[1], self.action_shape[2])
            perturb = perturb.clamp(-epsilon, epsilon)
        else:
            perturb = action
            perturb = action.clamp(-epsilon, epsilon)
        
        return (obs + perturb).detach()

class ObsPol_Attack(Attacker):
    def __init__(self, adv_policy, obs_attacker=None, det=False, cont=False):
        self.name = 'ObsPol'
        self.adv_policy = adv_policy
        self.obs_attacker = obs_attacker
        self.det = det
        self.cont = cont
        
    def attack(self, q_network, obs, epsilon):
        raise NotImplementedError

    def attack_torch(self, victim, obs, recurrent, masks, epsilon, fgsm=False, 
        pgd_steps=100, lr=1e-2, norm=np.inf, device=Param.device, rand_init=False, 
        momentum=False):
        with torch.no_grad():
            _, action, _, recurrent_hidden_states = self.adv_policy.act(
                obs, recurrent, masks, beta=False, deterministic=self.det)

        if self.cont:
            perturb_direction = action
        else:
            perturb_direction = torch.cat((action, -torch.sum(action, dim=1, keepdim=True)), 1)
        
        if self.obs_attacker != None:
            perturb = self.obs_attacker.perturb_batch(obs, perturb_direction)
        elif fgsm:
            perturb = dqn_dir_perturb_fgsm(victim, obs, perturb_direction, 
                    epsilon, device, cont=self.cont)
        elif momentum:
            perturb = dqn_dir_perturb_momentum(v_policy, obs, recurrent, masks, 
                    perturb_direction, epsilon, device, cont=self.cont,
                    maxiter=pgd_steps)
        else:
            perturb = dqn_dir_perturb_pgd(victim, obs, perturb_direction, 
                    epsilon, device, cont=self.cont, maxiter=pgd_steps, lr=lr, rand_init=rand_init)
                    
        return (obs + perturb).detach(), recurrent_hidden_states
    
    def attack_stoc(self, v_policy, obs, recurrent, masks, epsilon, fgsm=False, 
        pgd_steps=100, lr=1e-2, norm=np.inf, device=Param.device, rand_init=False, 
        momentum=False):

        with torch.no_grad():
            _, action, _, recurrent_hidden_states = self.adv_policy.act(
                obs, recurrent, masks, beta=False, deterministic=self.det)

        if self.cont:
            perturb_direction = action
        else:
            perturb_direction = torch.cat((action, -torch.sum(action, dim=1, keepdim=True)), 1)
        
        if self.obs_attacker != None:
            perturb = self.obs_attacker.perturb_batch(obs, perturb_direction)
        elif fgsm:
            perturb = obs_dir_perturb_fgsm(v_policy, obs, recurrent, masks, 
                    perturb_direction, epsilon, device, cont=self.cont)
        elif momentum:
            perturb = obs_dir_perturb_momentum(v_policy, obs, recurrent, masks, 
                    perturb_direction, epsilon, device, cont=self.cont,
                    maxiter=pgd_steps)
        else:
            perturb = obs_dir_perturb_pgd(v_policy, obs, recurrent, masks, 
                    perturb_direction, epsilon, device, cont=self.cont, 
                    maxiter=pgd_steps, lr=lr, rand_init=rand_init)
                    
        return (obs + perturb).detach()

class Random_Attack(Attacker):
    def __init__(self):
        self.name = 'Random'
    
    def attack(self, q_network, obs, epsilon):
        raise NotImplementedError
        
    def attack_torch(self, obs, epsilon, fgsm=False, device=Param.device):
        perturb = (2 * epsilon * torch.rand_like(obs) - epsilon * torch.ones_like(obs)).to(device)

        if fgsm:
            perturb = epsilon * torch.sign(perturb)
        
        return (obs + perturb).detach()
    
    def attack_stoc(self, obs, epsilon, fgsm=False, device=Param.device):
        
        perturb = (2 * epsilon * torch.rand_like(obs) - epsilon * torch.ones_like(obs)).to(device)

        if fgsm:
            perturb = epsilon * torch.sign(perturb)
        
        return (obs + perturb).detach()
        

### Created specifically for the test MDP
class Test_Attack_Policy:
    def __init__(self, env, agent, attacker, eps=0.05, states_attack=[0,1], num_states=2):
        self.q = agent.Q
        self.env = env
        self.agent = agent
        self.attacker = attacker
        self.eps = eps
        self.states_attack = states_attack
        self.num_states = num_states
        
    ### Return the softmax of the perturbed policy
    def policy(self):
        policy = np.zeros((2,3))
        obs = self.env.mdp.obs
        for i in range(self.num_states):
            if i in self.states_attack:
                perturbed_states = self.attacker.attack(self.q, obs[i], self.eps, lr=1e-2, pgd_steps=1000)
                print(obs[i], "perturb to", perturbed_states)
                q = self.q(torch.from_numpy(perturbed_states).to(Param.device).type(Param.dtype).unsqueeze(1))
                policy[i,:]=torch.nn.Softmax(-1)(q).detach().cpu().numpy()
            else:
                # act = int(self.agent.step(perturbed_states))
                # policy[i, act] = 1.0
                perturbed_states = self.attacker.attack(self.q, obs[i], 0, pgd_steps=0)
                q = self.q(torch.from_numpy(perturbed_states).to(Param.device).type(Param.dtype).unsqueeze(1))
                policy[i,:]=torch.nn.Softmax(-1)(q).detach().cpu().numpy()
        print("policy", policy)
        return policy
    
### Attack based on each single state, as apposed to SA-MDP approach
### num_trajectories: number of test trajectories
### max_steps: maximum steps in a trajectory
### pgd_steps: number of PGD iterations for every state
def single_state_attack(env_name, attacker, agent, pgd_steps=100, lr=5e-3,
                        epsilon=0.05, num_trajectories=2000, max_steps=10):
    if (env_name == 'Test'):
        from WocaR_DQN.envs.test_env import TestEnv
        env = TestEnv()
    else:
        env = gym.make(env_name)
    
    rews = []
    for i in range(num_trajectories):
        obs = env.reset()
        total_rew = 0
        for t in range(max_steps):
            perturbed_states = attacker.attack(agent.Q, obs, epsilon, pgd_steps=pgd_steps, lr=lr)
            a = int(agent.step(perturbed_states))
            (obs, reward, done, _info) = env.step(a)
            total_rew += reward
            if done: 
                #print("early stop at step:{}".format(t))
                break
        rews.append(total_rew)
        if i%10 == 0:
            print("Finishing {} Trajectories".format(i+1))
    ### Rollout Normal Agent
    clean_reward,_ = rollout(agent, env, num_trajectories, max_steps)
    ### Average Pertubed Reward
    perturbed_reward = sum(rews)/len(rews)
    
    print("------------------Attacker:{}--------------------".format(attacker.name))
    print("Mean Clean Rewards:{:.2f}".format(clean_reward))
    print("Mean Perturbed Rewards:{:.2f}".format(perturbed_reward))
    print("Std:{}".format(np.std(perturbed_reward)))
    return perturbed_reward

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='Pattanaik')
    parser.add_argument('--env', type=str, default='Test')
    parser.add_argument('--eps', type=float, default=0.02)
    parser.add_argument('--n_iter', type=int, default=100)
    parser.add_argument('--max_steps', type=int, default=1000)
    parser.add_argument('--num_trajectories', type=int, default=500)
    parser.add_argument('-no_cuda', action="store_true")
    args = parser.parse_args()
    
    if args.no_cuda:
        Param(torch.FloatTensor, torch.device("cpu"))
    else:
        Param(torch.cuda.FloatTensor, torch.device("cuda:0"))

    if (args.env=='Test'):
        agent = DQN_Agent(Q_Test)
        path = os.path.join(Param.model_dir, './dqn/test_mdp')
    else: 
        agent = DQN_Agent(Q_Lunar)
        path = os.path.join(Param.model_dir, './dqn/lunar_lander_doubleQ')
    agent.load_state_dict(torch.load(path, map_location=Param.device))
    
    if (args.method=='Huang'):
        attacker = Huang_Attack()
        single_state_attack(args.env, attacker, agent, pgd_steps = args.n_iter, lr=LR_HUANG_ATTACKER, 
                            max_steps=args.max_steps, num_trajectories=args.num_trajectories)
    elif (args.method == 'Pattanaik'):
        attacker = Pattanaik_Attack()
        single_state_attack(args.env, attacker, agent, pgd_steps = args.n_iter, lr=LR_PATTANAIK_ATTACKER, 
                            max_steps=args.max_steps, num_trajectories=args.num_trajectories)
    elif (args.method == 'SA'):
        attacker = SA_Attack()
        single_state_attack(args.env, attacker, agent, pgd_steps = args.n_iter, lr=LR_SA_ATTACKER, 
                            max_steps=args.max_steps, num_trajectories=args.num_trajectories)
    elif (args.method == 'KL'):
        attacker = KL_Attack()
        single_state_attack(args.env, attacker, agent, pgd_steps = args.n_iter, lr=LR_KL_ATTACKER, 
                            max_steps=args.max_steps, num_trajectories=args.num_trajectories)