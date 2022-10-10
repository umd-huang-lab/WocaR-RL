import torch
import torch.nn as nn
import torch.optim as optim
import functools

import numpy as np

from VaR_DQN.a2c_ppo_acktr.algo.kfac import KFACOptimizer

"""Computing an estimated upper bound of KL divergence using SGLD."""
"""convex relaxation method needs relaxed actor_critic"""
def get_state_kl_bound_sgld(actor_critic, rollouts, args, eps, stdev):
    steps = args.robust_pgd_steps
    num_steps, num_processes, _ = rollouts.rewards.size()

    obs_shape = rollouts.obs.size()[2:]
    action_shape = rollouts.actions.size()[-1]
    recurrent_hidden_state_size = actor_critic.recurrent_hidden_state_size

    # upper and lower state bounds for clipping with current obs
    obs_ub = rollouts.obs + eps
    obs_lb = rollouts.obs - eps
    step_eps = eps / steps

    # SGLD noise factor. We set (inverse) beta=1e-5 as gradients are relatively small here.
    beta = 1e-5
    noise_factor = np.sqrt(2 * step_eps * beta)
    noise = torch.randn_like(rollouts.obs) * noise_factor

    # calculate actions
    with torch.no_grad():
        _, actions, _, _ = actor_critic.act(rollouts.obs[:-1].view(-1, *obs_shape),
                rollouts.recurrent_hidden_states[0].view(-1, recurrent_hidden_state_size),
                rollouts.masks[:-1].view(-1, 1))

    actions = actions.view(num_steps, num_processes, action_shape)
    var_obs = (rollouts.obs.clone() + noise.sign() * step_eps).detach().requires_grad_()

    for i in range(steps):
        # Find a nearby obs that maximize the action difference
        var_obs.retain_grad()
        with torch.no_grad():
            _, var_actions, _, _ = actor_critic.act(var_obs[:-1].view(-1, *obs_shape),
                    rollouts.recurrent_hidden_states[0].view(-1, recurrent_hidden_state_size),
                    rollouts.masks[:-1].view(-1, 1))

        var_actions = var_actions.view(num_steps, num_processes, action_shape)
        
        diff = (var_actions - actions) / stdev.detach()
        kl = (diff * diff).sum(axis=-1, keepdim=True).mean().requires_grad_()
        # Need to clear gradients before the backward() for policy_loss
        kl.backward()
        # Reduce noise at every step.
        noise_factor = np.sqrt(2 * step_eps * beta) / (i+2)
        # Project noisy gradient to step boundary.
        var_obs.data += noise_factor * torch.randn_like(var_obs).sign() * step_eps
        # clip into the upper and lower bounds
        var_obs = torch.max(var_obs, obs_lb)
        var_obs = torch.min(var_obs, obs_ub)
        var_obs = var_obs.detach().requires_grad_()

    actor_critic.zero_grad()

    _, var_actions, _, _ = actor_critic.act(var_obs[:-1].view(-1, *obs_shape).detach(),
            rollouts.recurrent_hidden_states[0].view(-1, actor_critic.recurrent_hidden_state_size),
            rollouts.masks[:-1].view(-1, 1))
    var_actions = var_actions.view(num_steps, num_processes, action_shape)
    diff = (var_actions - actions) / stdev

    return (diff * diff).sum(axis=-1, keepdim=True)



class ROBUST_A2C_ACKTR():
    def __init__(self,
                 actor_critic,
                 value_loss_coef,
                 entropy_coef,
                 act_dim,
                 lr=None,
                 eps=None,
                 alpha=None,
                 max_grad_norm=None,
                 acktr=False,
                 beta=False):

        self.actor_critic = actor_critic
        self.acktr = acktr
        self.beta = beta

        stdev_init = torch.zeros(act_dim)
        self.log_stdev = torch.nn.Parameter(stdev_init)

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm

        if acktr:
            self.optimizer = KFACOptimizer(actor_critic)
        else:
            self.optimizer = optim.RMSprop(
                actor_critic.parameters(), lr, eps=eps, alpha=alpha)

    def robust_update(self, rollouts, args, eps):
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()
        
        stdev = torch.exp(self.log_stdev)
        if args.robust_detach_stdev:
            # Detach stdev so that it won't be too large.
            stdev = stdev.detach()

        values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
            rollouts.obs[:-1].view(-1, *obs_shape),
            rollouts.recurrent_hidden_states[0].view(
                -1, self.actor_critic.recurrent_hidden_state_size),
            rollouts.masks[:-1].view(-1, 1),
            rollouts.actions.view(-1, action_shape),
            beta=self.beta)
            
        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        advantages = rollouts.returns[:-1] - values
        value_loss = advantages.pow(2).mean()

        if args.robust_method == "sgld":
            kl_upper_bound = get_state_kl_bound_sgld(self.actor_critic, rollouts, args, eps, stdev).mean()
        else:
            raise ValueError(f"Unsupported robust method {args.robust_method}")

        reg_loss = args.robust_reg * kl_upper_bound.detach()
        action_loss = -(advantages.detach() * action_log_probs).mean() + reg_loss

        if self.acktr and self.optimizer.steps % self.optimizer.Ts == 0:
            # Compute fisher, see Martens 2014
            self.actor_critic.zero_grad()
            pg_fisher_loss = -action_log_probs.mean()

            value_noise = torch.randn(values.size())
            if values.is_cuda:
                value_noise = value_noise.to(values.device)

            sample_values = values + value_noise
            vf_fisher_loss = -(values - sample_values.detach()).pow(2).mean()

            fisher_loss = pg_fisher_loss + vf_fisher_loss
            self.optimizer.acc_stats = True
            fisher_loss.backward(retain_graph=True)
            self.optimizer.acc_stats = False

        self.optimizer.zero_grad()
        (value_loss * self.value_loss_coef + action_loss -
         dist_entropy * self.entropy_coef).backward()

        if self.acktr == False:
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                     self.max_grad_norm)

        self.optimizer.step()

        return value_loss.item(), action_loss.item(), dist_entropy.item()




