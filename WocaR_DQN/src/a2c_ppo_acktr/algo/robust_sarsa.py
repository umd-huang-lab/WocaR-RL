import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from auto_LiRPA import BoundedModule
from auto_LiRPA.eps_scheduler import LinearScheduler
from auto_LiRPA.bounded_tensor import BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    """
    Performs an in-place polyak update of the target module based on the source,
    by a ratio of tau. Note that source and target modules must have the same
    parameters, where:
        target = tau * source + (1-tau) * target
    :param source: Source module whose parameters will be used.
    :param target: Target module whose parameters will be updated.
    :param tau: Percentage of source parameters to use in average. Setting tau to
        1 will copy the source parameters to the target.
    """
    with torch.no_grad():
        for source_param, target_param in zip(
            source.parameters(), target.parameters()
        ):
            target_param.data.mul_(1.0 - tau)
            torch.add(
                target_param.data,
                source_param.data,
                alpha=tau,
                out=target_param.data,
            )

class RobustSarsa():
    def __init__(self,
                 q_net,
                 target_q,
                 gamma,
                 num_steps,
                 num_features,
                 num_actions,
                 lr=1e-3,
                 max_grad_norm=0.5,
                 reg=0.1,
                 eps=0.02):

        self.q_net = q_net
        self.target_q = target_q
        self.gamma = gamma
        self.lr = lr 
        self.num_features = num_features
        self.num_actions = num_actions
        self.max_grad_norm = max_grad_norm
        self.l1loss = torch.nn.SmoothL1Loss()
        self.num_steps = num_steps
        print("num", num_steps)

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        # learning rate scheduler: linearly annealing learning rate after 
        lr_decrease_point = num_steps * 2 / 3
        decreasing_steps = num_steps - lr_decrease_point
        lr_sch = lambda epoch: 1.0 if epoch < lr_decrease_point else (decreasing_steps - epoch + lr_decrease_point) / decreasing_steps
        # robust training scheduler. Currently using 1/3 epochs for warmup, 1/3 for schedule and 1/3 for final training.
        eps_start_point = int(num_steps * 1 / 3)
        # lr_sch = lambda epoch: max(0.01, 0.99 ** epoch)
        self.sarsa_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_sch)
        
        # self.sarsa_eps = lambda step: max(0.0, (step-1/num_steps) * eps)
        # self.sarsa_beta = lambda step: max(0.0, (step-1/num_steps) * 1.0)
        
        # Convert model with relaxation wrapper.
        dummy_input = torch.randn(1, self.num_features+ self.num_actions)
        self.relaxed_sarsa_model = BoundedModule(self.q_net, dummy_input)
        self.target_relaxed_sarsa_model = BoundedModule(self.target_q, dummy_input)
        self.sarsa_reg = reg
        self.step = 0
        self.eps_max = eps
        self.beta_max = 1.0
        
        soft_update(self.q_net, self.target_q, 1.0)
    
    def sarsa_eps(self):
        if self.step < self.num_steps * 0.1:
            return 0
        eps = self.eps_max * (self.step-self.num_steps*0.1) / self.num_steps 
        return eps
    
    def sarsa_beta(self):
        if self.step < self.num_steps * 0.1:
            return 0
        beta = self.beta_max * (self.step-self.num_steps*0.1) / self.num_steps 
        return beta
    
    def update_old(self, samples):
        obs_shape = samples.obs.size()[2:]
        action_shape = samples.action.size()[-1]
        num_steps, num_processes, _ = samples.reward.size()

        q_old = self.q_net(samples.obs, samples.action)
        q_next = self.q_net(samples.obs_next, samples.action_next)
        self.optimizer.zero_grad()
        loss_q = self.q_loss(samples.reward + self.gamma * q_next, q_old)
        loss_q.backward()
        self.optimizer.step()

        loss = loss_q.item()
        return loss

    def update(self, samples):
        # obs_shape = samples.obs.size()[2:]
        # action_shape = samples.action.size()[-1]
        num_steps, num_processes, _ = samples.reward.size()
        num_samples = num_steps * num_processes
        obs_all = samples.obs.view(num_samples, -1)
        action_all = samples.action.view(num_samples, -1)
        reward_all = samples.reward.view(num_samples, -1)
        done_all = samples.done.view(num_samples, -1)
        next_obs_all = samples.obs_next.view(num_samples, -1)
        next_action_all = samples.action_next.view(num_samples, -1)

#         q_old = self.q_net(torch.cat((obs_all, action_all), dim=-1))
#         q_next = self.target_q(torch.cat((next_obs_all, next_action_all), dim=-1)).detach()
#         self.optimizer.zero_grad()
#         # print(done_all.size())
#         # print((self.gamma * q_next * done_all).size())
#         # print("a", self.gamma * q_next)
#         # print("b", self.gamma * q_next * done_all)
#         loss_q = self.l1loss(reward_all + self.gamma * q_next * done_all, q_old)
#         print("q old", q_old.flatten()[:10])
#         print("q next", (reward_all + self.gamma * q_next * done_all).flatten()[:10])
        
#         loss_q.backward()
#         self.optimizer.step()
        
#         soft_update(self.q_net, self.target_q, 0.05)

#         loss = loss_q.item()
#         # self.sarsa_scheduler.step()
#         # print("loss", loss)
#         return loss

        self.step += 1
        eps = self.sarsa_eps()
        beta = self.sarsa_beta()
        print("eps", eps, "beta", beta, "lr", self.sarsa_scheduler.get_last_lr())
        
        reward_all = reward_all

        batch_size = 64
        for i in range(10):
            state_indices = np.arange(num_samples)
            np.random.shuffle(state_indices)
            splits = np.array_split(state_indices, 10)
            
            for idx in splits:
    #             perm = torch.randperm(num_samples)
    #             idx = perm[:batch_size]
                obs = obs_all[idx]
                action = action_all[idx]
                reward = reward_all[idx]
                done = done_all[idx]
                next_obs = next_obs_all[idx]
                next_action = next_action_all[idx]

                self.optimizer.zero_grad()

                inputs = torch.cat((obs, action), dim=-1)
                bounded_inputs = BoundedTensor(inputs, ptb=PerturbationLpNorm(norm=np.inf, eps=eps))

                q_old = self.relaxed_sarsa_model(bounded_inputs).squeeze(-1)
                inputs_n = torch.cat((next_obs, next_action), dim=-1)
                bounded_inputs_n = BoundedTensor(inputs_n, ptb=PerturbationLpNorm(norm=np.inf, eps=eps))

                q_next = self.relaxed_sarsa_model(bounded_inputs_n).squeeze(-1) \
                    * self.gamma * done.squeeze(-1) + reward.squeeze(-1)

                q_next = q_next.detach()
    #             print(self.gamma)
    #             print("rews", reward.squeeze(-1))
                # q_next = (self.q_net(inputs_n) * self.gamma + reward_all).detach()

                q_loss = self.l1loss(q_old, q_next)
                # Compute the robustness regularization.

                if eps > 0 and self.sarsa_reg > 0:
                    ilb, iub = self.relaxed_sarsa_model.compute_bounds(IBP=True, method=None)
                    if beta < 1:
                        clb, cub = self.relaxed_sarsa_model.compute_bounds(IBP=False, method='backward')
                        lb = beta * ilb + (1 - beta) * clb
                        ub = beta * iub + (1 - beta) * cub
                    else:
                        lb = ilb
                        ub = iub
                    # Output dimension is 1. Remove the extra dimension and keep only the batch dimension.
                    lb = lb.squeeze(-1)
                    ub = ub.squeeze(-1)
                    diff = torch.max(ub - q_old, q_old - lb)
                    reg_loss = self.sarsa_reg * (diff * diff).mean()
                    sarsa_loss = q_loss + reg_loss
                    reg_loss = reg_loss.item()
                else:
                    reg_loss = 0.0
                    sarsa_loss = q_loss
                sarsa_loss.backward()
                self.optimizer.step()
           
        print("rews", reward.squeeze(-1))
        print("q old", q_old)
        print("q next", q_next)
        print(f'q_loss={q_loss.item():.6g}, reg_loss={reg_loss:.6g}, sarsa_loss={sarsa_loss.item():.6g}')

        self.sarsa_scheduler.step()
        
        soft_update(self.q_net, self.target_q, 0.05)

        return sarsa_loss.item()
