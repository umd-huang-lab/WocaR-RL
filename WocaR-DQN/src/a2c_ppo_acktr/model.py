import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from WocaR_DQN.a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian, Beta
from WocaR_DQN.a2c_ppo_acktr.utils import init
from WocaR_DQN.utils.ppo_core import mlp


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class BetaMLP(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=[64, 64], activation=torch.nn.Tanh, epsilon=1):
        super(BetaMLP, self).__init__()
        
        self.base = mlp([obs_dim+act_dim] + \
                    hidden_sizes, activation, activation)
        
        self.dist = Beta(hidden_sizes[-1], obs_dim)

        self.epsilon = epsilon


    def forward(self, inputs):
        logits = self.base(inputs)
        beta = self.dist(logits)
        output = beta.sample()
        output = 2 * self.epsilon * output - self.epsilon
        return output

class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None, beta=False, epsilon=0):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                base = CNNBase
            elif len(obs_shape) == 1:
                base = MLPBase
            else:
                raise NotImplementedError
        if beta:
            self.high = torch.from_numpy(action_space.high)
            self.epsilon = epsilon

        self.base = base(obs_shape[0], **base_kwargs)
        
        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            if len(action_space.shape) > 1:
                num_outputs = 1
                for s in action_space.shape:
                    num_outputs *= s
            else:
                num_outputs = action_space.shape[0]
            if beta:
                self.dist = Beta(self.base.output_size, num_outputs)
            else:
                self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError
        
    def act_mean(self, inputs, rnn_hxs, masks):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)
        return self.dist.logits
    
    def act(self, inputs, rnn_hxs, masks, deterministic=False, beta=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)
                
        # if not beta:
        #     if torch.sum(torch.isnan(dist.probs)).item() > 0:
        #         print("dist", dist.probs)
        #         print("obs", inputs)
        #         return None, None, None, None
        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean() 

        if beta:
            action = 2 * self.epsilon * action - self.epsilon
            
            if torch.isinf(action).any() or torch.isnan(action).any():
                print("find action inf!!!!!!!!!")
                action[action != action] = 0

        return value, action, action_log_probs, rnn_hxs
        
    def get_dist(self, inputs, rnn_hxs, masks):
        _, actor_features, _ = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)
        return dist

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action, beta=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if beta:
            action = (action + self.epsilon) / (2 * self.epsilon) # recover the action to before clipping

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        if torch.isinf(action_log_probs).any():
            print("found inf!!!!!!!!!")
            action_log_probs = action_log_probs.detach()

        return value, action_log_probs, dist_entropy, rnn_hxs


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class CNNBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512, dim=84):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        if dim == 84:
            self.main = nn.Sequential(
                init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),
                init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
                init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
                init_(nn.Linear(32 * 7 * 7, hidden_size)), nn.ReLU())
        elif dim == 80:
            self.main = nn.Sequential(
                init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),
                init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
                init_(nn.Conv2d(64, 64, 3, stride=1)), nn.ReLU(), Flatten(),
                init_(nn.Linear(64*6*6, hidden_size)), nn.ReLU())

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        # x = self.main(inputs / 255.0)
        x = self.main(inputs)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs
