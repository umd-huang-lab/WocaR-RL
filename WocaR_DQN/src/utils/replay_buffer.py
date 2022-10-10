import numpy as np
from collections import namedtuple, deque
from itertools import count
import random
import torch
from VaR_DQN.utils.param import Param


class DQNReplayBuffer:

    def __init__(self, action_size, buffer_size, batch_size, seed, intrinsic=False):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            intrinsic: whether to store intrinsic reward
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.intrinsic = intrinsic
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward",\
                                                                "next_state", "done"])
        self.seed = random.seed(seed)
        
    def add(self, state, action, reward, next_state, done, int_reward=None):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([np.expand_dims(e.state,axis=0) for e \
                                             in experiences if e is not None])).float().to(Param.device)
        actions = torch.from_numpy(np.vstack([e.action for e \
                                              in experiences if e is not None])).long().to(Param.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e \
                                              in experiences if e is not None])).float().to(Param.device)
        next_states = torch.from_numpy(np.vstack([np.expand_dims(e.next_state,axis=0) for e in experiences \
                                                  if e is not None])).float().to(Param.device)
        dones = torch.from_numpy(np.vstack([e.done for e \
                                            in experiences if e is not \
                                            None]).astype(np.uint8)).float().to(Param.device)
        states.requires_grad = False
        actions.requires_grad = False
        rewards.requires_grad = False
        next_states.requires_grad = False
        dones.requires_grad = False
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

def from_numpy(n_array, dtype=None):
    if dtype is None:
        return torch.from_numpy(n_array).to(Param.device).type(Param.dtype)
    else:
        return torch.from_numpy(n_array).to(Param.device).type(dtype)
    
def from_tuple(t, dtype=None):
    if dtype is None:
        return torch.tensor(t).to(Param.device).type(Param.dtype)
    else:
        return torch.tensor(t).to(Param.device).type(dtype)

class DQNPrioritizedBuffer:
    def __init__(self, buffer_size, batch_size, prob_alpha=0.6, seed=0):
        self.prob_alpha   = prob_alpha
        self.buffer_size  = buffer_size
        self.batch_size   = batch_size
        self.buffer       = []
        self.pos          = 0
        self.priorities   = np.zeros((buffer_size,), dtype=np.float32)
        np.random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        
        max_prio = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.buffer_size:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.buffer_size
    
    def sample(self, beta=0.4):
        if len(self.buffer) == self.buffer_size:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        
        probs  = prios ** self.prob_alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), self.batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        total    = len(self.buffer)
        weights  = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights  = from_numpy(weights)
        
        batch       = list(zip(*samples))
        states      = from_numpy(np.concatenate(batch[0])).unsqueeze(1)
        actions     = from_tuple(batch[1], dtype=torch.int64).unsqueeze(1)
        rewards     = from_tuple(batch[2]).unsqueeze(1)
        next_states = from_numpy(np.concatenate(batch[3])).unsqueeze(1)
        dones       = from_tuple(batch[4])
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)
