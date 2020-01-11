import numpy as np
import random
from collections import namedtuple, deque

from models import DQN, DuelingDQN

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-2              # for soft update of target parameters
LR = 4.85e-4            # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DQNAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, name, state_size, action_size, use_double_dqn=False, use_dueling=False, seed=0, lr_decay=0.9999, use_prioritized_replay=False):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.name = name
        self.state_size = state_size
        self.action_size = action_size
        self.use_double_dqn = use_double_dqn
        self.use_dueling = use_dueling
        self.seed = random.seed(seed)
        self.use_prioritized_replay = use_prioritized_replay

        # Q-Network
        if use_dueling:
            self.qnetwork_local = DuelingDQN(state_size, action_size, seed).to(device)
            self.qnetwork_target = DuelingDQN(state_size, action_size, seed).to(device)
        else:
            self.qnetwork_local = DQN(state_size, action_size, seed).to(device)
            self.qnetwork_target = DQN(state_size, action_size, seed).to(device)

        self.qnetwork_target.eval()
            
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, lr_decay)

        # Replay memory
        if self.use_prioritized_replay:
            self.memory = PrioritizedReplayBuffer(BUFFER_SIZE, seed, alpha=0.2, beta=0.8, beta_scheduler=1.0)
        else:
            self.memory = ReplayBuffer(BUFFER_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample(BATCH_SIZE)
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        # Epsilon-greedy action selection
        if random.random() > eps:
            self.qnetwork_local.eval()
            with torch.no_grad():
                action_values = self.qnetwork_local(state)
            self.qnetwork_local.train()
        
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        if self.use_prioritized_replay:
            states, actions, rewards, next_states, dones, indices, weights = experiences
        else:
            states, actions, rewards, next_states, dones = experiences

        with torch.no_grad():
            # Get max predicted Q values (for next states) from target model
            if self.use_double_dqn:            
                best_local_actions = self.qnetwork_local(states).max(1)[1].unsqueeze(1)
                Q_targets_next = self.qnetwork_target(next_states).gather(1, best_local_actions).max(1)[0].unsqueeze(1)
            else:
                Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

            # Compute Q targets for current states 
            Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        if self.use_prioritized_replay:
            Q_targets.sub_(Q_expected)
            Q_targets.squeeze_()
            Q_targets.pow_(2)

            with torch.no_grad():
                td_error = Q_targets.detach()
                #td_error.pow_(0.5)
                td_error.mul_(weights)

                self.memory.update_priorities(indices, td_error)

            Q_targets.mul_(weights)
            loss = Q_targets.mean()
        else:                
            # Compute loss
            loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            buffer_size (int): maximum size of buffer
            seed (int): random seed
        """
        self.memory = deque(maxlen=buffer_size)  
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self, batch_size):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class PrioritizedReplayBuffer:
    def __init__(self, buffer_size, seed, alpha=0., beta=1., beta_scheduler=1.):
        """Initialize a PrioritizedReplayBuffer object.

        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            alpha (float): determines how much prioritization is used; α = 0 corresponding to the uniform case
            beta (float): amount of importance-sampling correction; β = 1 fully compensates for the non-uniform probabilities
            beta_scheduler (float): multiplicative factor (per sample) for increasing beta (should be >= 1.0)
        """
        self.buffer_size = buffer_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])        
        self.memory = []
        self.pos = 0        
        self.priorities = np.zeros((buffer_size,), dtype=np.float32)
        self.seed = random.seed(seed)
        self.alpha = alpha
        self.beta = beta
        self.beta_scheduler = beta_scheduler

        # Each new experience is added to the memory with
        # the maximum probability of being choosen
        self.max_prob = 0.001

        # Value to a non-zero probability
        self.nonzero_probability = 0.00001

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""

        e = self.experience(state, action, reward, next_state, done)

        if len(self.memory) < self.buffer_size:
            self.memory.append(e)
        else:
            self.memory[self.pos] = e

        self.priorities[self.pos] = self.max_prob
        self.pos = (self.pos + 1) % self.buffer_size

    def sample(self, batch_size):
        """Sample a batch of prioritized experiences from memory."""
        if len(self.memory) == self.buffer_size:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        # Normalize the probability of being chosen for each one of the memory registers        
        probs = prios # ** self.alpha
        probs = np.divide(probs, probs.sum())

        indices = np.random.choice(len(self.memory), batch_size, replace=False, p=probs)
        samples = [self.memory[idx] for idx in indices]

        # Compute importance-sampling weights for each one of the memory registers
        # w = ((N * P) ^ -β) / max(w)
        N    = len(self.memory)
        weights = np.multiply(prios[indices], N)
        weights = np.power(weights, -self.beta, where=weights!=0)
        weights = np.divide(weights, weights.max())

        self.beta = min(1, self.beta*self.beta_scheduler)

        states = torch.from_numpy(np.vstack([e.state for e in samples if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in samples if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in samples if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in samples if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in samples if e is not None]).astype(np.uint8)).float().to(device)
        weights  = torch.from_numpy(weights).float().to(device)

        return (states, actions, rewards, next_states, dones, indices, weights)

    def update_priorities(self, batch_indices, batch_priorities):
        # Balance the prioritization using the alpha value
        batch_priorities.pow_(self.alpha)

        # Guarantee a non-zero probability
        batch_priorities.add_(self.nonzero_probability)

        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

        self.max_prob = self.priorities.max()

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)