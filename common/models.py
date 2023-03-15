import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.distributions import Categorical


class RecurrentDiscreteCritic(nn.Module):
    """Recurrent discrete soft Q-network model for discrete SAC for POMDPs with discrete actions
    and continuous observations."""

    def __init__(self, env):
        """Initialize the critic model.

        Parameters
        ----------
        env : gym environment
            Gym environment being used for learning.
        """
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod(), 256)
        self.lstm1 = nn.LSTM(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, env.action_space.n)

    def forward(self, x, seq_lengths):
        """
        Calculates Q-values for each state-action.

        Parameters
        ----------
        x : tensor
            State or observation.
        seq_lengths : tensor
             Sequence lengths for data in batch.

        Returns
        -------
        q_values : tensor
            Q-values for all actions possible with input state.
        """
        # Embedding layer
        x = F.relu(self.fc1(x))

        # Padded LSTM layer
        x = pack_padded_sequence(x, seq_lengths, enforce_sorted=False)
        self.lstm1.flatten_parameters()
        x, _ = self.lstm1(x)
        x, x_unpacked_len = pad_packed_sequence(x)

        # Remaining layers
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values


class RecurrentDiscreteActor(nn.Module):
    """Recurrent discrete soft actor model for discrete SAC for POMDPs with discrete actions
    and continuous observations."""

    def __init__(self, env):
        """Initialize the actor model.

        Parameters
        ----------
        env : gym environment
            Gym environment being used for learning.
        """
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod(), 256)
        self.lstm1 = nn.LSTM(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_out = nn.Linear(256, env.action_space.n)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, seq_lengths, in_hidden=None):
        """
        Calculates probabilities for taking each action given a state.

        Parameters
        ----------
        x : tensor
            State or observation.
        seq_lengths : tensor
             Sequence lengths for data in batch.
        in_hidden : float
            LSTM hidden layer carrying over memory from previous timestep.

        Returns
        -------
        action_probs : tensor
            Probabilities for all actions possible with input state.
        out_hidden : tensor
            LSTM hidden layer for preserving memory for next timestep.
        """
        # Embedding layer
        x = F.relu(self.fc1(x))

        # Padded LSTM layer
        x = pack_padded_sequence(x, seq_lengths, enforce_sorted=False)
        self.lstm1.flatten_parameters()
        x, out_hidden = self.lstm1(x, in_hidden)
        x, x_unpacked_len = pad_packed_sequence(x)

        # Remaining layers
        x = F.relu(self.fc2(x))
        action_logits = self.fc_out(x)
        action_probs = self.softmax(action_logits)

        return action_probs, out_hidden

    def get_action(self, x, seq_lengths, in_hidden=None, epsilon=1e-6):
        """
        Calculates actions by sampling from action distribution.

        Parameters
        ----------
        x : tensor
            Action probabilities.
        seq_lengths : tensor
             Sequence lengths for data in batch.
        in_hidden : float
            LSTM hidden layer carrying over memory from previous timestep.
        epsilon : float
            Used to ensure no zero probability values.

        Returns
        -------
        action : tensor
            Sampled action from action distribution.
        action_probs : tensor
            Probabilities for all actions possible with input state.
        log_action_probs : tensor
            Log of action probabilities, used for entropy.
        out_hidden : tensor
            LSTM hidden layer for preserving memory for next timestep.
        """
        action_probs, out_hidden = self.forward(x, seq_lengths, in_hidden)

        dist = Categorical(action_probs)
        action = dist.sample().to(x.device)

        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_action_probs = torch.log(action_probs + z)
        return action, action_probs, log_action_probs, out_hidden


class DiscreteCritic(nn.Module):
    """Discrete value network for discrete A2C with discrete actions
    and continuous observations."""

    def __init__(self, env):
        """Initialize the critic model.

        Parameters
        ----------
        env : gym environment
            Gym environment being used for learning.
        """
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        """
        Calculates state value for a given input state.

        Parameters
        ----------
        x : tensor
            State or observation.

        Returns
        -------
        state_value : tensor
            State value for input state.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        state_value = self.fc3(x)
        return state_value


class DiscreteActor(nn.Module):
    """Discrete actor model for discrete A2C with discrete actions
    and continuous observations."""

    def __init__(self, env):
        """Initialize the actor model.

        Parameters
        ----------
        env : gym environment
            Gym environment being used for learning.
        """
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_out = nn.Linear(256, env.action_space.n)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Calculates probabilities for taking each action given a state.

        Parameters
        ----------
        x : tensor
            State or observation.

        Returns
        -------
        action_probs : tensor
            Probabilities for all actions possible with input state.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_logits = self.fc_out(x)
        action_probs = self.softmax(action_logits)

        return action_probs

    def get_action(self, x):
        """
        Calculates actions by sampling from action distribution.

        Parameters
        ----------
        x : tensor
            Action probabilities.

        Returns
        -------
        action : tensor
            Sampled action from action distribution.
        log_action_prob : tensor
            Log of probability of action sampled.
        entropy : tensor
            Entropy of policy.
        """
        action_probs = self.forward(x)

        dist = Categorical(action_probs)
        action = dist.sample().to(x.device)
        entropy = dist.entropy()
        log_action_prob = dist.log_prob(action)

        return action, log_action_prob, entropy


class RecurrentDiscreteCriticDiscreteObs(nn.Module):
    """Recurrent discrete soft Q-network model for discrete SAC for POMDPs with discrete actions
    and discrete observations."""

    def __init__(self, env):
        """Initialize the critic model.

        Parameters
        ----------
        env : gym environment
            Gym environment being used for learning.
        """
        super().__init__()
        self.embedding = nn.Embedding(env.observation_space.n, 256)
        self.lstm1 = nn.LSTM(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, env.action_space.n)

    def forward(self, x, seq_lengths):
        """
        Calculates Q-values for each state-action.

        Parameters
        ----------
        x : tensor
            State or observation.
        seq_lengths : tensor
             Sequence lengths for data in batch.

        Returns
        -------
        q_values : tensor
            Q-values for all actions possible with input state.
        """
        # Embedding layer
        x = self.embedding(x)

        # Padded LSTM layer
        x = pack_padded_sequence(x, seq_lengths, enforce_sorted=False)
        self.lstm1.flatten_parameters()
        x, _ = self.lstm1(x)
        x, x_unpacked_len = pad_packed_sequence(x)

        # Remaining layers
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values


class RecurrentDiscreteActorDiscreteObs(nn.Module):
    """Recurrent discrete actor model for discrete SAC for POMDPs with discrete actions
    and discrete observations."""

    def __init__(self, env):
        """Initialize the actor model.

        Parameters
        ----------
        env : gym environment
            Gym environment being used for learning.
        """
        super().__init__()
        self.embedding = nn.Embedding(env.observation_space.n, 256)
        self.lstm1 = nn.LSTM(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_out = nn.Linear(256, env.action_space.n)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, seq_lengths, in_hidden=None):
        """
        Calculates probabilities for taking each action given a state.

        Parameters
        ----------
        x : tensor
            State or observation.
        seq_lengths : tensor
             Sequence lengths for data in batch.
        in_hidden : float
            LSTM hidden layer carrying over memory from previous timestep.

        Returns
        -------
        action_probs : tensor
            Probabilities for all actions possible with input state.
        out_hidden : tensor
            LSTM hidden layer for preserving memory for next timestep.
        """
        # Embedding layer
        x = self.embedding(x)

        # Padded LSTM layer
        x = pack_padded_sequence(x, seq_lengths, enforce_sorted=False)
        self.lstm1.flatten_parameters()
        x, out_hidden = self.lstm1(x, in_hidden)
        x, x_unpacked_len = pad_packed_sequence(x)

        # Remaining layers
        x = F.relu(self.fc2(x))
        action_logits = self.fc_out(x)
        action_probs = self.softmax(action_logits)

        return action_probs, out_hidden

    def get_action(self, x, seq_lengths, in_hidden=None, epsilon=1e-6):
        """
        Calculates actions by sampling from action distribution.

        Parameters
        ----------
        x : tensor
            Action probabilities.
        seq_lengths : tensor
             Sequence lengths for data in batch.
        in_hidden : float
            LSTM hidden layer carrying over memory from previous timestep.
        epsilon : float
            Used to ensure no zero probability values.

        Returns
        -------
        action : tensor
            Sampled action from action distribution.
        action_probs : tensor
            Probabilities for all actions possible with input state.
        log_action_probs : tensor
            Log of action probabilities, used for entropy.
        out_hidden : tensor
            LSTM hidden layer for preserving memory for next timestep.
        """
        action_probs, out_hidden = self.forward(x, seq_lengths, in_hidden)

        dist = Categorical(action_probs)
        action = dist.sample().to(x.device)

        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_action_probs = torch.log(action_probs + z)
        return action, action_probs, log_action_probs, out_hidden


class DiscreteCriticDiscreteObs(nn.Module):
    """Discrete soft Q-network model for discrete SAC with discrete actions
    and discrete observations."""

    def __init__(self, env):
        """Initialize the critic model.

        Parameters
        ----------
        env : gym environment
            Gym environment being used for learning.
        """
        super().__init__()
        self.embedding = nn.Embedding(env.observation_space.n, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, env.action_space.n)

    def forward(self, x):
        """
        Calculates Q-values for each state-action.

        Parameters
        ----------
        x : tensor
            State or observation.

        Returns
        -------
        q_values : tensor
            Q-values for all actions possible with input state.
        """
        x = self.embedding(x)
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values


class DiscreteActorDiscreteObs(nn.Module):
    """Discrete actor model for discrete SAC with discrete actions
    and discrete observations."""

    def __init__(self, env):
        """Initialize the actor model.

        Parameters
        ----------
        env : gym environment
            Gym environment being used for learning.
        """
        super().__init__()
        self.embedding = nn.Embedding(env.observation_space.n, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_out = nn.Linear(256, env.action_space.n)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Calculates probabilities for taking each action given a state.

        Parameters
        ----------
        x : tensor
            State or observation.

        Returns
        -------
        action_probs : tensor
            Probabilities for all actions possible with input state.
        """
        x = self.embedding(x)
        x = F.relu(self.fc2(x))
        action_logits = self.fc_out(x)
        action_probs = self.softmax(action_logits)

        return action_probs

    def get_action(self, x, epsilon=1e-6):
        """
        Calculates actions by sampling from action distribution.

        Parameters
        ----------
        x : tensor
            Action probabilities.
        epsilon : float
            Used to ensure no zero probability values.

        Returns
        -------
        action : tensor
            Sampled action from action distribution.
        action_probs : tensor
            Probabilities for all actions possible with input state.
        log_action_probs : tensor
            Log of action probabilities, used for entropy.
        """
        action_probs = self.forward(x)

        dist = Categorical(action_probs)
        action = dist.sample().to(x.device)

        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_action_probs = torch.log(action_probs + z)
        return action, action_probs, log_action_probs
