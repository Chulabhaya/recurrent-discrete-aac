import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical


class RecurrentDiscreteCritic(nn.Module):
    """Recurrent discrete state value function model for discrete A2C for POMDPs with discrete actions
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
        self.fc3 = nn.Linear(256, 1)

    def forward(self, states):
        """
        Calculates state-action values.

        Parameters
        ----------
        states : tensor
            States or observations.

        Returns
        -------
        values : tensor
            State values for input states.
        """
        # Embedding layer
        x = F.relu(self.fc1(states))

        # Padded LSTM layer
        self.lstm1.flatten_parameters()
        x, _ = self.lstm1(x)

        # Remaining layers
        x = F.relu(self.fc2(x))
        values = self.fc3(x)
        return values


class RecurrentDiscreteActor(nn.Module):
    """Recurrent discrete actor for discrete A2C for POMDPs with discrete actions
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

    def forward(self, states, in_hidden=None):
        """
        Calculates action probabilities given states.

        Parameters
        ----------
        states : tensor
            States or observations.
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
        x = F.relu(self.fc1(states))

        # LSTM layer
        self.lstm1.flatten_parameters()
        x, out_hidden = self.lstm1(x, in_hidden)

        # Remaining layers
        x = F.relu(self.fc2(x))
        action_logits = self.fc_out(x)
        action_probs = self.softmax(action_logits)

        return action_probs, out_hidden

    def get_actions(self, states, in_hidden=None):
        """
        Calculates actions by sampling from action distribution.

        Parameters
        ----------
        states : tensor
            States or observations.
        in_hidden : float
            LSTM hidden layer carrying over memory from previous timestep.

        Returns
        -------
        actions : tensor
            Sampled actions from action distribution.
        log_action_probs : tensor
            Logs of action probabilities.
        entropies : tensor
            Policy entropies.
        out_hidden : tensor
            LSTM hidden layer for preserving memory for next timestep.
        """
        action_probs, out_hidden = self.forward(states, in_hidden)

        dist = Categorical(action_probs)
        actions = dist.sample().to(states.device)
        entropies = dist.entropy()
        log_action_probs = dist.log_prob(actions)

        return actions, log_action_probs, entropies, out_hidden


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

    def forward(self, states):
        """
        Calculates state value for a given input state.

        Parameters
        ----------
        states : tensor
            States or observations.

        Returns
        -------
        values : tensor
            State values for input states.
        """
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        values = self.fc3(x)
        return values


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

    def forward(self, states):
        """
        Calculates action probabilities given states.

        Parameters
        ----------
        states : tensor
            States or observations.

        Returns
        -------
        action_probs : tensor
            Probabilities for all actions possible with input state.
        """
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        action_logits = self.fc_out(x)
        action_probs = self.softmax(action_logits)

        return action_probs

    def get_actions(self, states):
        """
        Calculates actions by sampling from action distribution.

        Parameters
        ----------
        states : tensor
            States or observations.

        Returns
        -------
        actions : tensor
            Sampled actions from action distribution.
        log_action_probs : tensor
            Logs of action probabilities.
        entropies : tensor
            Policy entropies.
        """
        action_probs = self.forward(states)

        dist = Categorical(action_probs)
        actions = dist.sample().to(states.device)
        entropies = dist.entropy()
        log_action_probs = dist.log_prob(actions)

        return actions, log_action_probs, entropies


class RecurrentDiscreteCriticDiscreteObs(nn.Module):
    """Recurrent discrete state value network for discrete A2C for POMDPs with discrete actions
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
        self.fc3 = nn.Linear(256, 1)

    def forward(self, states):
        """
        Calculates state-action values.

        Parameters
        ----------
        states : tensor
            States or observations.

        Returns
        -------
        values : tensor
            State values for input states.
        """
        # Embedding layer
        x = self.embedding(states)

        # Padded LSTM layer
        self.lstm1.flatten_parameters()
        x, _ = self.lstm1(x)

        # Remaining layers
        x = F.relu(self.fc2(x))
        values = self.fc3(x)
        return values


class RecurrentDiscreteActorDiscreteObs(nn.Module):
    """Recurrent discrete actor model for discrete A2C for POMDPs with discrete actions
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

    def forward(self, states, in_hidden=None):
        """
        Calculates action probabilities given states.

        Parameters
        ----------
        states : tensor
            States or observations.
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
        x = self.embedding(states)

        # LSTM layer
        self.lstm1.flatten_parameters()
        x, out_hidden = self.lstm1(x, in_hidden)

        # Remaining layers
        x = F.relu(self.fc2(x))
        action_logits = self.fc_out(x)
        action_probs = self.softmax(action_logits)

        return action_probs, out_hidden

    def get_actions(self, states, in_hidden=None):
        """
        Calculates actions by sampling from action distribution.

        Parameters
        ----------
        states : tensor
            States or observations.
        in_hidden : float
            LSTM hidden layer carrying over memory from previous timestep.

        Returns
        -------
        actions : tensor
            Sampled actions from action distribution.
        log_action_probs : tensor
            Logs of action probabilities.
        entropies : tensor
            Policy entropies.
        out_hidden : tensor
            LSTM hidden layer for preserving memory for next timestep.
        """
        action_probs, out_hidden = self.forward(states, in_hidden)

        dist = Categorical(action_probs)
        actions = dist.sample().to(states.device)
        entropies = dist.entropy()
        log_action_probs = dist.log_prob(actions)

        return actions, log_action_probs, entropies, out_hidden


class DiscreteCriticDiscreteObs(nn.Module):
    """Discrete state value function network for discrete A2C with discrete actions
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
        self.fc3 = nn.Linear(256, 1)

    def forward(self, states):
        """
        Calculates state value for a given input state.

        Parameters
        ----------
        states : tensor
            States or observations.

        Returns
        -------
        values : tensor
            State values for input states.
        """
        x = self.embedding(states)
        x = F.relu(self.fc2(x))
        values = self.fc3(x)
        return values


class DiscreteActorDiscreteObs(nn.Module):
    """Discrete actor model for discrete A2C with discrete actions
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

    def forward(self, states):
        """
        Calculates action probabilities given states.

        Parameters
        ----------
        states : tensor
            States or observations.

        Returns
        -------
        action_probs : tensor
            Probabilities for all actions possible with input state.
        """
        x = self.embedding(states)
        x = F.relu(self.fc2(x))
        action_logits = self.fc_out(x)
        action_probs = self.softmax(action_logits)

        return action_probs

    def get_actions(self, states):
        """
        Calculates actions by sampling from action distribution.

        Parameters
        ----------
        states : tensor
            States or observations.

        Returns
        -------
        actions : tensor
            Sampled actions from action distribution.
        log_action_probs : tensor
            Logs of action probabilities.
        entropies : tensor
            Policy entropies.
        """
        action_probs = self.forward(states)

        dist = Categorical(action_probs)
        actions = dist.sample().to(states.device)
        entropies = dist.entropy()
        log_action_probs = dist.log_prob(actions)

        return actions, log_action_probs, entropies
