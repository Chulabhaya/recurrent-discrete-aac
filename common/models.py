import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical


class RecurrentDiscreteCritic(nn.Module):
    """Recurrent discrete state-value function model for discrete A2C for POMDPs with discrete actions
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

    def forward(self, state):
        """
        Calculates state-action value.

        Parameters
        ----------
        state : tensor
            State or observation.

        Returns
        -------
        state_value : tensor
            State value for input state.
        """
        # Embedding layer
        x = F.relu(self.fc1(state))

        # Padded LSTM layer
        self.lstm1.flatten_parameters()
        x, _ = self.lstm1(x)

        # Remaining layers
        x = F.relu(self.fc2(x))
        state_value = self.fc3(x)
        return state_value


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

    def forward(self, state, in_hidden=None):
        """
        Calculates probabilities for taking each action given a state.

        Parameters
        ----------
        state : tensor
            State or observation.
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
        x = F.relu(self.fc1(state))

        # LSTM layer
        self.lstm1.flatten_parameters()
        x, out_hidden = self.lstm1(x, in_hidden)

        # Remaining layers
        x = F.relu(self.fc2(x))
        action_logits = self.fc_out(x)
        action_probs = self.softmax(action_logits)

        return action_probs, out_hidden

    def get_action(self, state, in_hidden=None):
        """
        Calculates actions by sampling from action distribution.

        Parameters
        ----------
        state : tensor
            State or observation.
        in_hidden : float
            LSTM hidden layer carrying over memory from previous timestep.

        Returns
        -------
        action : tensor
            Sampled action from action distribution.
        log_action_prob : tensor
            Log of probability of action sampled.
        entropy : tensor
            Entropy of policy.
        out_hidden : tensor
            LSTM hidden layer for preserving memory for next timestep.
        """
        action_probs, out_hidden = self.forward(state, in_hidden)

        dist = Categorical(action_probs)
        action = dist.sample().to(state.device)
        entropy = dist.entropy()
        log_action_prob = dist.log_prob(action)

        return action, log_action_prob, entropy, out_hidden


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

    def forward(self, state):
        """
        Calculates state value for a given input state.

        Parameters
        ----------
        state : tensor
            State or observation.

        Returns
        -------
        state_value : tensor
            State value for input state.
        """
        x = F.relu(self.fc1(state))
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

    def forward(self, state):
        """
        Calculates probabilities for taking each action given a state.

        Parameters
        ----------
        state : tensor
            State or observation.

        Returns
        -------
        action_probs : tensor
            Probabilities for all actions possible with input state.
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_logits = self.fc_out(x)
        action_probs = self.softmax(action_logits)

        return action_probs

    def get_action(self, state):
        """
        Calculates actions by sampling from action distribution.

        Parameters
        ----------
        state : tensor
            State or observation.

        Returns
        -------
        action : tensor
            Sampled action from action distribution.
        log_action_prob : tensor
            Log of probability of action sampled.
        entropy : tensor
            Entropy of policy.
        """
        action_probs = self.forward(state)

        dist = Categorical(action_probs)
        action = dist.sample().to(state.device)
        entropy = dist.entropy()
        log_action_prob = dist.log_prob(action)

        return action, log_action_prob, entropy


class RecurrentDiscreteCriticDiscreteObs(nn.Module):
    """Recurrent discrete state-value network for discrete A2C for POMDPs with discrete actions
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

    def forward(self, state):
        """
        Calculates state-action value.

        Parameters
        ----------
        state : tensor
            State or observation.

        Returns
        -------
        state_value : tensor
            State value for input state.
        """
        # Embedding layer
        x = self.embedding(state)

        # Padded LSTM layer
        self.lstm1.flatten_parameters()
        x, _ = self.lstm1(x)

        # Remaining layers
        x = F.relu(self.fc2(x))
        state_value = self.fc3(x)
        return state_value


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

    def forward(self, state, in_hidden=None):
        """
        Calculates probabilities for taking each action given a state.

        Parameters
        ----------
        state : tensor
            State or observation.
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
        x = self.embedding(state)

        # LSTM layer
        self.lstm1.flatten_parameters()
        x, out_hidden = self.lstm1(x, in_hidden)

        # Remaining layers
        x = F.relu(self.fc2(x))
        action_logits = self.fc_out(x)
        action_probs = self.softmax(action_logits)

        return action_probs, out_hidden

    def get_action(self, state, in_hidden=None):
        """
        Calculates actions by sampling from action distribution.

        Parameters
        ----------
        state : tensor
            State or observation.
        in_hidden : float
            LSTM hidden layer carrying over memory from previous timestep.

        Returns
        -------
        action : tensor
            Sampled action from action distribution.
        log_action_prob : tensor
            Log of probability of action sampled.
        entropy : tensor
            Entropy of policy.
        out_hidden : tensor
            LSTM hidden layer for preserving memory for next timestep.
        """
        action_probs, out_hidden = self.forward(state, in_hidden)

        dist = Categorical(action_probs)
        action = dist.sample().to(x.device)
        entropy = dist.entropy()
        log_action_prob = dist.log_prob(action)

        return action, log_action_prob, entropy, out_hidden


class DiscreteCriticDiscreteObs(nn.Module):
    """Discrete state-value function network for discrete A2C with discrete actions
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

    def forward(self, state):
        """
        Calculates state value for a given input state.

        Parameters
        ----------
        state : tensor
            State or observation.

        Returns
        -------
        state_value : tensor
            State value for input state.
        """
        x = self.embedding(state)
        x = F.relu(self.fc2(x))
        state_value = self.fc3(x)
        return state_value


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

    def forward(self, state):
        """
        Calculates probabilities for taking each action given a state.

        Parameters
        ----------
        state : tensor
            State or observation.

        Returns
        -------
        action_probs : tensor
            Probabilities for all actions possible with input state.
        """
        x = self.embedding(state)
        x = F.relu(self.fc2(x))
        action_logits = self.fc_out(x)
        action_probs = self.softmax(action_logits)

        return action_probs

    def get_action(self, state):
        """
        Calculates actions by sampling from action distribution.

        Parameters
        ----------
        state : tensor
            State or observation.

        Returns
        -------
        action : tensor
            Sampled action from action distribution.
        log_action_prob : tensor
            Log of probability of action sampled.
        entropy : tensor
            Entropy of policy.
        """
        action_probs = self.forward(state)

        dist = Categorical(action_probs)
        action = dist.sample().to(state.device)
        entropy = dist.entropy()
        log_action_prob = dist.log_prob(action)

        return action, log_action_prob, entropy
