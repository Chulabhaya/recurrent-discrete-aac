import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from gymnasium import spaces
from collections import deque


class Memory:
    """Memory for storing on-going episode for A2C."""

    def __init__(
        self,
        device="cpu",
    ):
        """Initialize the episode memory.

        Parameters
        ----------
        device : string
            Device on which to process memory.
        """
        # Store a list of episodes
        self.obs = deque()
        self.actions = deque()
        self.next_obs = deque()
        self.rewards = deque()
        self.terminateds = deque()
        self.truncateds = deque()
        self.entropies = []
        self.log_action_probs = []

        self.device = device

    def save_memory(self):
        """Saves content of memory to allow for later reloading.

        Returns
        -------
        memory_data : dict
            Dictionary containing current status of memory.
        """
        memory_data = {
            "obs": self.obs,
            "actions": self.actions,
            "next_obs": self.next_obs,
            "rewards": self.rewards,
            "terminateds": self.terminateds,
            "truncateds": self.truncateds,
            "entropies": self.entropies,
            "log_action_probs": self.log_action_probs,
        }

        return memory_data

    def load_memory(self, buffer_data):
        """Load data from prior saved replay memory.

        Parameters
        ----------
        memory_data : dict
            Dictionary containing saved replay memory data.
        """
        self.obs = buffer_data["obs"]
        self.actions = buffer_data["actions"]
        self.next_obs = buffer_data["next_obs"]
        self.rewards = buffer_data["rewards"]
        self.terminateds = buffer_data["terminateds"]
        self.truncateds = buffer_data["truncateds"]
        self.entropies = buffer_data["entropies"]
        self.log_action_probs = buffer_data["log_action_probs"]

    def push(
        self,
        obs,
        action,
        next_obs,
        reward,
        terminated,
        truncated,
        entropy,
        log_action_prob,
    ):
        """Adds a timestep of data to memory.

        Parameters
        ----------
        obs : int
            Observation.
        action : int
            Action.
        next_obs : int
            Next observation.
        reward : float
            Reward.
        terminated : bool
            Terminated status.
        truncated : bool
            Truncated status.
        entropy : float
            Policy entropies.
        log_action_prob : float
            Log of probability of action taken by policy.
        """

        # Update on-going episode with new timestep
        self.obs.append(obs)
        self.actions.append(action)
        self.next_obs.append(next_obs)
        self.rewards.append(reward)
        self.terminateds.append(terminated)
        self.truncateds.append(truncated)
        self.entropies.append(entropy)
        self.log_action_probs.append(log_action_prob)

    def pop_all(self):
        """Return episode from memory.

        Parameters
        ----------
        batch_size : int
            Size of batch to sample from buffer.

        Returns
        -------
        obs : tensor
            Episode observations.
        actions : tensor
            Episode actions.
        next_obs : tensor
            Episode next bservations.
        rewards : tensor
            Episode rewards.
        terminateds : tensor
            Episode terminations.
        entropies : tensor
            Entropies of policy.
        log_action_probs : tensor
            Log action probs for all actions taken in episode.
        """
        episode = (
            torch.as_tensor(np.array(self.obs)).to(self.device),
            torch.as_tensor(np.array(self.actions)).to(self.device),
            torch.as_tensor(np.array(self.next_obs)).to(self.device),
            torch.as_tensor(np.array(self.rewards)).to(self.device),
            torch.as_tensor(np.array(self.terminateds)).to(self.device).long(),
            torch.stack(self.entropies, dim=0),
            torch.stack(self.log_action_probs, dim=0),
        )

        # Reset memory for new episode
        self._reset()

        return episode

    def _reset(self):
        """Resets memory to store new episode after doing
        policy updates with previous episode.
        """
        self.obs = deque()
        self.actions = deque()
        self.next_obs = deque()
        self.rewards = deque()
        self.terminateds = deque()
        self.truncateds = deque()
        self.entropies = []
        self.log_action_probs = []
