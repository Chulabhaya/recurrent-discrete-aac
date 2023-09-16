import os
import random

import gymnasium as gym
import gymnasium_pomdps
import numpy as np
import torch


def make_env(env_id, seed, max_episode_len=None):
    """
    Generate environment with seeding/wrapping.

    Args:
        env_id: ID of the environment to use.
        seed: Seed to set for environment.
        max_episode_len: Episode timeout length.

    Returns:
        Generated environment.

    """
    env = gym.make(env_id)
    if max_episode_len is not None:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_len)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    return env


def save(
    run_id,
    checkpoint_dir,
    global_step,
    models,
    optimizers,
    replay_buffer,
    rng_states,
):
    """
    Saves a checkpoint.

    Args:
        run_id: Wandb ID of run.
        checkpoint_dir: Directory to store checkpoint in.
        global_step: Timestep of training.
        models: State dict of models.
        optimizers: State dict of optimizers.
        replay_buffer: Replay buffer.
        rng_states: RNG states.
    """
    save_dir = checkpoint_dir + run_id + "/"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        # Prevent permission issues when writing to this directory
        # after resuming a training job
        os.chmod(save_dir, 0o777)

    save_path = save_dir + "global_step_" + str(global_step) + ".pth"
    print("Saving checkpoint: " + save_path, flush=True)
    torch.save(
        {
            "global_step": global_step,
            "model_state_dict": models,
            "optimizer_state_dict": optimizers,
            "replay_buffer": replay_buffer,
            "rng_states": rng_states,
        },
        save_path,
    )


def set_seed(seed, device):
    """
    Sets seeding for experiment.

    Args:
        seed: Seed.
        device: Device being used.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def generalized_advantage_estimate(
    gamma, lmbda, v_pred_values, v_next_pred_values, rewards, terminateds
):
    """Generates seeded environment.

    Parameters
    ----------
    gamma : float
        Discount factor gamma.
    lmbda : float
        Bias/variance trade-off factor lambda.
    v_pred_values : tensor
        Value function predictions for current observations.
    v_next_pred_values : tensor
        Value function predictions for next observations.
    rewards : tensor
        Episode rewards.
    terminateds : tensor
        Episode termination statuses.

    Returns
    -------
    advantages : tensor
        Sum of weighted advantages calculated through GAE.
    """
    # Initialize advantages with extra zero to represent that
    # next advantage after the last timestep is zero (because
    # it doesn't exist)
    advantages = torch.zeros(
        rewards.shape[0] + 1, device=rewards.device
    )

    # Calculate backwards for computational efficiency
    for t in reversed(range(rewards.shape[0])):
        delta_t = (
            rewards[t] + (1 - terminateds[t]) * gamma * v_next_pred_values[t] - v_pred_values[t]
        )
        advantages[t] = (
            delta_t + (1 - terminateds[t]) * gamma * lmbda * advantages[t + 1]
        )

    # Remove extra zero from advantages
    advantages = advantages[:-1]

    return advantages
