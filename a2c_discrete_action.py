import argparse
import os
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

import wandb
from common.memory import Memory
from common.models import DiscreteActor, DiscreteCritic
from common.utils import generalized_advantage_estimate, make_env, save, set_seed


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--wandb-project", type=str, default="a2c-discrete-action",
        help="wandb project name")
    parser.add_argument("--wandb-group", type=str, default=None,
        help="wandb group name to use for run")
    parser.add_argument("--wandb-dir", type=str, default="./",
        help="the wandb directory")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="CartPole-P-v0",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=200500,
        help="total timesteps of the experiments")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--lmbda", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--policy-lr", type=float, default=3e-4,
        help="the learning rate of the policy network optimizer")
    parser.add_argument("--v-lr", type=float, default=1e-3,
        help="the learning rate of the state value network optimizer")
    parser.add_argument("--beta", type=float, default=0.01,
        help="coefficient for entropy loss")

    # Checkpointing specific arguments
    parser.add_argument("--save", type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=True,
        help="checkpoint saving during training")
    parser.add_argument("--save-checkpoint-dir", type=str, default="./trained_models/",
        help="path to directory to save checkpoints in")
    parser.add_argument("--checkpoint-interval", type=int, default=5000,
        help="how often to save checkpoints during training (in timesteps)")
    parser.add_argument("--resume", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to resume training from a checkpoint")
    parser.add_argument("--resume-checkpoint-path", type=str, default=None,
        help="path to checkpoint to resume training from")
    parser.add_argument("--run-id", type=str, default=None,
        help="wandb unique run id for resuming")

    args = parser.parse_args()
    # fmt: on
    return args


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.exp_name}"
    wandb_id = wandb.util.generate_id()
    run_id = f"{run_name}_{wandb_id}"

    # If a unique wandb run id is given, then resume from that, otherwise
    # generate new run for resuming
    if args.resume and args.run_id is not None:
        run_id = args.run_id
        wandb.init(
            id=run_id,
            dir=args.wandb_dir,
            project=args.wandb_project,
            resume="must",
            mode="offline",
        )
    else:
        wandb.init(
            id=run_id,
            dir=args.wandb_dir,
            project=args.wandb_project,
            config=vars(args),
            name=run_name,
            save_code=True,
            settings=wandb.Settings(code_dir="."),
            group=args.wandb_group,
            mode="offline",
        )

    # Set training device
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print("Running on the following device: " + device.type, flush=True)

    # Set seeding
    set_seed(args.seed, device)

    # Load checkpoint if resuming
    if args.resume:
        print("Resuming from checkpoint: " + args.resume_checkpoint_path, flush=True)
        checkpoint = torch.load(args.resume_checkpoint_path)

    # Set RNG state for seeds if resuming
    if args.resume:
        random.setstate(checkpoint["rng_states"]["random_rng_state"])
        np.random.set_state(checkpoint["rng_states"]["numpy_rng_state"])
        torch.set_rng_state(checkpoint["rng_states"]["torch_rng_state"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(checkpoint["rng_states"]["torch_cuda_rng_state"])
            torch.cuda.set_rng_state_all(
                checkpoint["rng_states"]["torch_cuda_rng_state_all"]
            )

    # Env setup
    env = make_env(args.env_id, args.seed)
    assert isinstance(
        env.action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    # Initialize models and optimizers
    actor = DiscreteActor(env).to(device)
    vf1 = DiscreteCritic(env).to(device)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)
    v_optimizer = optim.Adam(list(vf1.parameters()), lr=args.v_lr)

    # If resuming training, load models and optimizers
    if args.resume:
        actor.load_state_dict(checkpoint["model_state_dict"]["actor_state_dict"])
        vf1.load_state_dict(checkpoint["model_state_dict"]["vf1_state_dict"])
        v_optimizer.load_state_dict(checkpoint["optimizer_state_dict"]["v_optimizer"])
        actor_optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]["actor_optimizer"]
        )

    # Initialize memory
    memory = Memory(
        device,
    )
    # If resuming training, then load memory
    if args.resume:
        memory_data = checkpoint["memory"]
        memory.load_memory(memory_data)

    # Start time tracking for run
    start_time = time.time()

    # Start the game
    start_global_step = 0
    # If resuming, update starting step
    if args.resume:
        start_global_step = checkpoint["global_step"] + 1

    obs, info = env.reset(seed=args.seed)
    # Set RNG state for env
    if args.resume:
        env.np_random.bit_generator.state = checkpoint["rng_states"]["env_rng_state"]
        env.action_space.np_random.bit_generator.state = checkpoint["rng_states"][
            "env_action_space_rng_state"
        ]
        env.observation_space.np_random.bit_generator.state = checkpoint["rng_states"][
            "env_obs_space_rng_state"
        ]
    global_step = start_global_step
    while global_step < args.total_timesteps:
        # Store values for data logging for each global step
        data_log = {}

        # Collect an episode rollout with current policy
        terminated, truncated = False, False
        while not (truncated or terminated):
            # Get action
            action, log_action_prob, entropy = actor.get_actions(
                torch.tensor(obs).to(device)
            )
            action = action.detach().cpu().numpy()

            # Take step in environment
            next_obs, reward, terminated, truncated, info = env.step(action)

            # Add data to memory
            memory.push(
                obs,
                action,
                next_obs,
                reward,
                terminated,
                truncated,
                entropy,
                log_action_prob,
            )
            global_step += 1

            # Update next obs
            obs = next_obs

            # Handle episode end, record rewards for plotting purposes
            if terminated or truncated:
                print(
                    f"global_step={global_step}, episodic_return={info['episode']['r'][0]}, episodic_length={info['episode']['l'][0]}",
                    flush=True,
                )
                data_log["misc/episodic_return"] = info["episode"]["r"][0]
                data_log["misc/episodic_length"] = info["episode"]["l"][0]

                obs, info = env.reset()

        # ---------- update critic ---------- #
        # Calculate state value predictions for current observations and next observations
        (
            episode_obs,
            episode_actions,
            episode_next_obs,
            episode_rewards,
            episode_terminateds,
            episode_entropies,
            episode_log_action_probs,
        ) = memory.pop_all()
        v_pred_values = vf1(episode_obs).squeeze()
        v_next_pred_values = vf1(episode_next_obs).squeeze()

        # Calculate advantages using Generalized Advantage Estimation (GAE)
        advantages = generalized_advantage_estimate(
            args.gamma,
            args.lmbda,
            v_pred_values.detach(),
            v_next_pred_values.detach(),
            episode_rewards,
            episode_terminateds,
        )

        # Calculate TD value target
        v_target_values = (advantages + v_pred_values).detach()

        # Calculate state value function loss
        vf_loss = F.mse_loss(v_target_values, v_pred_values)
        v_optimizer.zero_grad()
        vf_loss.backward()
        v_optimizer.step()

        # ---------- update actor ---------- #
        actor_loss = (
            -episode_log_action_probs * advantages
        ).mean() - args.beta * episode_entropies.mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        if global_step % 100 == 0:
            data_log["losses/vf1_values"] = v_pred_values.mean().item()
            data_log["losses/vf_loss"] = vf_loss.item()
            data_log["losses/actor_loss"] = actor_loss.item()
            data_log["misc/steps_per_second"] = int(
                global_step / (time.time() - start_time)
            )
            print("SPS:", int(global_step / (time.time() - start_time)), flush=True)

        data_log["misc/global_step"] = global_step
        wandb.log(data_log, step=global_step)

        # Save checkpoints during training
        if args.save:
            if global_step % args.checkpoint_interval == 0:
                # Save models
                models = {
                    "actor_state_dict": actor.state_dict(),
                    "vf1_state_dict": vf1.state_dict(),
                }
                # Save optimizers
                optimizers = {
                    "v_optimizer": v_optimizer.state_dict(),
                    "actor_optimizer": actor_optimizer.state_dict(),
                }
                # Save random states, important for reproducibility
                rng_states = {
                    "random_rng_state": random.getstate(),
                    "numpy_rng_state": np.random.get_state(),
                    "torch_rng_state": torch.get_rng_state(),
                    "env_rng_state": env.np_random.bit_generator.state,
                    "env_action_space_rng_state": env.action_space.np_random.bit_generator.state,
                    "env_obs_space_rng_state": env.observation_space.np_random.bit_generator.state,
                }
                if device.type == "cuda":
                    rng_states["torch_cuda_rng_state"] = torch.cuda.get_rng_state()
                    rng_states[
                        "torch_cuda_rng_state_all"
                    ] = torch.cuda.get_rng_state_all()

                save(
                    run_id,
                    args.save_checkpoint_dir,
                    global_step,
                    models,
                    memory,
                    optimizers,
                    rng_states,
                )

    env.close()
