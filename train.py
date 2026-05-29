import argparse
import os
import ray
import pygame
import numpy as np
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from football_env import FootballEnv

# 1. Wrapper function for RLlib
def env_creator(args):
    # Pass render_mode=None for headless training (MUCH faster)
    env = FootballEnv(render_mode=None) 
    return ParallelPettingZooEnv(env)


def eval_episode(algo, save_dir, max_steps=1000):
    os.makedirs(save_dir, exist_ok=True)
    env = ParallelPettingZooEnv(FootballEnv(render_mode="rgb_array"))
    obs, _ = env.reset()
    frame_index = 0

    for _ in range(max_steps):
        if not obs:
            break
        actions = {}
        for agent_id, agent_obs in obs.items():
            action = algo.compute_single_action(agent_obs, policy_id="shared_policy", explore=False)
            actions[agent_id] = action
        obs, _, terminations, truncations, _ = env.step(actions)
        frame = env.render()
        if frame is not None:
            surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
            frame_path = os.path.join(save_dir, f"frame_{frame_index:06d}.png")
            pygame.image.save(surface, frame_path)
            frame_index += 1
        if all(terminations.values()) or all(truncations.values()):
            break

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    args = parser.parse_args()

    ray.init()
    checkpoint_root = os.path.join("runs", "checkpoints")
    os.makedirs(checkpoint_root, exist_ok=True)
    
    # 2. Register the environment
    env_name = "multiagent_football"
    register_env(env_name, env_creator)

    # 3. Configure the PPO Algorithm
    config = (
        PPOConfig()
        .environment(env=env_name)
        .env_runners(num_env_runners=2) # Uses 2 CPU cores for collecting data
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .multi_agent(
            # Default policy mapping: all agents share the same "brain" initially
            policies={"shared_policy"},
            policy_mapping_fn=lambda agent_id, episode, **kwargs: "shared_policy",
        )
        .training(
            train_batch_size=4000,
            minibatch_size=128,
        )
    )

    # 4. Build and Train
    algo = config.build_algo()
    if args.resume:
        algo.restore(args.resume)
        print(f"Resumed from checkpoint: {args.resume}")
    
    print("Starting Training Loop...")
    for i in range(100): # Train for 100 iterations
        result = algo.train()
        mean_reward = result.get("episode_reward_mean")
        if mean_reward is None:
            mean_reward = result.get("sampler_results", {}).get("episode_reward_mean")
        if mean_reward is None:
            mean_reward = result.get("evaluation", {}).get("episode_reward_mean")
        if mean_reward is None:
            mean_reward = "N/A"
            print(f"Iteration: {i} | Mean Reward: {mean_reward} | Keys: {list(result.keys())}")
        else:
            print(f"Iteration: {i} | Mean Reward: {mean_reward}")
        
        if i % 20 == 0:
            iter_checkpoint_root = os.path.join(checkpoint_root, f"iter_{i:03d}")
            os.makedirs(iter_checkpoint_root, exist_ok=True)
            checkpoint_dir = algo.save(iter_checkpoint_root)
            print(f"Saved checkpoint to {checkpoint_dir}")

            # Quick evaluation render
            eval_dir = os.path.join("runs", "frames", f"iter_{i:03d}")
            eval_episode(algo, save_dir=eval_dir)

    final_checkpoint_root = os.path.join(checkpoint_root, "final")
    os.makedirs(final_checkpoint_root, exist_ok=True)
    final_checkpoint_dir = algo.save(final_checkpoint_root)
    print(f"Final checkpoint saved to {final_checkpoint_dir}")

    ray.shutdown()