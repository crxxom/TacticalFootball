import os
import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

from src.v0.football_env import CurriculumFootballEnv
from config import AgentRole

def env_creator(config):
    env = CurriculumFootballEnv(render_mode=None, config=config)
    return ParallelPettingZooEnv(env)

def curriculum_fn(train_results, task_settable_env, env_ctx):
    """
    Evaluates training results after every iteration. If the agents perform well, 
    level up the environment to the next stage.
    """
    current_task = task_settable_env.get_task()
    mean_reward = train_results.get("episode_reward_mean", 0)

    if current_task == "1v1" and mean_reward > 50.0:
        print("Leveling up to 3v3!")
        return "3v3"
    elif current_task == "3v3" and mean_reward > 100.0:
        print("Leveling up to 11v11!")
        return "11v11"
    
    return current_task

def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    """Maps an agent in the environment to a specific Neural Network"""
    # Look up the role based on how we named the agent in CurriculumFootballEnv
    # e.g., if agent_id is "blue_0", we need to check the active curriculum config
    
    # For simplicity, we can route policies strictly by role strings:
    env = worker.env.unwrapped
    # Access the base CurriculumFootballEnv to find this agent's role
    base_env = env.par_env.unwrapped
    role = base_env.agent_roles[agent_id]
    
    if role == AgentRole.TARGET_MAN:
        return "policy_target_man"
    elif role == AgentRole.PLAYMAKER:
        return "policy_playmaker"
    return "policy_anchor"

if __name__ == "__main__":
    ray.init()
    register_env("curriculum_football", env_creator)

    config = (
        PPOConfig()
        .environment(
            env="curriculum_football",
            env_config={"start_task": "1v1"},
            env_task_fn=curriculum_fn # Attach curriculum callback
        )
        .env_runners(num_env_runners=4, num_envs_per_env_runner=2)
        .multi_agent(
            # Define distinct neural networks for each role
            policies={"policy_target_man", "policy_playmaker", "policy_anchor"},
            policy_mapping_fn=policy_mapping_fn,
        )
        .training(
            train_batch_size=8000,
            minibatch_size=256,
        )
    )

    algo = config.build_algo()
    
    for i in range(500):
        result = algo.train()
        print(f"Iter: {i} | Mean Reward: {result.get('episode_reward_mean')}")