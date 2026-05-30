import os
import json
import argparse
import pygame
import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

from src.v1.football_env import CurriculumFootballEnv
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

    max_task = env_ctx.get("max_task", "11v11")

    if current_task == "1v1" and max_task != "1v1" and mean_reward > 50.0:
        print("Leveling up to 3v3!")
        return "3v3"
    elif current_task == "3v3" and max_task == "11v11" and mean_reward > 100.0:
        print("Leveling up to 11v11!")
        return "11v11"
    
    return current_task

ROLE_TO_POLICY = {
    AgentRole.GENERIC: "policy_generic",
    AgentRole.TARGET_MAN: "policy_target_man",
    AgentRole.PLAYMAKER: "policy_playmaker",
    AgentRole.ANCHOR_DEFENDER: "policy_anchor",
}

# Configure how each policy should be initialized when --init-from-map is used.
# - source_policy: which policy to copy weights from (within the checkpoint).
# - checkpoint: optional override path for this target policy.
DEFAULT_POLICY_INIT_MAP = {
    "policy_generic": {"source_policy": "policy_generic", "checkpoint": None},
    "policy_target_man": {"source_policy": "policy_generic", "checkpoint": None},
    "policy_playmaker": {"source_policy": "policy_generic", "checkpoint": None},
    "policy_anchor": {"source_policy": "policy_generic", "checkpoint": None},
}

def policy_mapping_generic(agent_id, episode, worker, **kwargs):
    return "policy_generic"

def policy_mapping_by_role(agent_id, episode, worker, **kwargs):
    env = worker.env.unwrapped
    base_env = env.par_env.unwrapped
    role = base_env.agent_roles[agent_id]
    return ROLE_TO_POLICY.get(role, "policy_generic")

def build_config(start_task, max_task, use_role_policies, use_curriculum, layout_path, layout_dir):
    policies = {"policy_generic"}
    if use_role_policies:
        policies.update({
            "policy_target_man",
            "policy_playmaker",
            "policy_anchor",
        })

    env_config = {"start_task": start_task, "max_task": max_task}
    if layout_path:
        env_config["layout_path"] = layout_path
    if layout_dir:
        env_config["layout_dir"] = layout_dir

    env_kwargs = {
        "env": "curriculum_football",
        "env_config": env_config,
    }
    if use_curriculum:
        env_kwargs["env_task_fn"] = curriculum_fn

    return (
        PPOConfig()
        .environment(**env_kwargs)
        .env_runners(num_env_runners=4, num_envs_per_env_runner=2)
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_by_role if use_role_policies else policy_mapping_generic,
        )
        .training(
            train_batch_size=8000,
            minibatch_size=256,
        )
    )

def save_eval_frames(algo, start_task, max_task, use_role_policies, frames_dir, max_steps, episode_id, layout_path, layout_dir):
    os.makedirs(frames_dir, exist_ok=True)
    env_config = {"start_task": start_task, "max_task": max_task}
    if layout_path:
        env_config["layout_path"] = layout_path
    if layout_dir:
        env_config["layout_dir"] = layout_dir
    env = CurriculumFootballEnv(render_mode="rgb_array", config=env_config)
    observations, _ = env.reset()
    done = False
    step = 0

    while not done and step < max_steps:
        actions = {}
        for agent_id, obs in observations.items():
            if use_role_policies:
                role = env.agent_roles[agent_id]
                policy_id = ROLE_TO_POLICY.get(role, "policy_generic")
            else:
                policy_id = "policy_generic"
            action = algo.compute_single_action(obs, policy_id=policy_id, explore=False)
            actions[agent_id] = action

        observations, _, terminations, truncations, _ = env.step(actions)
        frame = env.render()
        if frame is not None:
            frame_path = os.path.join(frames_dir, f"frame_{episode_id}_{step:04d}.png")
            surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
            pygame.image.save(surface, frame_path)

        done = any(terminations.values()) or any(truncations.values())
        step += 1

def load_policy_weights_from_checkpoint(checkpoint, policy_id, start_task, max_task, use_curriculum, layout_path, layout_dir):
    use_role_policies = policy_id != "policy_generic"
    temp_config = build_config(
        start_task,
        max_task,
        use_role_policies=use_role_policies,
        use_curriculum=use_curriculum,
        layout_path=layout_path,
        layout_dir=layout_dir,
    )
    temp_algo = temp_config.build_algo()
    temp_algo.restore(checkpoint)
    policy = temp_algo.get_policy(policy_id)
    if policy is None:
        temp_algo.stop()
        raise ValueError(f"Policy {policy_id} not found in checkpoint: {checkpoint}")
    weights = policy.get_weights()
    temp_algo.stop()
    return weights

def load_policy_init_map(path):
    if not path:
        return DEFAULT_POLICY_INIT_MAP
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Policy init map not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Policy init map must be a JSON object.")
    return data

def initialize_policies_from_map(algo, start_task, max_task, use_curriculum, default_checkpoint, policy_init_map, layout_path, layout_dir):
    cache = {}
    for target_policy, spec in policy_init_map.items():
        if not isinstance(spec, dict):
            raise ValueError(f"Policy init spec for {target_policy} must be an object.")
        source_policy = spec.get("source_policy", target_policy)
        checkpoint = spec.get("checkpoint") or default_checkpoint
        if checkpoint is None:
            continue
        if target_policy not in algo.workers.local_worker().policy_map:
            continue
        cache_key = (checkpoint, source_policy)
        if cache_key not in cache:
            cache[cache_key] = load_policy_weights_from_checkpoint(
                checkpoint,
                source_policy,
                start_task,
                max_task,
                use_curriculum,
                layout_path,
                layout_dir,
            )
        algo.get_policy(target_policy).set_weights(cache[cache_key])

def find_latest_checkpoint(checkpoint_dir):
    if not os.path.isdir(checkpoint_dir):
        return None
    candidates = []
    for name in os.listdir(checkpoint_dir):
        path = os.path.join(checkpoint_dir, name)
        if os.path.isdir(path) and name.startswith("checkpoint_"):
            candidates.append(path)
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p))
    return candidates[-1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-task", default="1v1")
    parser.add_argument("--max-task", default="3v3", choices=["1v1", "3v3", "11v11"])
    parser.add_argument("--role-policies", action="store_true")
    parser.add_argument("--auto-curriculum", action="store_true")
    parser.add_argument("--bootstrap-from", default=None)
    parser.add_argument("--init-from", default=None)
    parser.add_argument("--init-from-map", action="store_true")
    parser.add_argument("--force-init-map", action="store_true")
    parser.add_argument("--policy-map-file", default="src/v1/policy_init_map.json")
    parser.add_argument("--env-config-layout-path", default=None)
    parser.add_argument("--env-config-layout-dir", default=None)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--resume-latest", action="store_true")
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--checkpoint-dir", default="runs/checkpoints")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--save-frames", action="store_true")
    parser.add_argument("--frames-dir", default="runs/frames")
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--eval-steps", type=int, default=300)
    args = parser.parse_args()

    ray.init()
    register_env("curriculum_football", env_creator)

    use_curriculum = args.auto_curriculum

    if args.bootstrap_from and not args.init_from:
        args.init_from = args.bootstrap_from
    if args.bootstrap_from and not args.init_from_map:
        args.init_from_map = True

    config = build_config(
        args.start_task,
        args.max_task,
        use_role_policies=args.role_policies,
        use_curriculum=use_curriculum,
        layout_path=args.env_config_layout_path,
        layout_dir=args.env_config_layout_dir,
    )
    algo = config.build_algo()

    if args.resume_latest and args.resume_from:
        raise ValueError("Use only one of --resume-from or --resume-latest.")

    resume_checkpoint = args.resume_from
    if args.resume_latest:
        resume_checkpoint = find_latest_checkpoint(args.checkpoint_dir)

    if resume_checkpoint:
        algo.restore(resume_checkpoint)
        print(f"Resumed from checkpoint: {resume_checkpoint}")

    if args.init_from_map:
        if resume_checkpoint and not args.force_init_map:
            print("Init map skipped because resume is active (use --force-init-map to override).")
        else:
            policy_init_map = load_policy_init_map(args.policy_map_file)
            initialize_policies_from_map(
                algo,
                args.start_task,
                args.max_task,
                use_curriculum,
                args.init_from,
                policy_init_map,
                args.env_config_layout_path,
                args.env_config_layout_dir,
            )

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for i in range(args.iters):
        result = algo.train()
        print(f"Iter: {i} | Mean Reward: {result.get('episode_reward_mean')}")
        if args.checkpoint_every > 0 and (i + 1) % args.checkpoint_every == 0:
            checkpoint = algo.save(args.checkpoint_dir)
            print(f"Saved checkpoint: {checkpoint}")
        if args.save_frames and args.eval_every > 0 and (i + 1) % args.eval_every == 0:
            iter_dir = os.path.join(args.frames_dir, f"iter_{i + 1:04d}")
            for ep in range(args.eval_episodes):
                save_eval_frames(
                    algo,
                    args.start_task,
                    args.max_task,
                    args.role_policies,
                    iter_dir,
                    args.eval_steps,
                    ep,
                    args.env_config_layout_path,
                    args.env_config_layout_dir,
                )

    final_checkpoint = algo.save(args.checkpoint_dir)
    print(f"Saved final checkpoint: {final_checkpoint}")