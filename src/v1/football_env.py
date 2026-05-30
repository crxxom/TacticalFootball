import numpy as np
import pygame
import math
import functools
from pettingzoo import ParallelEnv
from gymnasium.spaces import Box
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv

# Imports from your modular architecture
from config import (
    CURRICULUM, WIDTH, HEIGHT, MAX_TEAMMATES, MAX_ENEMIES,
    FPS, DT, MAX_PLAYER_SPEED, MAX_SHOT_SPEED, FRICTION,
    PLAYER_RADIUS, GOAL_HALF_HEIGHT, BALL_RADIUS, BALL_COLOR,
    BALL_OUTLINE_COLOR, BALL_OWNER_OFFSET, POSSESSION_DISTANCE_BUFFER,
    POSSESSION_COOLDOWN_FRAMES, ROTATION_SPEED, PRESSURE_RADIUS,
    PASS_BASE_ANGLE_ERROR, PASS_POWER_ANGLE_ERROR, PASS_PRESSURE_ANGLE_ERROR,
    PASS_FACING_ANGLE_ERROR, PASS_BASE_POWER_ERROR, PASS_POWER_POWER_ERROR,
    PASS_PRESSURE_POWER_ERROR, PASS_FACING_POWER_ERROR, TEAM_REWARD_WEIGHT,
    load_stage_layout
)
from observations import DynamicObservationBuilder
from rewards import RoleBasedRewardCalculator

def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))

def angle_diff(a, b):
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)

class CurriculumFootballEnv(ParallelEnv, TaskSettableEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "name": "football_curriculum_v1"}

    def __init__(self, render_mode=None, config=None):
        self.render_mode = render_mode
        self.env_config = config or {}
        self.current_task = self.env_config.get("start_task", "1v1")
        self.layout_path = self.env_config.get("layout_path")
        self.layout_dir = self.env_config.get("layout_dir")
        self.width = WIDTH
        self.height = HEIGHT
        
        self.screen = None
        self.render_surface = None
        self.obs_builder = DynamicObservationBuilder(self.width, self.height)
        self.reward_calc = RoleBasedRewardCalculator(self.width, self.height)
        
        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            self.clock = pygame.time.Clock()
        elif self.render_mode == "rgb_array":
            pygame.init()
            self.render_surface = pygame.Surface((self.width, self.height))

        self._setup_agents()

    def _setup_agents(self):
        self.stage_config = CURRICULUM[self.current_task]
        self.possible_agents = []
        self.agent_roles = {}
        
        for i in range(self.stage_config.team_size):
            name = f"blue_{i}"
            role = self.stage_config.blue_roles[i]
            self.possible_agents.append(name)
            self.agent_roles[name] = role
            
        for i in range(self.stage_config.team_size):
            name = f"red_{i}"
            role = self.stage_config.red_roles[i]
            self.possible_agents.append(name)
            self.agent_roles[name] = role

        self.agent_ids = list(self.possible_agents)
        self.agent_index = {agent_id: idx for idx, agent_id in enumerate(self.agent_ids)}
        self.num_agents = len(self.agent_ids)
        
        obs_dim = 16 + (MAX_TEAMMATES * 4) + (MAX_ENEMIES * 4)
        
        self.action_spaces = {a: Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32) for a in self.possible_agents}
        self.observation_spaces = {a: Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32) for a in self.possible_agents}

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]

    def get_task(self):
        return self.current_task

    def set_task(self, task: str):
        self.current_task = task
        self._setup_agents()

    def _refresh_render_targets(self):
        if self.render_mode == "human":
            self.screen = pygame.display.set_mode((self.width, self.height))
        elif self.render_mode == "rgb_array":
            self.render_surface = pygame.Surface((self.width, self.height))

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.num_moves = 0

        layout = load_stage_layout(self.current_task, layout_path=self.layout_path, layout_dir=self.layout_dir)
        new_width = int(layout["pitch"]["width"])
        new_height = int(layout["pitch"]["height"])
        if new_width != self.width or new_height != self.height:
            self.width = new_width
            self.height = new_height
            self.obs_builder = DynamicObservationBuilder(self.width, self.height)
            self.reward_calc = RoleBasedRewardCalculator(self.width, self.height)
            self._refresh_render_targets()
        
        self.positions = np.zeros((self.num_agents, 2), dtype=np.float32)
        self.velocities = np.zeros((self.num_agents, 2), dtype=np.float32)
        self.teams = np.zeros(self.num_agents, dtype=np.int32)
        self.facing_angle = np.zeros(self.num_agents, dtype=np.float32)
        self.facing = np.zeros((self.num_agents, 2), dtype=np.float32)

        blue_idx = 0
        red_idx = 0
        
        # Dynamic Spawning Spread across the pitch
        for idx, agent_id in enumerate(self.agents):
            if "blue" in agent_id:
                self.teams[idx] = 0
                y_offset = (blue_idx - self.stage_config.team_size / 2.0) * 80
                default_pos = [self.width // 4 + np.random.randint(-50, 50), self.height // 2 + y_offset]
                blue_idx += 1
            else:
                self.teams[idx] = 1
                y_offset = (red_idx - self.stage_config.team_size / 2.0) * 80
                default_pos = [self.width * 3 // 4 + np.random.randint(-50, 50), self.height // 2 + y_offset]
                red_idx += 1

            if agent_id in layout["players"]:
                pos = layout["players"][agent_id]
                self.positions[idx] = [float(pos[0]), float(pos[1])]
            else:
                self.positions[idx] = default_pos

        self.positions[:, 0] = np.clip(self.positions[:, 0], PLAYER_RADIUS, self.width - PLAYER_RADIUS)
        self.positions[:, 1] = np.clip(self.positions[:, 1], PLAYER_RADIUS, self.height - PLAYER_RADIUS)
                
        self.facing[:, 0] = 1.0
        self.facing[:, 1] = 0.0

        self.ball = {
            "x": float(layout["ball"]["x"]),
            "y": float(layout["ball"]["y"]),
            "vx": 0.0,
            "vy": 0.0,
            "owner_idx": -1,
            "cooldown": 0,
            "last_touch_idx": -1,
            "last_touch_step": 0,
            "last_kick_idx": -1,
            "last_kick_team": None,
            "last_kick_goal_dist": None,
            "last_kick_pressure": 0.0,
            "last_passer_idx": -1,
            "owner_offset_x": BALL_OWNER_OFFSET,
            "owner_offset_y": 0.0,
            "step_count": 0,
            "prev_goal_dist_team0": math.hypot(self.width // 2 - self.width, self.height // 2 - self.height / 2.0),
            "prev_goal_dist_team1": math.hypot(self.width // 2 - 0.0, self.height // 2 - self.height / 2.0),
        }

        self.ball["x"] = clamp(self.ball["x"], BALL_RADIUS, self.width - BALL_RADIUS)
        self.ball["y"] = clamp(self.ball["y"], BALL_RADIUS, self.height - BALL_RADIUS)

        observations = {agent: self._get_local_observation(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def _get_local_observation(self, agent_id):
        index = self.agent_index[agent_id]
        return self.obs_builder.build(index, self.positions, self.velocities, self.teams, self.ball, self.facing)

    def step(self, actions):
        self.num_moves += 1
        active_agents = list(self.agents)
        rewards = np.zeros(self.num_agents, dtype=np.float32)
        event_flags = {
            agent_id: {
                "scored_goal": False,
                "took_shot_in_box": False,
                "completed_pass": False,
                "assisted_goal": False,
                "successful_tackle": False,
            }
            for agent_id in self.agents
        }
        
        prev_owner_idx = self.ball["owner_idx"]
        prev_owner_team = self.teams[prev_owner_idx] if prev_owner_idx != -1 else None

        action_array = np.zeros((self.num_agents, 6), dtype=np.float32)
        for agent_id, action in actions.items():
            if agent_id in self.agent_index:
                action_array[self.agent_index[agent_id]] = action

        move = action_array[:, 0:2].copy()
        lengths = np.linalg.norm(move, axis=1)
        scale = np.ones_like(lengths)
        mask = lengths > 1.0
        scale[mask] = 1.0 / lengths[mask]
        move *= scale[:, None]

        self.velocities = move * MAX_PLAYER_SPEED
        self.positions += self.velocities * DT

        self.facing_angle += action_array[:, 5] * ROTATION_SPEED
        self.facing[:, 0] = np.cos(self.facing_angle)
        self.facing[:, 1] = np.sin(self.facing_angle)

        self.positions[:, 0] = np.clip(self.positions[:, 0], PLAYER_RADIUS, self.width - PLAYER_RADIUS)
        self.positions[:, 1] = np.clip(self.positions[:, 1], PLAYER_RADIUS, self.height - PLAYER_RADIUS)

        # Base Vectorized Rewards
        rewards += self.reward_calc.step_rewards(self.positions, self.teams, self.ball)
        speeds = np.linalg.norm(self.velocities, axis=1)
        rewards += self.reward_calc.movement_reward(speeds)
        rewards += self.reward_calc.possession_reward(self.teams, self.ball)

        # Shooting Mechanics
        shoot_vectors = np.linalg.norm(action_array[:, 2:4], axis=1)
        shot_powers = np.clip(action_array[:, 4], 0.0, 1.0)
        shooter_indices = np.where((shot_powers > 0.0) & (self.ball["owner_idx"] == np.arange(self.num_agents)) & (self.ball["cooldown"] == 0))[0]

        for shooter in shooter_indices:
            team = self.teams[shooter]
            opponent_mask = self.teams != team
            deltas = self.positions[opponent_mask] - self.positions[shooter]
            if deltas.size:
                dists = np.hypot(deltas[:, 0], deltas[:, 1])
                pressures = 1.0 - (dists / PRESSURE_RADIUS)
                pressure = float(np.max(np.where(dists < PRESSURE_RADIUS, pressures, 0.0)))
            else:
                pressure = 0.0

            shoot_vec = action_array[shooter, 2:4]
            shot_angle = math.atan2(shoot_vec[1], shoot_vec[0]) if shoot_vectors[shooter] > 0 else float(self.facing_angle[shooter])
            facing_offset = angle_diff(shot_angle, float(self.facing_angle[shooter])) / math.pi
            angle_error = (
                PASS_BASE_ANGLE_ERROR
                + shot_powers[shooter] * PASS_POWER_ANGLE_ERROR
                + pressure * PASS_PRESSURE_ANGLE_ERROR
                + facing_offset * PASS_FACING_ANGLE_ERROR
            )
            power_error = (
                PASS_BASE_POWER_ERROR
                + shot_powers[shooter] * PASS_POWER_POWER_ERROR
                + pressure * PASS_PRESSURE_POWER_ERROR
                + facing_offset * PASS_FACING_POWER_ERROR
            )
            shot_angle = shot_angle + np.random.normal(0.0, angle_error)
            shot_power = clamp(shot_powers[shooter] * (1.0 + np.random.normal(0.0, power_error)), 0.0, 1.0)
            dir_x = math.cos(shot_angle)
            dir_y = math.sin(shot_angle)

            if team == 0:
                in_box = (self.positions[shooter, 0] > self.width - 200) and (abs(self.positions[shooter, 1] - self.height / 2.0) < GOAL_HALF_HEIGHT * 1.5)
            else:
                in_box = (self.positions[shooter, 0] < 200) and (abs(self.positions[shooter, 1] - self.height / 2.0) < GOAL_HALF_HEIGHT * 1.5)
            if in_box:
                event_flags[self.agent_ids[shooter]]["took_shot_in_box"] = True

            self.ball["owner_idx"] = -1
            self.ball["cooldown"] = 0
            self.ball["last_touch_idx"] = int(shooter)
            self.ball["last_touch_step"] = self.num_moves
            self.ball["last_kick_idx"] = int(shooter)
            self.ball["last_kick_team"] = int(team)
            self.ball["last_kick_pressure"] = float(pressure)
            enemy_goal_x = self.width if team == 0 else 0.0
            enemy_goal_y = self.height / 2.0
            self.ball["last_kick_goal_dist"] = math.hypot(self.ball["x"] - enemy_goal_x, self.ball["y"] - enemy_goal_y)
            self.ball["vx"] = dir_x * MAX_SHOT_SPEED * shot_power
            self.ball["vy"] = dir_y * MAX_SHOT_SPEED * shot_power

        # Prevent player overlap
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                dx = self.positions[j, 0] - self.positions[i, 0]
                dy = self.positions[j, 1] - self.positions[i, 1]
                dist = math.hypot(dx, dy)
                min_dist = PLAYER_RADIUS * 2
                if dist == 0:
                    dx, dy = 1.0, 0.0
                    dist = 1.0
                if dist < min_dist:
                    overlap = (min_dist - dist) / 2.0
                    nx = dx / dist
                    ny = dy / dist
                    self.positions[i, 0] -= nx * overlap
                    self.positions[i, 1] -= ny * overlap
                    self.positions[j, 0] += nx * overlap
                    self.positions[j, 1] += ny * overlap
                    self.positions[i, 0] = max(PLAYER_RADIUS, min(self.width - PLAYER_RADIUS, self.positions[i, 0]))
                    self.positions[i, 1] = max(PLAYER_RADIUS, min(self.height - PLAYER_RADIUS, self.positions[i, 1]))
                    self.positions[j, 0] = max(PLAYER_RADIUS, min(self.width - PLAYER_RADIUS, self.positions[j, 0]))
                    self.positions[j, 1] = max(PLAYER_RADIUS, min(self.height - PLAYER_RADIUS, self.positions[j, 1]))

        # Ball Physics and Possession
        if self.ball["cooldown"] > 0:
            self.ball["cooldown"] -= 1

        if self.ball["owner_idx"] != -1:
            owner = self.positions[self.ball["owner_idx"]]
            self.ball["x"] = owner[0] + self.ball["owner_offset_x"]
            self.ball["y"] = owner[1] + self.ball["owner_offset_y"]
            self.ball["vx"] = 0.0
            self.ball["vy"] = 0.0
        else:
            self.ball["x"] += self.ball["vx"] * DT
            self.ball["y"] += self.ball["vy"] * DT
            self.ball["vx"] *= FRICTION
            self.ball["vy"] *= FRICTION

        if self.ball["cooldown"] == 0:
            for idx in range(self.num_agents):
                if idx == self.ball["owner_idx"]:
                    continue
                dist_to_ball = math.hypot(self.positions[idx, 0] - self.ball["x"], self.positions[idx, 1] - self.ball["y"])
                if dist_to_ball < PLAYER_RADIUS + BALL_RADIUS + POSSESSION_DISTANCE_BUFFER:
                    self.ball["owner_idx"] = idx
                    self.ball["cooldown"] = POSSESSION_COOLDOWN_FRAMES
                    self.ball["last_touch_idx"] = idx
                    self.ball["last_touch_step"] = self.num_moves
                    pass_completed = (
                        self.ball.get("last_kick_idx", -1) != -1
                        and self.ball.get("last_kick_team") is not None
                        and self.ball["last_kick_idx"] != idx
                        and self.ball["last_kick_team"] == int(self.teams[idx])
                    )
                    if pass_completed:
                        self.ball["last_passer_idx"] = self.ball["last_kick_idx"]
                        passer_id = self.agent_ids[self.ball["last_kick_idx"]]
                        event_flags[passer_id]["completed_pass"] = True
                    elif self.ball.get("last_kick_team") is not None and self.ball["last_kick_team"] != int(self.teams[idx]):
                        self.ball["last_passer_idx"] = -1

                    rewards += self.reward_calc.pass_rewards(self.ball, self.teams, idx)
                    rewards += self.reward_calc.tackle_rewards(self.teams, prev_owner_team, idx)
                    if prev_owner_team is not None and int(self.teams[idx]) != int(prev_owner_team):
                        event_flags[self.agent_ids[idx]]["successful_tackle"] = True

                    self.ball["last_kick_idx"] = -1
                    self.ball["last_kick_team"] = None
                    self.ball["last_kick_goal_dist"] = None
                    self.ball["last_kick_pressure"] = 0.0

                    offset_x = self.ball["x"] - self.positions[idx, 0]
                    offset_y = self.ball["y"] - self.positions[idx, 1]
                    offset_len = math.hypot(offset_x, offset_y)
                    if offset_len == 0:
                        offset_x = BALL_OWNER_OFFSET
                        offset_y = 0.0
                        offset_len = BALL_OWNER_OFFSET
                    scale = BALL_OWNER_OFFSET / offset_len
                    self.ball["owner_offset_x"] = offset_x * scale
                    self.ball["owner_offset_y"] = offset_y * scale
                    break

        # Check Goals / Out of Bounds
        truncations = {a: False for a in self.agents}
        goal_rewards, terminated = self.reward_calc.goal_rewards(self.teams, self.ball)
        rewards += goal_rewards
        if goal_rewards.any():
            scorer_idx = self.ball.get("last_touch_idx", -1)
            if scorer_idx != -1:
                event_flags[self.agent_ids[scorer_idx]]["scored_goal"] = True
            assister_idx = self.ball.get("last_passer_idx", -1)
            if assister_idx != -1:
                event_flags[self.agent_ids[assister_idx]]["assisted_goal"] = True
        out_rewards, out_of_bounds = self.reward_calc.out_of_bounds_rewards(self.ball, self.num_agents)
        rewards += out_rewards
        if out_of_bounds and not terminated:
            terminated = True

        if self.num_moves > 1000:
            truncations = {a: True for a in self.agents}

        rewards += self.reward_calc.ball_progress_rewards(self.ball, self.teams)
        
        # --- Inject Role-Based Reward Modifiers ---
        for agent_id in self.agents:
            idx = self.agent_index[agent_id]
            role = self.agent_roles[agent_id]
            modifier = self.reward_calc.calculate_role_modifiers(role, {}, event_flags[agent_id])
            rewards[idx] += modifier
        # -------------------------------------------

        self.ball["step_count"] = self.num_moves
        self.ball["prev_goal_dist_team0"] = math.hypot(self.ball["x"] - self.width, self.ball["y"] - self.height / 2.0)
        self.ball["prev_goal_dist_team1"] = math.hypot(self.ball["x"] - 0.0, self.ball["y"] - self.height / 2.0)

        if terminated or any(truncations.values()):
            self.agents = []

        team_totals = {0: float(np.sum(rewards[self.teams == 0])), 1: float(np.sum(rewards[self.teams == 1]))}
        for idx in range(self.num_agents):
            team = int(self.teams[idx])
            opponent = 1 - team
            team_delta = team_totals[team] - team_totals[opponent]
            rewards[idx] = rewards[idx] + (TEAM_REWARD_WEIGHT * team_delta)

        observations = {a: self._get_local_observation(a) for a in active_agents}
        infos = {a: {} for a in active_agents}

        if self.render_mode == "human":
            self.render()

        terminations = {a: terminated for a in active_agents}
        reward_dict = {a: float(rewards[self.agent_index[a]]) for a in active_agents}
        truncations = {a: truncations[a] for a in active_agents}
        return observations, reward_dict, terminations, truncations, infos

    def _draw(self, surface):
        surface.fill((50, 150, 50))
        pygame.draw.rect(surface, (255, 255, 255), (0, 0, self.width, self.height), 2)
        pygame.draw.line(surface, (255, 255, 255), (self.width // 2, 0), (self.width // 2, self.height), 2)
        pygame.draw.circle(surface, (255, 255, 255), (self.width // 2, self.height // 2), 60, 2)
        pygame.draw.rect(surface, (200, 200, 50), (-10, self.height // 2 - 50, 20, 100))
        pygame.draw.rect(surface, (200, 200, 50), (self.width - 10, self.height // 2 - 50, 20, 100))

        for idx in range(self.num_agents):
            color = (50, 100, 200) if self.teams[idx] == 0 else (200, 50, 50)
            pygame.draw.circle(surface, color, (int(self.positions[idx, 0]), int(self.positions[idx, 1])), PLAYER_RADIUS)

        pygame.draw.circle(surface, BALL_COLOR, (int(self.ball["x"]), int(self.ball["y"])), BALL_RADIUS)
        pygame.draw.circle(surface, BALL_OUTLINE_COLOR, (int(self.ball["x"]), int(self.ball["y"])), BALL_RADIUS, 1)

    def render(self, render_mode=None):
        mode = render_mode or self.render_mode
        if mode == "human":
            self._draw(self.screen)
            pygame.display.flip()
            self.clock.tick(FPS)
            return None

        if mode == "rgb_array":
            if self.render_surface is None:
                self.render_surface = pygame.Surface((self.width, self.height))
            self._draw(self.render_surface)
            frame = pygame.surfarray.array3d(self.render_surface)
            return np.transpose(frame, (1, 0, 2))

        return None