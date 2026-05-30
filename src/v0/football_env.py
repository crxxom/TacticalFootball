import numpy as np
import pygame
import math
from pettingzoo import ParallelEnv
from gymnasium.spaces import Box
import functools

# --- Constants ---
WIDTH, HEIGHT = 1280, 800
FPS = 30
DT = 1.0 / FPS
MAX_PLAYER_SPEED = 220.0
MAX_SHOT_SPEED = 380.0
FRICTION = 0.98
PLAYER_RADIUS = 14
GOAL_HALF_HEIGHT = 70
BALL_RADIUS = 7
BALL_COLOR = (0, 0, 0)
BALL_OUTLINE_COLOR = (255, 255, 255)
BALL_OWNER_OFFSET = PLAYER_RADIUS + BALL_RADIUS
POSSESSION_DISTANCE_BUFFER = 6
POSSESSION_COOLDOWN_FRAMES = 10

OUT_OF_BOUNDS_PENALTY = -10.0
GOAL_REWARD = 100.0
GOAL_TEAM_REWARD = 10.0
GOAL_ASSIST_REWARD = 30.0
GOAL_CONCEDE_PENALTY = 10.0
POSSESSION_REWARD = 0.005
BALL_PROGRESS_REWARD_SCALE = 0.02
PASS_FIXED_REWARD = 2.0
PASS_PROGRESS_SCALE = 0.01
PASS_INTERCEPT_PENALTY = -2.0
PASS_RECEIVE_REWARD = 1.0
PASS_PRESSURE_BONUS_SCALE = 1.0
TACKLE_REWARD = 10
MOVE_REWARD = 0.0005
TEAM_REWARD_WEIGHT = 0.5


ROTATION_SPEED = math.radians(10.0)
PRESSURE_RADIUS = 90.0
PASS_BASE_ANGLE_ERROR = math.radians(0.5)
PASS_POWER_ANGLE_ERROR = math.radians(1.5)
PASS_PRESSURE_ANGLE_ERROR = math.radians(3.0)
PASS_FACING_ANGLE_ERROR = math.radians(3.0)
PASS_BASE_POWER_ERROR = 0.01
PASS_POWER_POWER_ERROR = 0.04
PASS_PRESSURE_POWER_ERROR = 0.06
PASS_FACING_POWER_ERROR = 0.05


class ObservationBuilder:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def build(self, index, positions, velocities, teams, ball, facing):
        pos = positions[index]
        vel = velocities[index]
        team = teams[index]
        obs = []
        pitch_diag = math.hypot(self.width, self.height)

        obs.extend([vel[0], vel[1]])

        my_goal_x = 0.0 if team == 0 else self.width
        my_goal_y = self.height / 2.0
        obs.extend([my_goal_x - pos[0], my_goal_y - pos[1]])

        enemy_goal_x = self.width if team == 0 else 0.0
        enemy_goal_y = self.height / 2.0
        obs.extend([enemy_goal_x - pos[0], enemy_goal_y - pos[1]])

        obs.extend([
            ball["x"] - pos[0],
            ball["y"] - pos[1],
            ball["vx"],
            ball["vy"],
        ])

        obs.extend([
            pos[1],
            self.height - pos[1],
            pos[0],
            self.width - pos[0],
        ])

        deltas = positions - pos
        dists = np.hypot(deltas[:, 0], deltas[:, 1])
        idxs = np.arange(len(teams))
        teammate_mask = (teams == team) & (idxs != index)
        enemy_mask = teams != team

        teammate_indices = idxs[teammate_mask]
        enemy_indices = idxs[enemy_mask]

        teammate_order = teammate_indices[np.argsort(dists[teammate_indices])] if teammate_indices.size else np.array([], dtype=int)
        enemy_order = enemy_indices[np.argsort(dists[enemy_indices])] if enemy_indices.size else np.array([], dtype=int)

        for k in range(2):
            if k < teammate_order.size:
                j = teammate_order[k]
                obs.extend([deltas[j, 0], deltas[j, 1], velocities[j, 0], velocities[j, 1]])
            else:
                obs.extend([0.0, 0.0, 0.0, 0.0])

        for k in range(3):
            if k < enemy_order.size:
                j = enemy_order[k]
                obs.extend([deltas[j, 0], deltas[j, 1], velocities[j, 0], velocities[j, 1]])
            else:
                obs.extend([0.0, 0.0, 0.0, 0.0])

        owner_idx = ball.get("owner_idx", -1)
        ownership = 0.0
        if owner_idx != -1:
            ownership = 1.0 if teams[owner_idx] == team else -1.0
        obs.append(ownership)

        ball_speed = math.hypot(ball["vx"], ball["vy"])
        ball_heading = math.atan2(ball["vy"], ball["vx"]) if ball_speed > 0 else 0.0
        obs.extend([math.cos(ball_heading), math.sin(ball_heading)])
        obs.append(ball_speed / MAX_SHOT_SPEED)

        obs.extend([facing[index, 0], facing[index, 1]])

        nearest_teammate = dists[teammate_order[0]] if teammate_order.size else pitch_diag
        nearest_opponent = dists[enemy_order[0]] if enemy_order.size else pitch_diag
        obs.append(nearest_teammate / pitch_diag)
        obs.append(nearest_opponent / pitch_diag)

        cooldown_norm = ball.get("cooldown", 0) / max(1, POSSESSION_COOLDOWN_FRAMES)
        obs.append(cooldown_norm)
        steps_since_touch = ball.get("step_count", 0) - ball.get("last_touch_step", 0)
        obs.append(min(1.0, steps_since_touch / (FPS * 5)))

        return np.array(obs, dtype=np.float32)

class RewardCalculator:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def boundary_penalty(self, positions):
        penalties = np.zeros(len(positions), dtype=np.float32)
        out = (
            (positions[:, 0] < PLAYER_RADIUS)
            | (positions[:, 0] > self.width - PLAYER_RADIUS)
            | (positions[:, 1] < PLAYER_RADIUS)
            | (positions[:, 1] > self.height - PLAYER_RADIUS)
        )
        penalties[out] = -0.05
        return penalties

    def approach_ball_reward(self, positions, ball):
        dists = np.hypot(positions[:, 0] - ball["x"], positions[:, 1] - ball["y"])
        return -0.001 * dists

    def forward_position_penalty(self, positions, teams):
        return np.zeros(len(positions), dtype=np.float32)

    def step_rewards(self, positions, teams, ball):
        return (
            self.boundary_penalty(positions)
            + self.approach_ball_reward(positions, ball)
            + self.forward_position_penalty(positions, teams)
        )

    def movement_reward(self, speeds):
        return MOVE_REWARD * (speeds / MAX_PLAYER_SPEED)

    def possession_reward(self, teams, ball):
        rewards = np.zeros(len(teams), dtype=np.float32)
        owner_idx = ball.get("owner_idx", -1)
        if owner_idx != -1:
            owner_team = teams[owner_idx]
            rewards[teams == owner_team] = POSSESSION_REWARD
        return rewards

    def ball_progress_rewards(self, ball, teams):
        rewards = np.zeros(len(teams), dtype=np.float32)
        dist_team0 = ball.get("prev_goal_dist_team0")
        dist_team1 = ball.get("prev_goal_dist_team1")
        if dist_team0 is None or dist_team1 is None:
            return rewards

        new_dist_team0 = math.hypot(ball["x"] - WIDTH, ball["y"] - HEIGHT / 2.0)
        new_dist_team1 = math.hypot(ball["x"] - 0.0, ball["y"] - HEIGHT / 2.0)
        progress_team0 = dist_team0 - new_dist_team0
        progress_team1 = dist_team1 - new_dist_team1

        rewards[teams == 0] += BALL_PROGRESS_REWARD_SCALE * progress_team0
        rewards[teams == 1] += BALL_PROGRESS_REWARD_SCALE * progress_team1
        return rewards

    def goal_rewards(self, teams, ball):
        rewards = np.zeros(len(teams), dtype=np.float32)
        terminations = False
        scorer_idx = ball.get("last_touch_idx", -1)
        assist_idx = ball.get("last_passer_idx", -1)
        scoring_team = None

        if ball["x"] < 0 and (self.height // 2 - GOAL_HALF_HEIGHT < ball["y"] < self.height // 2 + GOAL_HALF_HEIGHT):
            scoring_team = 1
            terminations = True
        elif ball["x"] > self.width and (self.height // 2 - GOAL_HALF_HEIGHT < ball["y"] < self.height // 2 + GOAL_HALF_HEIGHT):
            scoring_team = 0
            terminations = True

        if scoring_team is not None:
            rewards[teams == scoring_team] += GOAL_TEAM_REWARD
            rewards[teams != scoring_team] -= GOAL_CONCEDE_PENALTY

            if scorer_idx != -1:
                if teams[scorer_idx] == scoring_team:
                    rewards[scorer_idx] += GOAL_REWARD
                else:
                    rewards[scorer_idx] -= GOAL_REWARD * 10

            if assist_idx != -1 and assist_idx != scorer_idx and teams[assist_idx] == scoring_team:
                rewards[assist_idx] += GOAL_ASSIST_REWARD

        return rewards, terminations

    def pass_rewards(self, ball, teams, new_owner_idx):
        rewards = np.zeros(len(teams), dtype=np.float32)
        if new_owner_idx == -1:
            return rewards

        new_team = teams[new_owner_idx]
        last_kick_idx = ball.get("last_kick_idx", -1)
        last_kick_team = ball.get("last_kick_team")

        if last_kick_idx != -1 and last_kick_team is not None:
            if new_team == last_kick_team and last_kick_idx != new_owner_idx:
                enemy_goal_x = WIDTH if new_team == 0 else 0.0
                enemy_goal_y = HEIGHT / 2.0
                new_dist = math.hypot(ball["x"] - enemy_goal_x, ball["y"] - enemy_goal_y)
                prev_dist = ball.get("last_kick_goal_dist") or new_dist
                progress = prev_dist - new_dist
                rewards[last_kick_idx] += PASS_FIXED_REWARD + (PASS_PROGRESS_SCALE * progress)
                rewards[last_kick_idx] += PASS_PRESSURE_BONUS_SCALE * (ball.get("last_kick_pressure") or 0.0)
                rewards[new_owner_idx] += PASS_RECEIVE_REWARD
            elif new_team != last_kick_team:
                rewards[last_kick_idx] += PASS_INTERCEPT_PENALTY

        return rewards

    def tackle_rewards(self, teams, prev_owner_team, new_owner_idx):
        rewards = np.zeros(len(teams), dtype=np.float32)
        if new_owner_idx == -1:
            return rewards

        new_team = teams[new_owner_idx]
        if prev_owner_team is not None and new_team != prev_owner_team:
            rewards[new_owner_idx] += TACKLE_REWARD

        return rewards

    def out_of_bounds_rewards(self, ball, num_agents):
        rewards = np.zeros(num_agents, dtype=np.float32)
        out_of_bounds = (
            ball["x"] < 0
            or ball["x"] > WIDTH
            or ball["y"] < 0
            or ball["y"] > HEIGHT
        )
        last_touch_idx = ball.get("last_touch_idx", -1)
        if out_of_bounds and last_touch_idx != -1:
            rewards[last_touch_idx] += OUT_OF_BOUNDS_PENALTY

        return rewards, out_of_bounds


def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def angle_diff(a, b):
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


class FootballEnv(ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "name": "football_v1"}

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        
        # Heterogeneous agents for specialized roles
        self.possible_agents = [
            "blue_forward", "blue_midfielder", "blue_defender", 
            "red_forward", "red_midfielder", "red_defender"
        ]
        
        # Action Space: [move_x, move_y, shoot_x, shoot_y, shoot_power, rotate] (-1.0 to 1.0)
        self.action_spaces = {a: Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32) for a in self.possible_agents}
        
        # Observation Space: 44 dimensions
        # Self Vel (2) + My Goal (2) + Enemy Goal (2) + Ball (4) + Boundaries (4) + 2 Teammates (8) + 3 Enemies (12)
        self.observation_spaces = {a: Box(low=-np.inf, high=np.inf, shape=(44,), dtype=np.float32) for a in self.possible_agents}
        
        self.screen = None
        self.render_surface = None
        self.obs_builder = ObservationBuilder(WIDTH, HEIGHT)
        self.reward_calc = RewardCalculator(WIDTH, HEIGHT)
        self.agent_ids = list(self.possible_agents)
        self.agent_index = {agent_id: idx for idx, agent_id in enumerate(self.agent_ids)}
        self.num_agents = len(self.agent_ids)
        self.positions = np.zeros((self.num_agents, 2), dtype=np.float32)
        self.velocities = np.zeros((self.num_agents, 2), dtype=np.float32)
        self.teams = np.zeros(self.num_agents, dtype=np.int32)
        self.facing_angle = np.zeros(self.num_agents, dtype=np.float32)
        self.facing = np.zeros((self.num_agents, 2), dtype=np.float32)
        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            self.clock = pygame.time.Clock()
        elif self.render_mode == "rgb_array":
            pygame.init()
            self.render_surface = pygame.Surface((WIDTH, HEIGHT))

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.num_moves = 0
        self.positions[:, :] = np.array([
            [WIDTH // 4 + 200, HEIGHT // 2],
            [WIDTH // 4, HEIGHT // 2 - 80],
            [WIDTH // 4, HEIGHT // 2 + 80],
            [WIDTH * 3 // 4 - 100, HEIGHT // 2],
            [WIDTH * 3 // 4, HEIGHT // 2 - 80],
            [WIDTH * 3 // 4, HEIGHT // 2 + 80],
        ], dtype=np.float32)
        self.velocities[:, :] = 0.0
        self.teams[:] = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
        self.facing_angle[:] = 0.0
        self.facing[:, 0] = 1.0
        self.facing[:, 1] = 0.0
        
        self.ball = {
            "x": WIDTH//2,
            "y": HEIGHT//2,
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
            "prev_goal_dist_team0": math.hypot(WIDTH//2 - WIDTH, HEIGHT//2 - HEIGHT / 2.0),
            "prev_goal_dist_team1": math.hypot(WIDTH//2 - 0.0, HEIGHT//2 - HEIGHT / 2.0),
        }

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
        
        # 1. Apply Movement and Physics
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

        self.positions[:, 0] = np.clip(self.positions[:, 0], PLAYER_RADIUS, WIDTH - PLAYER_RADIUS)
        self.positions[:, 1] = np.clip(self.positions[:, 1], PLAYER_RADIUS, HEIGHT - PLAYER_RADIUS)

        rewards += self.reward_calc.step_rewards(self.positions, self.teams, self.ball)
        speeds = np.linalg.norm(self.velocities, axis=1)
        rewards += self.reward_calc.movement_reward(speeds)
        rewards += self.reward_calc.possession_reward(self.teams, self.ball)

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

            self.ball["owner_idx"] = -1
            self.ball["cooldown"] = 0
            self.ball["last_touch_idx"] = int(shooter)
            self.ball["last_touch_step"] = self.num_moves
            self.ball["last_kick_idx"] = int(shooter)
            self.ball["last_kick_team"] = int(team)
            self.ball["last_kick_pressure"] = float(pressure)
            enemy_goal_x = WIDTH if team == 0 else 0.0
            enemy_goal_y = HEIGHT / 2.0
            self.ball["last_kick_goal_dist"] = math.hypot(self.ball["x"] - enemy_goal_x, self.ball["y"] - enemy_goal_y)
            self.ball["vx"] = dir_x * MAX_SHOT_SPEED * shot_power
            self.ball["vy"] = dir_y * MAX_SHOT_SPEED * shot_power

        # 1b. Prevent player overlap (simple separation)
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
                    self.positions[i, 0] = max(PLAYER_RADIUS, min(WIDTH - PLAYER_RADIUS, self.positions[i, 0]))
                    self.positions[i, 1] = max(PLAYER_RADIUS, min(HEIGHT - PLAYER_RADIUS, self.positions[i, 1]))
                    self.positions[j, 0] = max(PLAYER_RADIUS, min(WIDTH - PLAYER_RADIUS, self.positions[j, 0]))
                    self.positions[j, 1] = max(PLAYER_RADIUS, min(HEIGHT - PLAYER_RADIUS, self.positions[j, 1]))

        # 2. Ball Physics and Possession
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
                    elif self.ball.get("last_kick_team") is not None and self.ball["last_kick_team"] != int(self.teams[idx]):
                        self.ball["last_passer_idx"] = -1

                    rewards += self.reward_calc.pass_rewards(self.ball, self.teams, idx)
                    rewards += self.reward_calc.tackle_rewards(self.teams, prev_owner_team, idx)

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

        # 3. Check Goals / Out of Bounds
        truncations = {a: False for a in self.agents}
        goal_rewards, terminated = self.reward_calc.goal_rewards(self.teams, self.ball)
        rewards += goal_rewards
        out_rewards, out_of_bounds = self.reward_calc.out_of_bounds_rewards(self.ball, self.num_agents)
        rewards += out_rewards
        if out_of_bounds and not terminated:
            terminated = True

        # Time Limit
        if self.num_moves > 1000:
            truncations = {a: True for a in self.agents}

        rewards += self.reward_calc.ball_progress_rewards(self.ball, self.teams)

        self.ball["step_count"] = self.num_moves
        self.ball["prev_goal_dist_team0"] = math.hypot(self.ball["x"] - WIDTH, self.ball["y"] - HEIGHT / 2.0)
        self.ball["prev_goal_dist_team1"] = math.hypot(self.ball["x"] - 0.0, self.ball["y"] - HEIGHT / 2.0)

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
        pygame.draw.rect(surface, (255, 255, 255), (0, 0, WIDTH, HEIGHT), 2)
        pygame.draw.line(surface, (255, 255, 255), (WIDTH//2, 0), (WIDTH//2, HEIGHT), 2)
        pygame.draw.circle(surface, (255, 255, 255), (WIDTH//2, HEIGHT//2), 60, 2)
        pygame.draw.rect(surface, (200, 200, 50), (-10, HEIGHT//2 - 50, 20, 100))
        pygame.draw.rect(surface, (200, 200, 50), (WIDTH - 10, HEIGHT//2 - 50, 20, 100))

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
                self.render_surface = pygame.Surface((WIDTH, HEIGHT))
            self._draw(self.render_surface)
            frame = pygame.surfarray.array3d(self.render_surface)
            return np.transpose(frame, (1, 0, 2))

        return None