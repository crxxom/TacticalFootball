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

    def build(self, agent_id, agent_states, agents, ball):
        agent = agent_states[agent_id]
        obs = []
        pitch_diag = math.hypot(self.width, self.height)

        # 1. Self Velocity
        obs.extend([agent["vx"], agent["vy"]])

        # 2. My Goal (Relative)
        my_goal_x = 0.0 if agent["team"] == 0 else self.width
        my_goal_y = self.height / 2.0
        obs.extend([my_goal_x - agent["x"], my_goal_y - agent["y"]])

        # 3. Enemy Goal (Relative)
        enemy_goal_x = self.width if agent["team"] == 0 else 0.0
        enemy_goal_y = self.height / 2.0
        obs.extend([enemy_goal_x - agent["x"], enemy_goal_y - agent["y"]])

        # 4. Ball (Relative Position and Velocity)
        obs.extend([
            ball["x"] - agent["x"],
            ball["y"] - agent["y"],
            ball["vx"],
            ball["vy"],
        ])

        # 5. Boundaries (Distance to Top, Bottom, Left, Right)
        obs.extend([
            agent["y"],
            self.height - agent["y"],
            agent["x"],
            self.width - agent["x"],
        ])

        def get_relative_state(other_id):
            other = agent_states[other_id]
            dx = other["x"] - agent["x"]
            dy = other["y"] - agent["y"]
            dist = math.hypot(dx, dy)
            return {"data": [dx, dy, other["vx"], other["vy"]], "dist": dist}

        # 6. Teammates (Exact length: 2, Sorted by distance)
        teammate_ids = [a for a in agents if agent_states[a]["team"] == agent["team"] and a != agent_id]
        teammate_states = [get_relative_state(t) for t in teammate_ids]
        teammate_states.sort(key=lambda item: item["dist"])

        for t_state in teammate_states:
            obs.extend(t_state["data"])

        # 7. Enemies (Exact length: 3, Sorted by distance)
        enemy_ids = [a for a in agents if agent_states[a]["team"] != agent["team"]]
        enemy_states = [get_relative_state(e) for e in enemy_ids]
        enemy_states.sort(key=lambda item: item["dist"])

        for e_state in enemy_states:
            obs.extend(e_state["data"])

        ownership = 0.0
        owner_id = ball.get("owner")
        if owner_id in agent_states:
            owner_team = agent_states[owner_id]["team"]
            ownership = 1.0 if owner_team == agent["team"] else -1.0
        obs.append(ownership)

        ball_speed = math.hypot(ball["vx"], ball["vy"])
        ball_heading = math.atan2(ball["vy"], ball["vx"]) if ball_speed > 0 else 0.0
        obs.extend([math.cos(ball_heading), math.sin(ball_heading)])
        obs.append(ball_speed / MAX_SHOT_SPEED)

        facing_angle = math.atan2(agent["facing_y"], agent["facing_x"])
        obs.extend([math.cos(facing_angle), math.sin(facing_angle)])

        nearest_teammate = teammate_states[0]["dist"] if teammate_states else pitch_diag
        nearest_opponent = enemy_states[0]["dist"] if enemy_states else pitch_diag
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

    def boundary_penalty(self, agent):
        if agent["x"] < PLAYER_RADIUS or agent["x"] > self.width - PLAYER_RADIUS or agent["y"] < PLAYER_RADIUS or agent["y"] > self.height - PLAYER_RADIUS:
            return -0.05
        return 0.0

    def approach_ball_reward(self, agent, ball):
        dist_to_ball = math.hypot(agent["x"] - ball["x"], agent["y"] - ball["y"])
        return -0.001 * dist_to_ball

    def forward_position_penalty(self, agent_id, agent):
        return 0.0

    def step_rewards(self, agent_id, agent, ball):
        return (
            self.boundary_penalty(agent)
            + self.approach_ball_reward(agent, ball)
            + self.forward_position_penalty(agent_id, agent)
        )

    def movement_reward(self, speed):
        return MOVE_REWARD * (speed / MAX_PLAYER_SPEED)

    def possession_reward(self, agent_id, agent_states, ball):
        owner = ball.get("owner")
        if owner in agent_states and agent_states[owner]["team"] == agent_states[agent_id]["team"]:
            return POSSESSION_REWARD
        return 0.0

    def ball_progress_rewards(self, ball, agent_states):
        rewards = {a: 0.0 for a in agent_states.keys()}
        dist_team0 = ball.get("prev_goal_dist_team0")
        dist_team1 = ball.get("prev_goal_dist_team1")
        if dist_team0 is None or dist_team1 is None:
            return rewards

        new_dist_team0 = math.hypot(ball["x"] - WIDTH, ball["y"] - HEIGHT / 2.0)
        new_dist_team1 = math.hypot(ball["x"] - 0.0, ball["y"] - HEIGHT / 2.0)
        progress_team0 = dist_team0 - new_dist_team0
        progress_team1 = dist_team1 - new_dist_team1

        for agent_id, agent in agent_states.items():
            if agent["team"] == 0:
                rewards[agent_id] += BALL_PROGRESS_REWARD_SCALE * progress_team0
            else:
                rewards[agent_id] += BALL_PROGRESS_REWARD_SCALE * progress_team1

        return rewards

    def shot_quality_reward(self, agent, shot_angle):
        return 0.0

    def goal_rewards(self, agents, ball, agent_states):
        rewards = {a: 0.0 for a in agents}
        terminations = {a: False for a in agents}
        scorer = ball.get("last_touch")
        assist = ball.get("last_passer")
        scoring_team = None

        if ball["x"] < 0 and (self.height // 2 - GOAL_HALF_HEIGHT < ball["y"] < self.height // 2 + GOAL_HALF_HEIGHT):
            scoring_team = 1
            for a in agents:
                terminations[a] = True
        elif ball["x"] > self.width and (self.height // 2 - GOAL_HALF_HEIGHT < ball["y"] < self.height // 2 + GOAL_HALF_HEIGHT):
            scoring_team = 0
            for a in agents:
                terminations[a] = True
        if scoring_team is not None:
            for a in agents:
                team = agent_states[a]["team"]
                if team == scoring_team:
                    rewards[a] += GOAL_TEAM_REWARD
                else:
                    rewards[a] -= GOAL_CONCEDE_PENALTY

            if scorer in agent_states:
                if agent_states[scorer]["team"] == scoring_team:
                    rewards[scorer] += GOAL_REWARD
                else:
                    rewards[scorer] -= GOAL_REWARD * 10

            if assist in agent_states and assist != scorer and agent_states[assist]["team"] == scoring_team:
                rewards[assist] += GOAL_ASSIST_REWARD

        return rewards, terminations

    def pass_rewards(self, ball, agent_states, new_owner_id):
        rewards = {a: 0.0 for a in agent_states.keys()}
        if new_owner_id not in agent_states:
            return rewards

        new_team = agent_states[new_owner_id]["team"]
        last_kick = ball.get("last_kick")
        last_kick_team = ball.get("last_kick_team")

        if last_kick is not None and last_kick_team is not None and last_kick in agent_states:
            if new_team == last_kick_team and last_kick != new_owner_id:
                enemy_goal_x = WIDTH if new_team == 0 else 0.0
                enemy_goal_y = HEIGHT / 2.0
                new_dist = math.hypot(ball["x"] - enemy_goal_x, ball["y"] - enemy_goal_y)
                prev_dist = ball.get("last_kick_goal_dist") or new_dist
                progress = prev_dist - new_dist
                rewards[last_kick] += PASS_FIXED_REWARD + (PASS_PROGRESS_SCALE * progress)
                rewards[last_kick] += PASS_PRESSURE_BONUS_SCALE * (ball.get("last_kick_pressure") or 0.0)
                rewards[new_owner_id] += PASS_RECEIVE_REWARD
            elif new_team != last_kick_team:
                rewards[last_kick] += PASS_INTERCEPT_PENALTY

        return rewards

    def tackle_rewards(self, agent_states, prev_owner_team, new_owner_id):
        rewards = {a: 0.0 for a in agent_states.keys()}
        if new_owner_id not in agent_states:
            return rewards

        new_team = agent_states[new_owner_id]["team"]
        if prev_owner_team is not None and new_team != prev_owner_team:
            rewards[new_owner_id] += TACKLE_REWARD

        return rewards

    def out_of_bounds_rewards(self, ball, agents):
        rewards = {a: 0.0 for a in agents}
        out_of_bounds = (
            ball["x"] < 0
            or ball["x"] > WIDTH
            or ball["y"] < 0
            or ball["y"] > HEIGHT
        )
        if out_of_bounds and ball.get("last_touch") in rewards:
            rewards[ball["last_touch"]] += OUT_OF_BOUNDS_PENALTY

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
        
        # Initialize Agent States (x, y, vx, vy, team_id)
        self.agent_states = {
            "blue_forward": {"x": WIDTH//4 + 200, "y": HEIGHT//2, "vx": 0.0, "vy": 0.0, "team": 0},
            "blue_midfielder": {"x": WIDTH//4, "y": HEIGHT//2 - 80, "vx": 0.0, "vy": 0.0, "team": 0},
            "blue_defender": {"x": WIDTH//4, "y": HEIGHT//2 + 80, "vx": 0.0, "vy": 0.0, "team": 0},
            
            "red_forward": {"x": WIDTH*3//4 - 100, "y": HEIGHT//2, "vx": 0.0, "vy": 0.0, "team": 1},
            "red_midfielder": {"x": WIDTH*3//4, "y": HEIGHT//2 - 80, "vx": 0.0, "vy": 0.0, "team": 1},
            "red_defender": {"x": WIDTH*3//4, "y": HEIGHT//2 + 80, "vx": 0.0, "vy": 0.0, "team": 1}
        }

        for agent in self.agent_states.values():
            agent["facing_x"] = 1.0
            agent["facing_y"] = 0.0
            agent["facing_angle"] = 0.0
        
        self.ball = {
            "x": WIDTH//2,
            "y": HEIGHT//2,
            "vx": 0.0,
            "vy": 0.0,
            "owner": None,
            "cooldown": 0,
            "last_touch": None,
            "last_touch_step": 0,
            "last_kick": None,
            "last_kick_team": None,
            "last_kick_goal_dist": None,
            "last_kick_pressure": 0.0,
            "last_passer": None,
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
        return self.obs_builder.build(agent_id, self.agent_states, self.agents, self.ball)

    def step(self, actions):
        self.num_moves += 1
        rewards = {a: 0.0 for a in self.agents}
        
        # 1. Apply Movement and Physics
        prev_owner = self.ball["owner"]
        prev_owner_team = None
        if prev_owner in self.agent_states:
            prev_owner_team = self.agent_states[prev_owner]["team"]

        for agent_id, action in actions.items():
            move_x, move_y, shoot_x, shoot_y, shoot_power, rotate = action
            agent = self.agent_states[agent_id]
            
            # Normalize movement vector
            length = math.hypot(move_x, move_y)
            if length > 1.0: 
                move_x, move_y = move_x/length, move_y/length
            
            # Update Velocity and Position
            agent["vx"] = move_x * MAX_PLAYER_SPEED
            agent["vy"] = move_y * MAX_PLAYER_SPEED
            agent["x"] += agent["vx"] * DT
            agent["y"] += agent["vy"] * DT

            agent["facing_angle"] += rotate * ROTATION_SPEED
            agent["facing_x"] = math.cos(agent["facing_angle"])
            agent["facing_y"] = math.sin(agent["facing_angle"])

            # Clamp to pitch boundaries
            agent["x"] = max(PLAYER_RADIUS, min(WIDTH - PLAYER_RADIUS, agent["x"]))
            agent["y"] = max(PLAYER_RADIUS, min(HEIGHT - PLAYER_RADIUS, agent["y"]))

            rewards[agent_id] += self.reward_calc.step_rewards(agent_id, agent, self.ball)
            speed = math.hypot(agent["vx"], agent["vy"])
            rewards[agent_id] += self.reward_calc.movement_reward(speed)
            rewards[agent_id] += self.reward_calc.possession_reward(agent_id, self.agent_states, self.ball)

            shoot_vector = math.hypot(shoot_x, shoot_y)
            shot_power = clamp(shoot_power, 0.0, 1.0)
            if shot_power > 0.0 and self.ball["owner"] == agent_id and self.ball["cooldown"] == 0:
                pressure = 0.0
                for other_id, other in self.agent_states.items():
                    if other_id == agent_id or other["team"] == agent["team"]:
                        continue
                    dist = math.hypot(other["x"] - agent["x"], other["y"] - agent["y"])
                    if dist < PRESSURE_RADIUS:
                        pressure = max(pressure, 1.0 - (dist / PRESSURE_RADIUS))

                shot_angle = math.atan2(shoot_y, shoot_x) if shoot_vector > 0 else agent["facing_angle"]
                facing_offset = angle_diff(shot_angle, agent["facing_angle"]) / math.pi
                angle_error = (
                    PASS_BASE_ANGLE_ERROR
                    + shot_power * PASS_POWER_ANGLE_ERROR
                    + pressure * PASS_PRESSURE_ANGLE_ERROR
                    + facing_offset * PASS_FACING_ANGLE_ERROR
                )
                power_error = (
                    PASS_BASE_POWER_ERROR
                    + shot_power * PASS_POWER_POWER_ERROR
                    + pressure * PASS_PRESSURE_POWER_ERROR
                    + facing_offset * PASS_FACING_POWER_ERROR
                )
                shot_angle = shot_angle + np.random.normal(0.0, angle_error)
                shot_power = clamp(shot_power * (1.0 + np.random.normal(0.0, power_error)), 0.0, 1.0)
                dir_x = math.cos(shot_angle)
                dir_y = math.sin(shot_angle)
                self.ball["owner"] = None
                self.ball["cooldown"] = 0
                self.ball["last_touch"] = agent_id
                self.ball["last_touch_step"] = self.num_moves
                self.ball["last_kick"] = agent_id
                self.ball["last_kick_team"] = agent["team"]
                self.ball["last_kick_pressure"] = pressure
                enemy_goal_x = WIDTH if agent["team"] == 0 else 0.0
                enemy_goal_y = HEIGHT / 2.0
                self.ball["last_kick_goal_dist"] = math.hypot(self.ball["x"] - enemy_goal_x, self.ball["y"] - enemy_goal_y)
                self.ball["vx"] = dir_x * MAX_SHOT_SPEED * shot_power
                self.ball["vy"] = dir_y * MAX_SHOT_SPEED * shot_power

        # 1b. Prevent player overlap (simple separation)
        agent_items = list(self.agent_states.items())
        for i in range(len(agent_items)):
            id_a, agent_a = agent_items[i]
            for j in range(i + 1, len(agent_items)):
                id_b, agent_b = agent_items[j]
                dx = agent_b["x"] - agent_a["x"]
                dy = agent_b["y"] - agent_a["y"]
                dist = math.hypot(dx, dy)
                min_dist = PLAYER_RADIUS * 2
                if dist == 0:
                    dx, dy = 1.0, 0.0
                    dist = 1.0
                if dist < min_dist:
                    overlap = (min_dist - dist) / 2.0
                    nx = dx / dist
                    ny = dy / dist
                    agent_a["x"] -= nx * overlap
                    agent_a["y"] -= ny * overlap
                    agent_b["x"] += nx * overlap
                    agent_b["y"] += ny * overlap
                    agent_a["x"] = max(PLAYER_RADIUS, min(WIDTH - PLAYER_RADIUS, agent_a["x"]))
                    agent_a["y"] = max(PLAYER_RADIUS, min(HEIGHT - PLAYER_RADIUS, agent_a["y"]))
                    agent_b["x"] = max(PLAYER_RADIUS, min(WIDTH - PLAYER_RADIUS, agent_b["x"]))
                    agent_b["y"] = max(PLAYER_RADIUS, min(HEIGHT - PLAYER_RADIUS, agent_b["y"]))

        # 2. Ball Physics and Possession
        if self.ball["cooldown"] > 0:
            self.ball["cooldown"] -= 1

        if self.ball["owner"] is not None:
            owner = self.agent_states[self.ball["owner"]]
            self.ball["x"] = owner["x"] + self.ball["owner_offset_x"]
            self.ball["y"] = owner["y"] + self.ball["owner_offset_y"]
            self.ball["vx"] = 0.0
            self.ball["vy"] = 0.0
        else:
            self.ball["x"] += self.ball["vx"] * DT
            self.ball["y"] += self.ball["vy"] * DT
            self.ball["vx"] *= FRICTION
            self.ball["vy"] *= FRICTION

        if self.ball["cooldown"] == 0:
            for agent_id, agent in self.agent_states.items():
                if agent_id == self.ball["owner"]:
                    continue
                dist_to_ball = math.hypot(agent["x"] - self.ball["x"], agent["y"] - self.ball["y"])
                if dist_to_ball < PLAYER_RADIUS + BALL_RADIUS + POSSESSION_DISTANCE_BUFFER:
                    self.ball["owner"] = agent_id
                    self.ball["cooldown"] = POSSESSION_COOLDOWN_FRAMES
                    self.ball["last_touch"] = agent_id
                    self.ball["last_touch_step"] = self.num_moves
                    pass_completed = (
                        self.ball.get("last_kick") is not None
                        and self.ball.get("last_kick_team") is not None
                        and self.ball["last_kick"] != agent_id
                        and self.ball["last_kick_team"] == agent["team"]
                    )
                    if pass_completed:
                        self.ball["last_passer"] = self.ball["last_kick"]
                    elif self.ball.get("last_kick_team") is not None and self.ball["last_kick_team"] != agent["team"]:
                        self.ball["last_passer"] = None
                    pass_rewards = self.reward_calc.pass_rewards(
                        self.ball,
                        self.agent_states,
                        agent_id,
                    )
                    for reward_agent, reward in pass_rewards.items():
                        if reward_agent in rewards:
                            rewards[reward_agent] += reward

                    tackle_rewards = self.reward_calc.tackle_rewards(
                        self.agent_states,
                        prev_owner_team,
                        agent_id,
                    )
                    for reward_agent, reward in tackle_rewards.items():
                        if reward_agent in rewards:
                            rewards[reward_agent] += reward

                    self.ball["last_kick"] = None
                    self.ball["last_kick_team"] = None
                    self.ball["last_kick_goal_dist"] = None
                    self.ball["last_kick_pressure"] = 0.0
                    offset_x = self.ball["x"] - agent["x"]
                    offset_y = self.ball["y"] - agent["y"]
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
        goal_rewards, terminations = self.reward_calc.goal_rewards(self.agents, self.ball, self.agent_states)
        for agent_id, reward in goal_rewards.items():
            rewards[agent_id] += reward
        out_rewards, out_of_bounds = self.reward_calc.out_of_bounds_rewards(self.ball, self.agents)
        for agent_id, reward in out_rewards.items():
            rewards[agent_id] += reward
        if out_of_bounds and not any(terminations.values()):
            for a in self.agents:
                terminations[a] = True

        # Time Limit
        if self.num_moves > 1000:
            truncations = {a: True for a in self.agents}

        progress_rewards = self.reward_calc.ball_progress_rewards(self.ball, self.agent_states)
        for agent_id, reward in progress_rewards.items():
            if agent_id in rewards:
                rewards[agent_id] += reward

        self.ball["step_count"] = self.num_moves
        self.ball["prev_goal_dist_team0"] = math.hypot(self.ball["x"] - WIDTH, self.ball["y"] - HEIGHT / 2.0)
        self.ball["prev_goal_dist_team1"] = math.hypot(self.ball["x"] - 0.0, self.ball["y"] - HEIGHT / 2.0)

        if any(terminations.values()) or any(truncations.values()):
            self.agents = [] # End episode

        team_totals = {0: 0.0, 1: 0.0}
        for agent_id, reward in rewards.items():
            team_totals[self.agent_states[agent_id]["team"]] += reward

        for agent_id in rewards.keys():
            team = self.agent_states[agent_id]["team"]
            opponent = 1 - team
            team_delta = team_totals[team] - team_totals[opponent]
            rewards[agent_id] = rewards[agent_id] + (TEAM_REWARD_WEIGHT * team_delta)

        observations = {a: self._get_local_observation(a) for a in self.agents}
        infos = {a: {} for a in self.agents}

        if self.render_mode == "human":
            self.render()

        return observations, rewards, terminations, truncations, infos

    def _draw(self, surface):
        surface.fill((50, 150, 50))
        pygame.draw.rect(surface, (255, 255, 255), (0, 0, WIDTH, HEIGHT), 2)
        pygame.draw.line(surface, (255, 255, 255), (WIDTH//2, 0), (WIDTH//2, HEIGHT), 2)
        pygame.draw.circle(surface, (255, 255, 255), (WIDTH//2, HEIGHT//2), 60, 2)
        pygame.draw.rect(surface, (200, 200, 50), (-10, HEIGHT//2 - 50, 20, 100))
        pygame.draw.rect(surface, (200, 200, 50), (WIDTH - 10, HEIGHT//2 - 50, 20, 100))

        for agent in self.agent_states.values():
            color = (50, 100, 200) if agent["team"] == 0 else (200, 50, 50)
            pygame.draw.circle(surface, color, (int(agent["x"]), int(agent["y"])), PLAYER_RADIUS)

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