import numpy as np
import math
from config import *

class RoleBasedRewardCalculator:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def calculate_role_modifiers(self, agent_role: AgentRole, base_rewards: dict, event_flags: dict) -> float:
        """
        Takes generic environment rewards and modifies them based on the agent's specific class.
        """
        modifier = 0.0
        
        if agent_role == AgentRole.TARGET_MAN:
            # Target Man cares primarily about goals and taking shots
            if event_flags.get("scored_goal"):
                modifier += GOAL_REWARD * 1.5 
            if event_flags.get("took_shot_in_box"):
                modifier += 5.0
                
        elif agent_role == AgentRole.PLAYMAKER:
            # Playmaker cares about progressive passes and assists
            if event_flags.get("completed_pass"):
                modifier += PASS_RECEIVE_REWARD * 2.0
            if event_flags.get("assisted_goal"):
                modifier += GOAL_REWARD * 1.0
                
        elif agent_role == AgentRole.ANCHOR_DEFENDER:
            # Anchor cares about tackles and interceptions
            if event_flags.get("successful_tackle"):
                modifier += 15.0
                
        return modifier
    
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

        new_dist_team0 = math.hypot(ball["x"] - self.width, ball["y"] - self.height / 2.0)
        new_dist_team1 = math.hypot(ball["x"] - 0.0, ball["y"] - self.height / 2.0)
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
            or ball["x"] > self.width
            or ball["y"] < 0
            or ball["y"] > self.height
        )
        last_touch_idx = ball.get("last_touch_idx", -1)
        if out_of_bounds and last_touch_idx != -1:
            rewards[last_touch_idx] += OUT_OF_BOUNDS_PENALTY

        return rewards, out_of_bounds



