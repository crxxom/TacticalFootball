import numpy as np
import math

# Assuming these are defined in your config.py
from config import MAX_TEAMMATES, MAX_ENEMIES, MAX_SHOT_SPEED, POSSESSION_COOLDOWN_FRAMES, FPS

class DynamicObservationBuilder:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def build(self, index, positions, velocities, teams, ball, facing):
        pos = positions[index]
        vel = velocities[index]
        team = teams[index]
        obs = []
        pitch_diag = math.hypot(self.width, self.height)

        # Self Velocity
        obs.extend([vel[0], vel[1]])

        # My Goal Distance
        my_goal_x = 0.0 if team == 0 else self.width
        my_goal_y = self.height / 2.0
        obs.extend([my_goal_x - pos[0], my_goal_y - pos[1]])

        # Enemy Goal Distance
        enemy_goal_x = self.width if team == 0 else 0.0
        enemy_goal_y = self.height / 2.0
        obs.extend([enemy_goal_x - pos[0], enemy_goal_y - pos[1]])

        # Ball Position and Velocity (Relative)
        obs.extend([
            ball["x"] - pos[0],
            ball["y"] - pos[1],
            ball["vx"],
            ball["vy"],
        ])

        # Pitch Boundaries
        obs.extend([
            pos[1],
            self.height - pos[1],
            pos[0],
            self.width - pos[0],
        ])

        # Calculate Distances to all other players
        deltas = positions - pos
        dists = np.hypot(deltas[:, 0], deltas[:, 1])
        idxs = np.arange(len(teams))
        
        teammate_mask = (teams == team) & (idxs != index)
        enemy_mask = teams != team

        teammate_indices = idxs[teammate_mask]
        enemy_indices = idxs[enemy_mask]

        teammate_order = teammate_indices[np.argsort(dists[teammate_indices])] if teammate_indices.size else np.array([], dtype=int)
        enemy_order = enemy_indices[np.argsort(dists[enemy_indices])] if enemy_indices.size else np.array([], dtype=int)

        # Pad Teammates up to MAX_TEAMMATES
        for k in range(MAX_TEAMMATES):
            if k < teammate_order.size:
                j = teammate_order[k]
                obs.extend([deltas[j, 0], deltas[j, 1], velocities[j, 0], velocities[j, 1]])
            else:
                obs.extend([0.0, 0.0, 0.0, 0.0]) # Padding for non-existent players

        # Pad Enemies up to MAX_ENEMIES
        for k in range(MAX_ENEMIES):
            if k < enemy_order.size:
                j = enemy_order[k]
                obs.extend([deltas[j, 0], deltas[j, 1], velocities[j, 0], velocities[j, 1]])
            else:
                obs.extend([0.0, 0.0, 0.0, 0.0]) # Padding for non-existent players

        # Ball Ownership Status
        owner_idx = ball.get("owner_idx", -1)
        ownership = 0.0
        if owner_idx != -1:
            ownership = 1.0 if teams[owner_idx] == team else -1.0
        obs.append(ownership)

        # Ball Heading and Speed
        ball_speed = math.hypot(ball["vx"], ball["vy"])
        ball_heading = math.atan2(ball["vy"], ball["vx"]) if ball_speed > 0 else 0.0
        obs.extend([math.cos(ball_heading), math.sin(ball_heading)])
        obs.append(ball_speed / MAX_SHOT_SPEED)

        # Facing Direction
        obs.extend([facing[index, 0], facing[index, 1]])

        # Nearest player proximities
        nearest_teammate = dists[teammate_order[0]] if teammate_order.size else pitch_diag
        nearest_opponent = dists[enemy_order[0]] if enemy_order.size else pitch_diag
        obs.append(nearest_teammate / pitch_diag)
        obs.append(nearest_opponent / pitch_diag)

        # Cooldowns and Touch Timers
        cooldown_norm = ball.get("cooldown", 0) / max(1, POSSESSION_COOLDOWN_FRAMES)
        obs.append(cooldown_norm)
        steps_since_touch = ball.get("step_count", 0) - ball.get("last_touch_step", 0)
        obs.append(min(1.0, steps_since_touch / (FPS * 5)))

        return np.array(obs, dtype=np.float32)