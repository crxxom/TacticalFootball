from enum import Enum
from dataclasses import dataclass
from typing import List, Dict
import math
import os
import json

# --- Physics Constants ---
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

OUT_OF_BOUNDS_PENALTY = -30.0
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


class AgentRole(Enum):
    GENERIC = "generic"
    TARGET_MAN = "target_man"
    PLAYMAKER = "playmaker"
    ANCHOR_DEFENDER = "anchor"
   

@dataclass
class CurriculumStage:
    name: str
    team_size: int
    blue_roles: List[AgentRole]
    red_roles: List[AgentRole]

# Define the progression of your environment
CURRICULUM = {
    "1v1": CurriculumStage(
        name="1v1", 
        team_size=1,
        blue_roles=[AgentRole.GENERIC],
        red_roles=[AgentRole.GENERIC]
    ),
    "3v3": CurriculumStage(
        name="3v3", 
        team_size=3,
        blue_roles=[AgentRole.TARGET_MAN, AgentRole.PLAYMAKER, AgentRole.ANCHOR_DEFENDER],
        red_roles=[AgentRole.TARGET_MAN, AgentRole.PLAYMAKER, AgentRole.ANCHOR_DEFENDER]
    ),
    "11v11": CurriculumStage(
        name="11v11", 
        team_size=11,
        # Expand roles to 11 players...
        blue_roles=[AgentRole.TARGET_MAN] * 3 + [AgentRole.PLAYMAKER] * 4 + [AgentRole.ANCHOR_DEFENDER] * 4,
        red_roles=[AgentRole.TARGET_MAN] * 3 + [AgentRole.PLAYMAKER] * 4 + [AgentRole.ANCHOR_DEFENDER] * 4,
    )
}

# The absolute maximum size of the environment dictates the Neural Net input layer size
MAX_TEAMMATES = 10
MAX_ENEMIES = 11

LAYOUTS_DIR = os.path.join(os.path.dirname(__file__), "layouts")

def _default_stage_layout():
    return {
        "pitch": {"width": WIDTH, "height": HEIGHT},
        "ball": {"x": WIDTH // 2, "y": HEIGHT // 2},
        "players": {},
    }

def load_stage_layout(stage_name, layout_path=None, layout_dir=None):
    if layout_path:
        path = layout_path
    else:
        layout_root = layout_dir or LAYOUTS_DIR
        path = os.path.join(layout_root, f"{stage_name}.json")

    if not os.path.isfile(path):
        return _default_stage_layout()

    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    layout = _default_stage_layout()
    pitch = data.get("pitch") or {}
    layout["pitch"]["width"] = int(pitch.get("width", layout["pitch"]["width"]))
    layout["pitch"]["height"] = int(pitch.get("height", layout["pitch"]["height"]))

    ball = data.get("ball") or {}
    layout["ball"]["x"] = float(ball.get("x", layout["ball"]["x"]))
    layout["ball"]["y"] = float(ball.get("y", layout["ball"]["y"]))

    players = data.get("players") or {}
    if isinstance(players, dict):
        layout["players"] = players

    return layout