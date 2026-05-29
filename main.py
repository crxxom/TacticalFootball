import pygame
import math
import random
import sys

# --- Constants ---
WIDTH, HEIGHT = 800, 500
FPS = 60

# Colors
GREEN = (50, 150, 50)
WHITE = (255, 255, 255)
RED = (200, 50, 50)
BLUE = (50, 100, 200)
YELLOW = (200, 200, 50)

# Physics
FRICTION = 0.98
MAX_PLAYER_SPEED = 4.0
MAX_SHOT_SPEED = 15.0
POSSESSION_COOLDOWN_FRAMES = 20
BALL_MASS = 1.0
PRESSURE_RADIUS = 60.0
BALL_RENDER_RADIUS = 6
BALL_COLLISION_RADIUS = 4
PASS_BASE_ANGLE_ERROR = math.radians(1.0)
PASS_POWER_ANGLE_ERROR = math.radians(3.0)
PASS_PRESSURE_ANGLE_ERROR = math.radians(6.0)
PASS_FACING_ANGLE_ERROR = math.radians(6.0)
PASS_BASE_POWER_ERROR = 0.02
PASS_POWER_POWER_ERROR = 0.08
PASS_PRESSURE_POWER_ERROR = 0.12
PASS_FACING_POWER_ERROR = 0.1

class Ball:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        self.radius = BALL_RENDER_RADIUS
        self.collision_radius = BALL_COLLISION_RADIUS
        self.owner = None # The player currently dribbling the ball
        self.possession_cooldown = 0

    def update(self):
        if self.possession_cooldown > 0:
            self.possession_cooldown -= 1

        # If a player has the ball, it sticks to them
        if self.owner:
            self.x = self.owner.x + self.owner.facing_x * 12
            self.y = self.owner.y + self.owner.facing_y * 12
            self.vx, self.vy = 0, 0
            return

        # Apply velocity
        self.x += self.vx
        self.y += self.vy

        # Apply friction
        self.vx *= FRICTION
        self.vy *= FRICTION

        # Stop completely if very slow
        if abs(self.vx) < 0.1: self.vx = 0
        if abs(self.vy) < 0.1: self.vy = 0

        # Boundary Collisions (Top and Bottom)
        if self.y - self.collision_radius < 0:
            self.y = self.collision_radius
            self.vy *= -0.8
        elif self.y + self.collision_radius > HEIGHT:
            self.y = HEIGHT - self.collision_radius
            self.vy *= -0.8

        # Boundary Collisions (Left and Right - non-goal areas)
        if self.x - self.collision_radius < 0:
            if not (HEIGHT//2 - 50 < self.y < HEIGHT//2 + 50): # Not in goal
                self.x = self.collision_radius
                self.vx *= -0.8
        elif self.x + self.collision_radius > WIDTH:
            if not (HEIGHT//2 - 50 < self.y < HEIGHT//2 + 50): # Not in goal
                self.x = WIDTH - self.collision_radius
                self.vx *= -0.8

    def draw(self, surface):
        pygame.draw.circle(surface, WHITE, (int(self.x), int(self.y)), self.radius)
        pygame.draw.circle(surface, (0, 0, 0), (int(self.x), int(self.y)), self.radius, 1)

class Player:
    def __init__(self, x, y, color, team_id):
        self.x = x
        self.y = y
        self.color = color
        self.team_id = team_id
        self.radius = 12
        self.speed = MAX_PLAYER_SPEED
        self.facing_x = 1.0
        self.facing_y = 0.0
        self.kick_cooldown = 0

    def move(self, move_x, move_y):
        # Normalize movement to keep max magnitude at 1.0
        length = math.hypot(move_x, move_y)
        if length > 1.0:
            move_x /= length
            move_y /= length

        self.x += move_x * self.speed
        self.y += move_y * self.speed

        # Update facing direction for kicking/dribbling
        if move_x != 0 or move_y != 0:
            self.facing_x = move_x
            self.facing_y = move_y

        # Keep player on pitch
        self.x = max(self.radius, min(WIDTH - self.radius, self.x))
        self.y = max(self.radius, min(HEIGHT - self.radius, self.y))

    def update(self):
        if self.kick_cooldown > 0:
            self.kick_cooldown -= 1

    def draw(self, surface):
        pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), self.radius)
        # Draw a small line to show facing direction
        end_x = self.x + self.facing_x * self.radius
        end_y = self.y + self.facing_y * self.radius
        pygame.draw.line(surface, WHITE, (self.x, self.y), (end_x, end_y), 2)

def draw_pitch(surface):
    surface.fill(GREEN)
    
    # Outer bounds
    pygame.draw.rect(surface, WHITE, (0, 0, WIDTH, HEIGHT), 2)
    
    # Center line and circle
    pygame.draw.line(surface, WHITE, (WIDTH//2, 0), (WIDTH//2, HEIGHT), 2)
    pygame.draw.circle(surface, WHITE, (WIDTH//2, HEIGHT//2), 60, 2)
    
    # Penalty boxes
    pygame.draw.rect(surface, WHITE, (0, HEIGHT//2 - 100, 120, 200), 2)
    pygame.draw.rect(surface, WHITE, (WIDTH - 120, HEIGHT//2 - 100, 120, 200), 2)
    
    # Goals
    pygame.draw.rect(surface, YELLOW, (-10, HEIGHT//2 - 50, 20, 100))
    pygame.draw.rect(surface, YELLOW, (WIDTH - 10, HEIGHT//2 - 50, 20, 100))

    # Half-spaces (Dashed lines to divide pitch into 5 vertical lanes)
    lane_width = HEIGHT / 5
    for i in range(1, 5):
        y_pos = int(i * lane_width)
        for x in range(0, WIDTH, 20):
            pygame.draw.line(surface, (100, 200, 100), (x, y_pos), (x + 10, y_pos), 1)

class AgentAction:
    def __init__(self, move_x=0.0, move_y=0.0, shoot_angle=0.0, shoot_power=0.0):
        self.move_x = move_x
        self.move_y = move_y
        self.shoot_angle = shoot_angle
        self.shoot_power = shoot_power

def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))

def angle_diff(a, b):
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)

def apply_action(player, action, ball, agents):
    player.move(action.move_x, action.move_y)
    player.update()

    if action.shoot_power > 0 and ball.owner == player and player.kick_cooldown == 0:
        shot_power = clamp(action.shoot_power, 0.0, 1.0)
        pressure = 0.0
        for agent in agents:
            if agent == player or agent.team_id == player.team_id:
                continue
            dist = math.hypot(agent.x - player.x, agent.y - player.y)
            if dist < PRESSURE_RADIUS:
                pressure = max(pressure, 1.0 - (dist / PRESSURE_RADIUS))

        facing_angle = math.atan2(player.facing_y, player.facing_x)
        facing_offset = angle_diff(action.shoot_angle, facing_angle) / math.pi
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
        shot_angle = action.shoot_angle + random.gauss(0.0, angle_error)
        shot_power = clamp(shot_power * (1.0 + random.gauss(0.0, power_error)), 0.0, 1.0)
        ball.owner = None
        ball.possession_cooldown = 0
        ball.vx = math.cos(shot_angle) * MAX_SHOT_SPEED * shot_power
        ball.vy = math.sin(shot_angle) * MAX_SHOT_SPEED * shot_power
        player.kick_cooldown = 30

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Phase 1: Football Sandbox")
    clock = pygame.time.Clock()

    # Instantiate Entities
    ball = Ball(WIDTH//2, HEIGHT//2)
    player = Player(WIDTH//4, HEIGHT//2, BLUE, 0)
    blue_support_1 = Player(WIDTH//4, HEIGHT//2 - 80, BLUE, 0)
    blue_support_2 = Player(WIDTH//4, HEIGHT//2 + 80, BLUE, 0)

    red_defender_1 = Player(WIDTH*3//4, HEIGHT//2, RED, 1)
    red_defender_2 = Player(WIDTH*3//4, HEIGHT//2 - 80, RED, 1)
    red_defender_3 = Player(WIDTH*3//4, HEIGHT//2 + 80, RED, 1)

    agents = [
        player,
        blue_support_1,
        blue_support_2,
        red_defender_1,
        red_defender_2,
        red_defender_3,
    ]

    running = True
    while running:
        # 1. Handle Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        
        # 2. Agent Actions
        move_x, move_y = 0, 0
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            move_y -= 1
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            move_y += 1
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            move_x -= 1
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            move_x += 1

        shoot_angle = 0.0
        shoot_power = 0.0
        if keys[pygame.K_SPACE]:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            shoot_angle = math.atan2(mouse_y - player.y, mouse_x - player.x)
            shoot_power = 1.0

        player_action = AgentAction(move_x, move_y, shoot_angle, shoot_power)
        idle_action = AgentAction(0.0, 0.0, 0.0, 0.0)
        apply_action(player, player_action, ball, agents)
        for agent in agents:
            if agent == player:
                continue
            apply_action(agent, idle_action, ball, agents)

        # 3. Ball Interaction Logic
        # Possession and tackle mapping
        if ball.possession_cooldown == 0:
            for agent in agents:
                if agent == ball.owner:
                    continue
                dist_to_ball = math.hypot(agent.x - ball.x, agent.y - ball.y)
                if dist_to_ball < agent.radius + ball.collision_radius + 5 and agent.kick_cooldown == 0:
                    ball.owner = agent
                    ball.possession_cooldown = POSSESSION_COOLDOWN_FRAMES
                    break

        ball.update()

        # 4. Rules: Goal Scoring
        if ball.x < 0 and (HEIGHT//2 - 50 < ball.y < HEIGHT//2 + 50):
            print("GOAL! Right Team Scores!")
            ball = Ball(WIDTH//2, HEIGHT//2) # Reset ball
        elif ball.x > WIDTH and (HEIGHT//2 - 50 < ball.y < HEIGHT//2 + 50):
            print("GOAL! Left Team Scores!")
            ball = Ball(WIDTH//2, HEIGHT//2) # Reset ball

        # 5. Render Graphics
        draw_pitch(screen)
        for agent in agents:
            agent.draw(screen)
        ball.draw(screen)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()