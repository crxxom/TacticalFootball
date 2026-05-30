import argparse
import json
import os
import pygame

from config import CURRICULUM, WIDTH, HEIGHT, PLAYER_RADIUS, BALL_RADIUS

PITCH_COLOR = (50, 150, 50)
LINE_COLOR = (255, 255, 255)
BLUE_COLOR = (50, 100, 200)
RED_COLOR = (200, 50, 50)
BALL_COLOR = (0, 0, 0)
BALL_OUTLINE = (255, 255, 255)


def build_agent_ids(task, team_size=None):
    size = team_size or CURRICULUM[task].team_size
    blue_ids = [f"blue_{i}" for i in range(size)]
    red_ids = [f"red_{i}" for i in range(size)]
    return blue_ids + red_ids, size


def default_positions(width, height, team_size):
    positions = {}
    for i in range(team_size):
        y_offset = (i - team_size / 2.0) * 80
        positions[f"blue_{i}"] = [width * 0.25, height * 0.5 + y_offset]
        positions[f"red_{i}"] = [width * 0.75, height * 0.5 + y_offset]
    return positions


def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def save_layout(path, width, height, ball_pos, positions):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        "pitch": {"width": int(width), "height": int(height)},
        "ball": {"x": float(ball_pos[0]), "y": float(ball_pos[1])},
        "players": positions,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def draw_pitch(surface, width, height):
    surface.fill(PITCH_COLOR)
    pygame.draw.rect(surface, LINE_COLOR, (0, 0, width, height), 2)
    pygame.draw.line(surface, LINE_COLOR, (width // 2, 0), (width // 2, height), 2)
    pygame.draw.circle(surface, LINE_COLOR, (width // 2, height // 2), 60, 2)
    pygame.draw.rect(surface, (200, 200, 50), (-10, height // 2 - 50, 20, 100))
    pygame.draw.rect(surface, (200, 200, 50), (width - 10, height // 2 - 50, 20, 100))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="1v1", choices=list(CURRICULUM.keys()))
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--team-size", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    agent_ids, team_size = build_agent_ids(args.task, args.team_size)
    out_path = args.out
    if out_path is None:
        out_path = os.path.join(os.path.dirname(__file__), "layouts", f"{args.task}.json")

    width = args.width
    height = args.height

    pygame.init()
    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()

    positions = default_positions(width, height, team_size)
    ball_pos = [width * 0.5, height * 0.5]

    selected = None
    dragging = False

    print("Drag players or ball with left mouse. Press S to save, ESC to quit.")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_s:
                    save_layout(out_path, width, height, ball_pos, positions)
                    print(f"Saved layout to {out_path}")
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                selected = None
                if (mx - ball_pos[0]) ** 2 + (my - ball_pos[1]) ** 2 <= BALL_RADIUS ** 2:
                    selected = "ball"
                else:
                    for agent_id in agent_ids:
                        px, py = positions[agent_id]
                        if (mx - px) ** 2 + (my - py) ** 2 <= PLAYER_RADIUS ** 2:
                            selected = agent_id
                            break
                dragging = selected is not None
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                dragging = False
                selected = None

        if dragging and selected:
            mx, my = pygame.mouse.get_pos()
            if selected == "ball":
                ball_pos[0] = clamp(mx, BALL_RADIUS, width - BALL_RADIUS)
                ball_pos[1] = clamp(my, BALL_RADIUS, height - BALL_RADIUS)
            else:
                positions[selected][0] = clamp(mx, PLAYER_RADIUS, width - PLAYER_RADIUS)
                positions[selected][1] = clamp(my, PLAYER_RADIUS, height - PLAYER_RADIUS)

        draw_pitch(screen, width, height)
        for agent_id in agent_ids:
            color = BLUE_COLOR if agent_id.startswith("blue") else RED_COLOR
            px, py = positions[agent_id]
            pygame.draw.circle(screen, color, (int(px), int(py)), PLAYER_RADIUS)

        pygame.draw.circle(screen, BALL_COLOR, (int(ball_pos[0]), int(ball_pos[1])), BALL_RADIUS)
        pygame.draw.circle(screen, BALL_OUTLINE, (int(ball_pos[0]), int(ball_pos[1])), BALL_RADIUS, 1)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
