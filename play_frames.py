import os
import sys
import time
import pygame

def main():
    if len(sys.argv) < 2:
        print("Usage: python play_frames.py <frames_dir> [fps]")
        sys.exit(1)

    frames_dir = sys.argv[1]
    fps = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0

    frame_files = sorted(
        f for f in os.listdir(frames_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )

    if not frame_files:
        print(f"No image frames found in {frames_dir}")
        sys.exit(1)

    pygame.init()
    first_frame = pygame.image.load(os.path.join(frames_dir, frame_files[0]))
    width, height = first_frame.get_size()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Frame Player")
    clock = pygame.time.Clock()

    running = True
    index = 0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        frame_path = os.path.join(frames_dir, frame_files[index])
        frame_surface = pygame.image.load(frame_path)
        screen.blit(frame_surface, (0, 0))
        pygame.display.flip()

        index = (index + 1) % len(frame_files)
        clock.tick(60.0)

    pygame.quit()

if __name__ == "__main__":
    main()
