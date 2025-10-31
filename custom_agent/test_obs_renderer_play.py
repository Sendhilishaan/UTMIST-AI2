"""
Show Observation Renderer Output in Pygame Window
Displays the observation-based rendering in real-time using pygame.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pygame
import numpy as np
import cv2
from environment.environment import WarehouseBrawl, RenderMode, CameraResolution
from custom_agent.obs_renderer import ObservationRenderer


def fix_logger_structure(env):
    """Fix the logger structure to prevent KeyError in handle_ui."""
    # Initialize logger as dictionaries if they're strings
    for i in range(len(env.logger)):
        if isinstance(env.logger[i], str):
            env.logger[i] = {}
        # Ensure required keys exist
        if 'transition' not in env.logger[i]:
            env.logger[i]['transition'] = ''
        if 'move_type' not in env.logger[i]:
            env.logger[i]['move_type'] = ''
        if 'total_reward' not in env.logger[i]:
            env.logger[i]['total_reward'] = ''
        if 'reward' not in env.logger[i]:
            env.logger[i]['reward'] = ''


def numpy_to_pygame_surface(numpy_array):
    """Convert numpy array to pygame surface."""
    # Fix orientation - transpose to match pygame's expected format
    if len(numpy_array.shape) == 3:
        # Transpose from (height, width, channels) to (width, height, channels)
        pygame_array = np.transpose(numpy_array, (1, 0, 2))
        return pygame.surfarray.make_surface(pygame_array)
    else:
        # Grayscale
        pygame_array = np.transpose(numpy_array, (1, 0))
        return pygame.surfarray.make_surface(pygame_array)


def main():
    """Main function to display observation renderer in pygame window."""
    print("Initializing pygame display...")
    
    # Initialize pygame
    pygame.init()
    
    # Set up display - double width for side-by-side comparison
    frame_width, frame_height = 720, 480
    screen_width = frame_width * 2  # Two frames side by side
    screen_height = frame_height
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Observation Renderer - UTMIST AI2 (Synthetic vs Actual)")
    
    # Create environment
    print("Creating environment...")
    env = WarehouseBrawl(
        mode=RenderMode.RGB_ARRAY,
        resolution=CameraResolution.LOW,
        train_mode=True
    )
    
    # Reset environment
    obs, _ = env.reset(seed=42)
    
    # Create observation renderer - use full resolution for better quality
    obs_renderer = ObservationRenderer(width=frame_width, height=frame_height)
    
    # Game loop
    clock = pygame.time.Clock()
    running = True
    step_count = 0
    
    print("Starting pygame display loop...")
    print("Display: Synthetic Observation (left) vs Actual Frame (right)")
    print("Controls:")
    print("  R - Reset environment")
    print("  ESC - Exit")
    print("Player 1 (Agent 1) Controls:")
    print("  W - Aim up")
    print("  A - Move left")
    print("  S - Aim down/fastfall")
    print("  D - Move right")
    print("  SPACE - Jump")
    print("  H - Pickup/Throw")
    print("  L - Dash/Dodge")
    print("  J - Light Attack")
    print("  K - Heavy Attack")
    print("  G - Taunt")
    print("Agent 0 will use random actions!")
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    # Reset environment
                    obs, _ = env.reset()
                    step_count = 0
                    print("Environment reset!")
        
        # Get keyboard input for agent 1 (player 1)
        keys = pygame.key.get_pressed()
        agent1_action = env.act_helper.zeros()
        
        # Map keyboard input to action for agent 1
        if keys[pygame.K_w]:
            agent1_action = env.act_helper.press_keys(['w'], agent1_action)
        if keys[pygame.K_a]:
            agent1_action = env.act_helper.press_keys(['a'], agent1_action)
        if keys[pygame.K_s]:
            agent1_action = env.act_helper.press_keys(['s'], agent1_action)
        if keys[pygame.K_d]:
            agent1_action = env.act_helper.press_keys(['d'], agent1_action)
        if keys[pygame.K_SPACE]:
            agent1_action = env.act_helper.press_keys(['space'], agent1_action)
        if keys[pygame.K_h]:
            agent1_action = env.act_helper.press_keys(['h'], agent1_action)
        if keys[pygame.K_j]:
            agent1_action = env.act_helper.press_keys(['j'], agent1_action)
        if keys[pygame.K_k]:
            agent1_action = env.act_helper.press_keys(['k'], agent1_action)
        if keys[pygame.K_l]:
            agent1_action = env.act_helper.press_keys(['l'], agent1_action)
        if keys[pygame.K_g]:
            agent1_action = env.act_helper.press_keys(['g'], agent1_action)
        
        # Step environment: agent 0 uses random actions, agent 1 uses user input
        actions = {
            0: env.action_space.sample(),
            1: agent1_action
        }
        obs, rewards, terminated, truncated, info = env.step(actions)
        step_count += 1
        
        if terminated or truncated:
            obs, _ = env.reset()
            step_count = 0
            print("Environment reset!")
        
        # Get observation for player 0
        player_obs = obs[0]
        
        # Render from observation (synthetic)
        obs_img = obs_renderer.render(player_obs)
        
        # Fix logger structure before rendering actual frame
        fix_logger_structure(env)
        
        # Get actual frame from environment
        actual_frame = env.camera.get_frame(
            env, 
            mode=RenderMode.RGB_ARRAY, 
            draw_ui=True, 
            hitboxes_only=False, 
            simplify_obstacles=False
        )
        
        # Rotate and flip actual frame to match observation renderer orientation
        actual_frame = np.rot90(actual_frame, k=-1)
        actual_frame = np.fliplr(actual_frame)
        
        # Resize actual frame to match observation renderer size if needed
        if obs_img.shape[:2] != actual_frame.shape[:2]:
            actual_frame = cv2.resize(
                actual_frame, 
                (obs_img.shape[1], obs_img.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )
        
        # Ensure both are uint8 and in RGB format
        if obs_img.dtype != np.uint8:
            obs_img = (obs_img * 255).astype(np.uint8) if obs_img.max() <= 1.0 else obs_img.astype(np.uint8)
        if actual_frame.dtype != np.uint8:
            actual_frame = (actual_frame * 255).astype(np.uint8) if actual_frame.max() <= 1.0 else actual_frame.astype(np.uint8)
        
        # Convert both to pygame surfaces
        synthetic_surface = numpy_to_pygame_surface(obs_img)
        actual_surface = numpy_to_pygame_surface(actual_frame)
        
        # Scale to fit if needed
        if synthetic_surface.get_size() != (frame_width, frame_height):
            synthetic_surface = pygame.transform.scale(synthetic_surface, (frame_width, frame_height))
        if actual_surface.get_size() != (frame_width, frame_height):
            actual_surface = pygame.transform.scale(actual_surface, (frame_width, frame_height))
        
        # Draw to screen side by side
        screen.fill((0, 0, 0))  # Black background
        screen.blit(synthetic_surface, (0, 0))  # Left: Synthetic
        screen.blit(actual_surface, (frame_width, 0))  # Right: Actual
        
        # Add text labels
        font = pygame.font.Font(None, 24)
        synthetic_label = font.render("From Observation", True, (255, 255, 255))
        actual_label = font.render("Actual Frame", True, (255, 255, 255))
        screen.blit(synthetic_label, (10, 10))
        screen.blit(actual_label, (frame_width + 10, 10))
        
        # Update display
        pygame.display.flip()
        clock.tick(30)  # 30 FPS
    
    # Cleanup
    pygame.quit()
    env.close()
    print("Display closed. Goodbye!")


if __name__ == "__main__":
    main()
