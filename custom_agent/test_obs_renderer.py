"""
Test Observation Renderer
Compares images rendered from observations vs actual environment frames.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import cv2
import matplotlib.pyplot as plt
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




def compute_similarity(img1: np.ndarray, img2: np.ndarray) -> dict:
    """
    Compute various similarity metrics between two images.
    
    Returns:
        dict with MSE, PSNR, and structural similarity info
    """
    # Ensure same size
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    # Convert to grayscale for simpler comparison
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY) if len(img1.shape) == 3 else img1
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY) if len(img2.shape) == 3 else img2
    
    # Mean Squared Error
    mse = np.mean((gray1.astype(float) - gray2.astype(float)) ** 2)
    
    # Peak Signal-to-Noise Ratio
    if mse == 0:
        psnr = float('inf')
    else:
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    return {
        'mse': mse,
        'psnr': psnr,
        'shape_match': img1.shape == img2.shape
    }


def test_single_frame():
    """Test rendering a single frame from observation."""
    print("=" * 60)
    print("TEST 1: Single Frame Comparison")
    print("=" * 60)
    
    # Create environment
    env = WarehouseBrawl(
        mode=RenderMode.RGB_ARRAY,
        resolution=CameraResolution.LOW,
        train_mode=True
    )
    
    # Reset environment
    obs, _ = env.reset(seed=42)
    
    # Get observation for player 0
    player_obs = obs[0]
    
    # Render from observation
    obs_renderer = ObservationRenderer(width=720, height=480)  # Match camera window: width=720, height=480
    obs_img = obs_renderer.render(player_obs)
    
    # Fix logger structure before rendering
    fix_logger_structure(env)
    
    # Get actual frame from environment
    actual_frame = env.render()
    
    # Rotate actual frame to match observation renderer orientation
    actual_frame = np.rot90(actual_frame, k=-1)
    actual_frame = np.fliplr(actual_frame)
    
    print(f"Observation rendering shape: {obs_img.shape}")
    print(f"Actual frame shape: {actual_frame.shape}")
    
    # Compute similarity
    similarity = compute_similarity(obs_img, actual_frame)
    print(f"\nSimilarity Metrics:")
    print(f"  MSE: {similarity['mse']:.2f}")
    print(f"  PSNR: {similarity['psnr']:.2f} dB")
    print(f"  Shape match: {similarity['shape_match']}")
    
    # Create side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].imshow(obs_img)
    axes[0].set_title('From Observation (Simplified)', fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(actual_frame)
    axes[1].set_title('From env.render() (Actual)', fontsize=14)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('test_obs_renderer_single_frame.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison to: test_obs_renderer_single_frame.png")
    plt.close()
    
    env.close()
    print("\n✓ Test 1 completed\n")


def test_multiple_frames():
    """Test rendering multiple frames during gameplay."""
    print("=" * 60)
    print("TEST 2: Multiple Frames During Gameplay")
    print("=" * 60)
    
    # Create environment
    env = WarehouseBrawl(
        mode=RenderMode.RGB_ARRAY,
        resolution=CameraResolution.LOW,
        train_mode=True
    )
    
    # Reset environment
    obs, _ = env.reset(seed=42)
    
    # Create renderer
    obs_renderer = ObservationRenderer(width=720, height=480)  # Match camera window: width=720, height=480
    
    # Run for a few steps
    num_steps = 5
    comparison_frames = []
    
    print(f"\nRunning {num_steps} steps with random actions...")
    
    for step in range(num_steps):
        # Random actions for both agents
        actions = {
            0: env.action_space.sample(),
            1: env.action_space.sample()
        }
        
        # Step environment
        obs, rewards, terminated, truncated, info = env.step(actions)
        
        # Get observation for player 0
        player_obs = obs[0]
        
        # Render from observation
        obs_img = obs_renderer.render(player_obs)
        
        # Fix logger structure before rendering
        fix_logger_structure(env)
        
        # Get actual frame
        actual_frame = env.render()
        
        # Rotate actual frame to match observation renderer orientation
        actual_frame = np.rot90(actual_frame, k=-1)
        actual_frame = np.fliplr(actual_frame)
        
        # Compute similarity
        similarity = compute_similarity(obs_img, actual_frame)
        
        print(f"  Step {step+1}: MSE={similarity['mse']:.2f}, PSNR={similarity['psnr']:.2f} dB")
        
        # Store for visualization
        comparison_frames.append((obs_img, actual_frame, step+1))
        
        if terminated or truncated:
            break
    
    # Create grid comparison
    n_frames = len(comparison_frames)
    fig, axes = plt.subplots(n_frames, 2, figsize=(14, 5 * n_frames))
    
    if n_frames == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (obs_img, actual_frame, step_num) in enumerate(comparison_frames):
        axes[idx, 0].imshow(obs_img)
        axes[idx, 0].set_title(f'Step {step_num}: From Observation', fontsize=12)
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(actual_frame)
        axes[idx, 1].set_title(f'Step {step_num}: Actual Frame', fontsize=12)
        axes[idx, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('test_obs_renderer_multiple_frames.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison to: test_obs_renderer_multiple_frames.png")
    plt.close()
    
    env.close()
    print("\n✓ Test 2 completed\n")


def test_simplified_vs_full():
    """Test simplified rendering (hitboxes only) vs observation rendering."""
    print("=" * 60)
    print("TEST 3: Simplified Rendering vs Observation Rendering")
    print("=" * 60)
    
    # Create environment
    env = WarehouseBrawl(
        mode=RenderMode.RGB_ARRAY,
        resolution=CameraResolution.LOW,
        train_mode=True
    )
    
    # Reset environment
    obs, _ = env.reset(seed=42)
    
    # Run a few steps to get interesting state
    for _ in range(10):
        actions = {
            0: env.action_space.sample(),
            1: env.action_space.sample()
        }
        obs, _, terminated, truncated, _ = env.step(actions)
        if terminated or truncated:
            obs, _ = env.reset()
    
    # Get observation for player 0
    player_obs = obs[0]
    
    # Render from observation
    obs_renderer = ObservationRenderer(width=720, height=480)  # Match camera window: width=720, height=480
    obs_img = obs_renderer.render(player_obs)
    
    # Fix logger structure before rendering
    fix_logger_structure(env)
    
    # Get simplified frame (no UI, simplified obstacles)
    simplified_frame = env.camera.get_frame(
        env, 
        mode=RenderMode.RGB_ARRAY,
        draw_ui=False,
        hitboxes_only=True,
        simplify_obstacles=True
    )
    
    # Get full frame (no UI text overlay for cleaner comparison)
    full_frame = env.camera.get_frame(
        env,
        mode=RenderMode.RGB_ARRAY,
        draw_ui=False
    )
    
    # Rotate frames to match observation renderer orientation
    simplified_frame = np.rot90(simplified_frame, k=-1)
    full_frame = np.rot90(full_frame, k=-1)
    simplified_frame = np.fliplr(simplified_frame)
    full_frame = np.fliplr(full_frame)
    
    # Create three-way comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(obs_img)
    axes[0].set_title('From Observation\n(Completely Synthetic)', fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(simplified_frame)
    axes[1].set_title('Simplified Frame\n(No UI, Simple Background)', fontsize=12)
    axes[1].axis('off')
    
    axes[2].imshow(full_frame)
    axes[2].set_title('Full Frame\n(No UI Text Overlay)', fontsize=12)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('test_obs_renderer_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved three-way comparison to: test_obs_renderer_comparison.png")
    plt.close()
    
    # Compute similarities
    sim_obs_simple = compute_similarity(obs_img, simplified_frame)
    sim_obs_full = compute_similarity(obs_img, full_frame)
    
    print(f"\nObservation vs Simplified: MSE={sim_obs_simple['mse']:.2f}, PSNR={sim_obs_simple['psnr']:.2f} dB")
    print(f"Observation vs Full: MSE={sim_obs_full['mse']:.2f}, PSNR={sim_obs_full['psnr']:.2f} dB")
    
    env.close()
    print("\n✓ Test 3 completed\n")


def test_video_comparison():
    """Create a video comparing observation rendering vs actual frames."""
    print("=" * 60)
    print("TEST 4: Video Comparison (30 frames)")
    print("=" * 60)
    
    # Create environment
    env = WarehouseBrawl(
        mode=RenderMode.RGB_ARRAY,
        resolution=CameraResolution.LOW,
        train_mode=True
    )
    
    # Reset environment
    obs, _ = env.reset(seed=42)
    
    # Create renderer
    obs_renderer = ObservationRenderer(width=720, height=480, draw_info=False)  # Match camera window: width=720, height=480
    
    # Video writer setup - try different codecs for better compatibility
    codecs_to_try = [
        ('avc1', 'test_obs_renderer_video.mp4'),  # H.264 MP4
        ('mp4v', 'test_obs_renderer_video.mp4'),  # MP4V
        ('XVID', 'test_obs_renderer_video.avi'),  # XVID AVI
        ('MJPG', 'test_obs_renderer_video.avi')   # Motion JPEG AVI
    ]
    out = None
    video_filename = None
    
    for codec, filename in codecs_to_try:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(filename, fourcc, 30.0, (1440, 480))
        if out.isOpened():
            print(f"Using video codec: {codec} -> {filename}")
            video_filename = filename
            break
        else:
            out.release()
            out = None
    
    if out is None:
        print("Warning: Could not initialize video writer with any codec")
        return
    
    # Run for 1000 frames
    num_frames = 1000
    avg_mse = 0
    avg_psnr = 0
    
    print(f"\nGenerating {num_frames} frames...")
    
    for frame_num in range(num_frames):
        # Random actions
        actions = {
            0: env.action_space.sample(),
            1: env.action_space.sample()
        }
        
        # Step environment
        obs, _, terminated, truncated, _ = env.step(actions)
        
        if terminated or truncated:
            obs, _ = env.reset()
        
        # Get observation for player 0
        player_obs = obs[0]
        
        # Render from observation
        obs_img = obs_renderer.render(player_obs)
        
        # Fix logger structure before rendering
        fix_logger_structure(env)
        
        # Get actual frame
        actual_frame = env.camera.get_frame(env, mode=RenderMode.RGB_ARRAY, draw_ui=False, hitboxes_only=True, simplify_obstacles=True)
        
        # Rotate actual frame to match observation renderer orientation
        actual_frame = np.rot90(actual_frame, k=-1)
        actual_frame = np.fliplr(actual_frame)
        
        # Resize to match if needed
        if obs_img.shape != actual_frame.shape:
            actual_frame = cv2.resize(actual_frame, (obs_img.shape[1], obs_img.shape[0]))
        
        # Compute similarity
        similarity = compute_similarity(obs_img, actual_frame)
        avg_mse += similarity['mse']
        avg_psnr += similarity['psnr']
        
        # Create side-by-side frame
        combined = np.hstack([obs_img, actual_frame])
        
        # Add labels
        cv2.putText(combined, 'From Observation', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined, 'Actual Frame', (730, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined, f'Frame {frame_num+1}/{num_frames}', (10, 460), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Write frame (convert RGB to BGR for OpenCV)
        out.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        
        if (frame_num + 1) % 100 == 0:
            print(f"  Processed {frame_num + 1}/{num_frames} frames")
    
    # Cleanup
    if out is not None:
        out.release()
        avg_mse /= num_frames
        avg_psnr /= num_frames
    else:
        print("Video generation failed - no video file created")
        return
    
    print(f"\nAverage Similarity Metrics over {num_frames} frames:")
    print(f"  MSE: {avg_mse:.2f}")
    print(f"  PSNR: {avg_psnr:.2f} dB")
    print(f"\nSaved video comparison to: {video_filename}")
    
    env.close()
    print("\n✓ Test 4 completed\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("OBSERVATION RENDERER TEST SUITE")
    print("=" * 60 + "\n")
    
    try:
        # Test 1: Single frame
        test_single_frame()
        
        # Test 2: Multiple frames
        test_multiple_frames()
        
        # Test 3: Simplified vs full rendering
        test_simplified_vs_full()
        
        # Test 4: Video comparison
        test_video_comparison()
        
        print("=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY! ✓")
        print("=" * 60)
        print("\nGenerated files:")
        print("  - test_obs_renderer_single_frame.png")
        print("  - test_obs_renderer_multiple_frames.png")
        print("  - test_obs_renderer_comparison.png")
        print("  - test_obs_renderer_video.mp4")
        print("\nNote: The observation-based rendering is a simplified")
        print("approximation and will not match the actual frames exactly.")
        print("The purpose is to provide fast, headless visualization.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

