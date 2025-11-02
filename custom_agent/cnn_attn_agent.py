"""
Custom Agent with CNN + Attention Architecture
Uses observation renderer to generate synthetic images from 64-dim observations
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import numpy as np
from typing import Optional, List, Type, Dict, Any, Tuple
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor

from environment.environment import CameraResolution
from custom_agent.obs_renderer import ObservationRenderer
from custom_agent.attack_state_tracker import AttackStateTracker
from environment.agent import Agent


class CustomCallback(BaseCallback):
    """Custom callback for detailed training metrics."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.episode_count = 0
        
    def _on_step(self) -> bool:
        # Track rewards manually since episode info is not available
        reward = self.locals['rewards'][0] if len(self.locals['rewards']) > 0 else 0
        done = self.locals['dones'][0] if len(self.locals['dones']) > 0 else False
        
        # Accumulate episode reward and length
        self.current_episode_reward += reward
        self.current_episode_length += 1
        
        # Check if episode ended
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.episode_count += 1
            
            if self.verbose > 0:
                print(f"Episode {self.episode_count} - Reward: {self.current_episode_reward:.2f}, Length: {self.current_episode_length}")
            
            # Reset for next episode
            self.current_episode_reward = 0
            self.current_episode_length = 0
        
        return True
    
    def _on_rollout_end(self) -> None:
        # Log training metrics
        if hasattr(self.model, 'logger'):
            logger = self.model.logger
            if hasattr(logger, 'name_to_value'):
                metrics = logger.name_to_value
                print(f"\n--- Training Metrics ---")
                policy_loss = metrics.get('train/policy_gradient_loss', None)  # Correct metric name
                value_loss = metrics.get('train/value_loss', None)
                entropy_loss = metrics.get('train/entropy_loss', None)
                explained_var = metrics.get('train/explained_variance', None)
                lr = metrics.get('train/learning_rate', None)
                
                print(f"Policy Loss: {policy_loss:.4f}" if policy_loss is not None else "Policy Loss: N/A")
                print(f"Value Loss: {value_loss:.4f}" if value_loss is not None else "Value Loss: N/A")
                print(f"Entropy Loss: {entropy_loss:.4f}" if entropy_loss is not None else "Entropy Loss: N/A")
                print(f"Explained Variance: {explained_var:.4f}" if explained_var is not None else "Explained Variance: N/A")
                print(f"Learning Rate: {lr:.6f}" if lr is not None else "Learning Rate: N/A")
                
                if self.episode_rewards:
                    avg_reward = np.mean(self.episode_rewards[-10:])  # Last 10 episodes
                    print(f"Average Reward (last 10): {avg_reward:.2f}")
                print("------------------------\n")


class ConvBlock(nn.Module):
    """A single convolutional block with two conv layers, batch norm, ReLU, and max pooling."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, padding: int = 1, pool_kernel: int = 2, pool_stride: int = 2):
        super().__init__()
        
        self.block = nn.Sequential(
            # First conv layer
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            
            # Second conv layer
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            
            # Max pooling
            nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride)
        )
    
    def forward(self, x):
        # Use gradient checkpointing to save memory during training
        if self.training and x.requires_grad:
            # Only use gradient checkpointing when gradients are needed
            x = checkpoint.checkpoint(self.block, x, use_reentrant=False)
        else:
            x = self.block(x)
        return x


class CNNFeatureExtractor(nn.Module):
    """CNN feature extractor for processing synthetic images."""
    
    def __init__(self, observation_space, feature_dim=64, img_width=360, img_height=240):
        super().__init__()
        fdim = feature_dim
        self.img_width = img_width
        self.img_height = img_height
        
        # Input: RGB image (height, width, 3) -> (3, height, width)
        # Separate the four convolutional blocks for configurable image size
        self.conv_block1 = ConvBlock(3, fdim)  # img_height x img_width -> img_height/2 x img_width/2
        self.conv_block2 = ConvBlock(fdim, fdim)  # img_height/2 x img_width/2 -> img_height/4 x img_width/4
        self.conv_block3 = ConvBlock(fdim, fdim)  # img_height/4 x img_width/4 -> img_height/8 x img_width/8
        self.conv_block4 = ConvBlock(fdim, fdim)  # img_height/8 x img_width/8 -> img_height/16 x img_width/16
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert from (batch, height, width, 3) to (batch, 3, height, width)
        if x.dim() == 4 and x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)
        
        # Convert to float32 and normalize to [0, 1] - reuse variable
        x = x.float() / 255.0
        
        # Apply CNN blocks sequentially - reuse variable for memory efficiency
        x = self.conv_block1(x)  # img_height x img_width -> img_height/2 x img_width/2
        x = self.conv_block2(x)  # img_height/2 x img_width/2 -> img_height/4 x img_width/4
        x = self.conv_block3(x)  # img_height/4 x img_width/4 -> img_height/8 x img_width/8
        x = self.conv_block4(x)  # img_height/8 x img_width/8 -> img_height/16 x img_width/16
        
        return x


class FeatureWiseAttention(BaseFeaturesExtractor):
    """Custom feature extractor combining CNN and Linear layers."""
    
    def __init__(self, observation_space, features_dim: int = 128, img_width: int = 360, img_height: int = 240):
        # For CnnPolicy, we need to define the output as image-like
        # But BaseFeaturesExtractor expects features_dim to be an integer
        # We'll use the features_dim as the total number of features
        super().__init__(observation_space, features_dim)
        
        # Store image dimensions as hyperparameters
        self.img_width = img_width
        self.img_height = img_height
        
        # Create observation renderer for generating synthetic images (configurable resolution)
        self.obs_renderer = ObservationRenderer(width=img_width, height=img_height, draw_info=False)
        # Attach attack state tracker for optional reuse by agents
        import os
        # Get repo root path - works in both scripts and notebooks (e.g., Kaggle)
        repo_root = None
        try:
            # Try using __file__ (works in scripts, not in notebooks)
            try:
                current_file = os.path.abspath(__file__)
                repo_root = os.path.abspath(os.path.join(os.path.dirname(current_file), '..'))
            except NameError:
                # __file__ not available (e.g., in Kaggle notebooks)
                # Fallback: look for repo root from current working directory
                cwd = os.getcwd()
                # Navigate up until we find the repo root (should have 'environment' and 'custom_agent' folders)
                current = cwd
                while current != os.path.dirname(current):
                    if os.path.exists(os.path.join(current, 'environment')) and \
                       os.path.exists(os.path.join(current, 'custom_agent')):
                        repo_root = current
                        break
                    current = os.path.dirname(current)
                # If not found, try parent directory as fallback
                if repo_root is None:
                    repo_root = os.path.abspath(os.path.join(cwd, '..'))
                    # Verify it has the expected structure
                    if not (os.path.exists(os.path.join(repo_root, 'environment')) and
                            os.path.exists(os.path.join(repo_root, 'custom_agent'))):
                        repo_root = cwd  # Final fallback to current directory
        except Exception:
            repo_root = os.getcwd()  # Ultimate fallback
        
        try:
            if repo_root and os.path.exists(os.path.join(repo_root, 'environment')):
                self.attack_tracker = AttackStateTracker(repo_root)
            else:
                self.attack_tracker = None
        except Exception as e:
            # Fail silently - attack tracker is optional
            self.attack_tracker = None
        
        # CNN feature extractor for images (using configurable synthetic image space)
        synthetic_img_space = spaces.Box(low=0, high=255, shape=(img_height, img_width, 3), dtype=np.uint8)
        self.cnn_extractor = CNNFeatureExtractor(synthetic_img_space, features_dim, img_width, img_height)
        
        # Linear extractor for 64-dim state
        self.linear_extractor = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim),  # Half for linear
            nn.ReLU(),
        )
        
        self.attn = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=features_dim, nhead=8),
            num_layers=1,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Store original observations for linear features
        original_obs = x
        
        # Generate synthetic images and reuse variable
        x = self.get_synthetic_image(x)
        
        # Extract CNN features - reuse variable for memory efficiency
        x = self.cnn_extractor(x)  # (N, C, H, W)
        # Reshape and permute in one operation
        x = x.reshape(x.size(0), x.size(1), -1).permute(2, 0, 1)  # (H*W, N, C)
        
        # Extract linear features from original observations
        linear_features = self.linear_extractor(original_obs)  # (N, C)
        linear_features = linear_features.unsqueeze(0)  # (1, N, C)
        
        # TransformerDecoder attention - reuse variable
        x = self.attn(linear_features, x)  # (1, N, C)
        x = x.squeeze(0)  # (N, C)
        
        # Clear intermediate variables to help with memory management
        del original_obs, linear_features
        
        return x
    
    def get_synthetic_image(self, obs: torch.Tensor) -> torch.Tensor:
        batch_size = obs.size(0)
        device = obs.device
        
        # Pre-allocate tensor on target device (configurable size for memory efficiency)
        synthetic_images = torch.zeros(batch_size, 3, self.img_height, self.img_width, device=device, dtype=torch.float32)
        
        # Process images in parallel for speed (if batch size is small enough)
        if batch_size <= 32:  # Process all at once for small batches
            for i in range(batch_size):
                obs_np = obs[i].cpu().numpy()
                # Update optional attack tracker for 'self' side
                if self.attack_tracker is not None and obs_np.shape[0] >= 64:
                    idxs = {
                        'move_type': 14,
                        'weapon': 15,
                        'state': 8,
                    }
                    last_hitboxes = self.attack_tracker.update_from_obs('self', obs_np, idxs)
                    # Expose last hitboxes for downstream use if needed
                    self.last_hitboxes = last_hitboxes
                synthetic_img = self.obs_renderer.render(obs_np)  # Shape: (img_height, img_width, 3)
                # Convert to tensor and transpose to (3, img_height, img_width)
                synthetic_images[i] = torch.from_numpy(synthetic_img).float().permute(2, 0, 1)
        else:  # Process in chunks for large batches
            chunk_size = 16
            for chunk_start in range(0, batch_size, chunk_size):
                chunk_end = min(chunk_start + chunk_size, batch_size)
                for i in range(chunk_start, chunk_end):
                    obs_np = obs[i].cpu().numpy()
                    if self.attack_tracker is not None and obs_np.shape[0] >= 64:
                        idxs = {
                            'move_type': 14,
                            'weapon': 15,
                            'state': 8,
                        }
                        self.last_hitboxes = self.attack_tracker.update_from_obs('self', obs_np, idxs)
                    synthetic_img = self.obs_renderer.render(obs_np)
                    synthetic_images[i] = torch.from_numpy(synthetic_img).float().permute(2, 0, 1)
        
        return synthetic_images


class CNNAttentionAgent(Agent):
    """Custom agent using CNN + Attention architecture."""
    
    def __init__(self, 
                 sb3_class: Optional[Type[BaseAlgorithm]] = PPO,
                 file_path: str = None, 
                 features_dim: int = 128,  # Reduced from 256
                 net_arch: Optional[List[int]] = [64],  # Reduced from [128]
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 use_sde: bool = False,
                 normalize_images: bool = False,
                 img_width: int = 360,  # Much smaller default
                 img_height: int = 240):  # Much smaller default
        self.sb3_class = sb3_class
        self.features_dim = features_dim
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.use_sde = use_sde
        self.normalize_images = normalize_images
        self.img_width = img_width
        self.img_height = img_height
        super().__init__(file_path)
    
    def _initialize(self) -> None:
        if self.file_path is None:
            # Create custom feature extractor with modified architecture
            policy_kwargs = {
                'features_extractor_class': FeatureWiseAttention,
                'features_extractor_kwargs': {
                    'features_dim': self.features_dim,
                    'img_width': self.img_width,
                    'img_height': self.img_height
                },
                'net_arch': self.net_arch,
                'activation_fn': nn.ReLU,   # Activation function
                'normalize_images': False,  # No image normalization (we handle it in feature extractor)
            }
            
            self.model = self.sb3_class(
                "CnnPolicy",
                self.env,
                policy_kwargs=policy_kwargs,
                verbose=1,  # Enable verbose logging
                n_steps=512,  # Increased for faster training
                batch_size=32,  # Increased batch size for speed
                n_epochs=4,  # Reduced epochs per update for speed
                ent_coef=0.01,
                learning_rate=3e-4,
                use_sde=self.use_sde,
            )
            
            # from torchinfo import summary
            # summary(self.model.policy, input_size=(1, 64))

            # from torchview_custom.torchview import draw_graphs
            # draw_graphs(self.model.policy, inputs=[torch.randn(1, 64)], min_depth=1, max_depth=5, output_names=["Action", "Value", "LogProb"], directory='./model_viz/')

            # Model created successfully
            print(f"Custom CNN + Attention agent created with:")
            print(f"  - CNN features: {self.features_dim}")
            print(f"  - Combined features: CNN + Attention")
            print(f"  - Input: 64-dim state + 3D synthetic image")
            
            del self.env
        else:
            self.model = PPO.load(self.file_path)
            print(f"Custom CNN + Attention agent loaded from {self.file_path}!")
    
    def predict(self, obs):
        """Predict action using the custom policy."""
        # Ensure obs is a numpy array (not torch tensor)
        if isinstance(obs, torch.Tensor):
            obs = obs.numpy()
        
        # Don't add batch dimension - let Stable Baselines3 handle it
        action, _ = self.model.predict(obs)
        return action
    
    def save(self, file_path: str) -> None:
        """Save the model."""
        self.model.save(file_path, include=['num_timesteps'])
    
    def create_parallel_env(self, base_env, n_envs: int = 4, use_subproc: bool = False):
        """Create parallel environments for faster training."""
        # Get the environment creation parameters from the base environment
        env_params = self._extract_env_params(base_env)
        
        def make_env():
            # Create a new environment instance using the same parameters
            return self._create_env_from_params(env_params)
        
        if use_subproc and n_envs > 1:
            # Use subprocess environments for true parallelism (not recommended in Jupyter)
            try:
                env = SubprocVecEnv([make_env for _ in range(n_envs)])
            except Exception as e:
                print(f"SubprocVecEnv failed ({e}), falling back to DummyVecEnv")
                env = DummyVecEnv([make_env for _ in range(n_envs)])
        elif n_envs > 1:
            # Use dummy vectorized environment (single process, works in Jupyter)
            env = DummyVecEnv([make_env for _ in range(n_envs)])
        else:
            # Single environment
            env = base_env
            
        return env
    
    def _extract_env_params(self, env):
        """Extract parameters needed to recreate the environment."""
        # For SelfPlayWarehouseBrawl, we need to extract the key parameters
        if hasattr(env, 'reward_manager') and hasattr(env, 'opponent_cfg'):
            return {
                'reward_manager': env.reward_manager,
                'opponent_cfg': env.opponent_cfg,
                'save_handler': getattr(env, 'save_handler', None),
                'resolution': getattr(env, 'resolution', None),
                'record_every_episodes': getattr(env, 'record_every_episodes', None),
                'record_dir': getattr(env, 'record_dir', None)
            }
        else:
            # Fallback: return the environment itself (not ideal but works for single env)
            return {'env': env}
    
    def _create_env_from_params(self, params):
        """Create a new environment instance from parameters."""
        if 'env' in params:
            # Fallback: return the same environment (not ideal for parallel)
            return params['env']
        else:
            # Create new SelfPlayWarehouseBrawl instance
            from environment.agent import SelfPlayWarehouseBrawl
            return SelfPlayWarehouseBrawl(
                reward_manager=params['reward_manager'],
                opponent_cfg=params['opponent_cfg'],
                save_handler=params['save_handler'],
                resolution=params['resolution'],
                record_every_episodes=params.get('record_every_episodes', None),
                record_dir=params.get('record_dir', None)
            )
    
    def _recreate_model_for_envs(self, env, verbose=1):
        """Recreate the model with the correct number of environments."""
        # Save current model parameters
        current_lr = self.model.learning_rate
        current_n_steps = self.model.n_steps
        current_batch_size = self.model.batch_size
        current_n_epochs = self.model.n_epochs
        current_ent_coef = self.model.ent_coef
        current_use_sde = self.model.use_sde
        
        # Get policy kwargs from current model
        policy_kwargs = {
            'features_extractor_class': FeatureWiseAttention,
            'features_extractor_kwargs': {
                'features_dim': self.features_dim,
                'img_width': self.img_width,
                'img_height': self.img_height
            },
            'net_arch': self.net_arch,
            'activation_fn': nn.ReLU,
            'normalize_images': False,
        }
        
        # Create new model with parallel environment
        self.model = self.sb3_class(
            "CnnPolicy",
            env,  # Use the parallel environment directly
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            n_steps=current_n_steps,
            batch_size=current_batch_size,
            n_epochs=current_n_epochs,
            ent_coef=current_ent_coef,
            learning_rate=current_lr,
            use_sde=current_use_sde,
        )
        
        if verbose > 0:
            print(f"Model recreated for {env.num_envs} environments")
    
    def learn(self, env, total_timesteps, log_interval: int = 1, verbose=1, tensorboard_log: str = None, n_envs: int = 4):
        """Train the model with detailed logging and parallel environments."""
        # Create parallel environments for faster training
        if n_envs > 1:
            if verbose > 0:
                print(f"Creating {n_envs} parallel environments for faster training...")
            parallel_env = self.create_parallel_env(env, n_envs=n_envs, use_subproc=False)  # Use DummyVecEnv by default

            # Attach VecMonitor to record monitor.csv like the single-env Monitor does
            log_dir = None
            try:
                base_env = env.unwrapped if hasattr(env, 'unwrapped') else env
                if hasattr(base_env, 'save_handler') and base_env.save_handler is not None:
                    log_dir = getattr(base_env.save_handler, 'log_dir', None)
            except Exception:
                pass
            if log_dir is None:
                log_dir = './logs/'
            parallel_env = VecMonitor(parallel_env, log_dir)
            
            # Recreate the model with the correct number of environments
            if verbose > 0:
                print(f"Recreating model for {n_envs} parallel environments...")
            self._recreate_model_for_envs(parallel_env, verbose=verbose)
        else:
            self.model.set_env(env)
            
        self.model.verbose = verbose
        
        # Enable TensorBoard logging if specified
        if tensorboard_log:
            self.model.tensorboard_log = tensorboard_log
        
        # Create custom callback for detailed metrics
        callback = CustomCallback(verbose=verbose)
        
        self.model.learn(
            total_timesteps=total_timesteps,
            log_interval=log_interval,
            tb_log_name="custom_cnn_agent",  # TensorBoard log name
            callback=callback,  # Add custom callback
        )


def create_custom_agent_with_renderer(img_width: int = 360, img_height: int = 240):
    """Create a custom agent that uses observation renderer with configurable image dimensions.
    
    Args:
        img_width: Width of synthetic images (default: 360)
        img_height: Height of synthetic images (default: 240)
    
    Returns:
        tuple: (agent, env) - Configured agent and environment
    """
    
    # Create environment using SelfPlayWarehouseBrawl (Gymnasium-compatible)
    from environment.agent import SelfPlayWarehouseBrawl, OpponentsCfg, RewardManager
    env = SelfPlayWarehouseBrawl(
        reward_manager=RewardManager(),
        opponent_cfg=OpponentsCfg(),
        save_handler=None,
        resolution=CameraResolution.LOW
    )
    
    # Create custom agent with configurable image dimensions
    agent = CNNAttentionAgent(
        img_width=img_width,
        img_height=img_height
    )
    agent.get_env_info(env)  # Use Agent's method to set environment info
    
    print(f"Custom CNN + Attention agent created!")
    print(f"  - Image dimensions: {img_width}x{img_height}")
    print(f"  - Memory usage: ~{img_width * img_height * 3 / 1024 / 1024:.1f}MB per image")
    print(f"  - Use agent.learn(env, total_timesteps, n_envs=4) for parallel training")
    
    return agent, env


if __name__ == "__main__":
    # Test the custom agent
    print("Creating custom CNN + Attention agent...")
    agent, env = create_custom_agent_with_renderer()
    
    print("Testing agent prediction...")
    obs, _ = env.reset(seed=42)
    action = agent.predict(obs)
    print(f"Predicted action: {action}")
    
    print("\nTesting parallel environment creation...")
    parallel_env = agent.create_parallel_env(env, n_envs=2, use_subproc=False)  # Use DummyVecEnv for testing
    print(f"Parallel environment created: {type(parallel_env)}")
    
    print("Custom agent created successfully!")
    env.close()
    if hasattr(parallel_env, 'close'):
        parallel_env.close()
    