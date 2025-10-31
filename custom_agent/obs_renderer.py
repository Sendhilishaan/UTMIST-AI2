"""
Observation Renderer
Creates simplified RGB visualizations from observation vectors without requiring full rendering.
"""

import numpy as np
import cv2
import json
import os
from typing import Tuple, Dict, List, Any


class ObservationRenderer:
    def __init__(self, width=720, height=480, draw_info=True, draw_hitboxes=False):
        """Create a simplified renderer that uses only observation data."""
        self.width = width
        self.height = height
        # Screen dimensions from environment: width=29.8, height=16.8, center at origin
        # Y-axis is inverted: positive Y points down (like screen coordinates)
        # World bounds match camera's effective field of view with zoom=2.0
        # Camera shows 15.0 x 10.0 tiles, so bounds are [-7.5, 7.5, -5.0, 5.0]
        self.world_bounds = (-7.5, 7.5, -5.0, 5.0)  # (min_x, max_x, min_y, max_y)
        self.draw_info = draw_info
        self.draw_hitboxes = draw_hitboxes

        # Match environment scale for converting JSON hitbox units to world units
        self.BRAWL_TO_UNITS = 1.024 / 320.0

        # Simple temporal smoothing for hitbox visibility (debounce/linger)
        # Tweak to: appear a bit later, disappear quickly
        self.hitbox_debounce_frames = 0  # align closely to env timing
        self.hitbox_linger_frames = 0    # align closely to env timing
        # Per-agent timers/state
        self._hb_state = {
            'self': {
                'visible': False,
                'debounce': 0,
                'linger': 0,
                'last_active': False,
            },
            'opp': {
                'visible': False,
                'debounce': 0,
                'linger': 0,
                'last_active': False,
            }
        }

        # Hardcoded observation indices based on environment.py structure
        # Player observations (first half)
        self.PLAYER_POS = slice(0, 2)           # x, y position
        self.PLAYER_VEL = slice(2, 4)           # x, y velocity
        self.PLAYER_FACING = 4                   # 1.0 = right, 0.0 = left
        self.PLAYER_GROUNDED = 5                 # 1.0 = on ground
        self.PLAYER_AERIAL = 6                   # 1.0 = in air
        self.PLAYER_JUMPS = 7                    # jumps remaining
        self.PLAYER_STATE = 8                    # state index
        self.PLAYER_RECOVERIES = 9               # recoveries left
        self.PLAYER_DODGE_TIMER = 10             # dodge cooldown
        self.PLAYER_STUN_FRAMES = 11             # stun frames remaining
        self.PLAYER_DAMAGE = 12                  # damage (0-700, normalized)
        self.PLAYER_STOCKS = 13                  # lives remaining (0-3)
        self.PLAYER_MOVE_TYPE = 14               # current move type
        self.PLAYER_WEAPON = 15                  # weapon type (0=punch, 1=spear, 2=hammer)
        self.PLAYER_SPAWNER_1 = slice(16, 19)    # x, y, weapon_type
        self.PLAYER_SPAWNER_2 = slice(19, 22)
        self.PLAYER_SPAWNER_3 = slice(22, 25)
        self.PLAYER_SPAWNER_4 = slice(25, 28)
        self.PLAYER_PLATFORM_POS = slice(28, 30) # moving platform x, y
        self.PLAYER_PLATFORM_VEL = slice(30, 32) # moving platform vx, vy
        
        # Opponent observations (second half) - offset by 32 (32 values per player)
        self.OPP_POS = slice(32, 34)
        self.OPP_VEL = slice(34, 36)
        self.OPP_FACING = 36
        self.OPP_GROUNDED = 37
        self.OPP_AERIAL = 38
        self.OPP_JUMPS = 39
        self.OPP_STATE = 40
        self.OPP_RECOVERIES = 41
        self.OPP_DODGE_TIMER = 42
        self.OPP_STUN_FRAMES = 43
        self.OPP_DAMAGE = 44
        self.OPP_STOCKS = 45
        self.OPP_MOVE_TYPE = 46
        self.OPP_WEAPON = 47
        self.OPP_SPAWNER_1 = slice(48, 51)
        self.OPP_SPAWNER_2 = slice(51, 54)
        self.OPP_SPAWNER_3 = slice(54, 57)
        self.OPP_SPAWNER_4 = slice(57, 60)
        self.OPP_PLATFORM_POS = slice(60, 62)
        self.OPP_PLATFORM_VEL = slice(62, 64)
        
        # Preload attack hitbox templates from JSONs (constant data)
        # geometry-only template (legacy)
        self.attack_hitbox_templates: Dict[int, Dict[int, List[Dict[str, Any]]]] = {}
        # timing-aware move data mirroring env powers/casts
        # weapon -> move -> {
        #   'initialPowerIndex': int,
        #   'powers': [ {'casts':[{'startupFrames','attackFrames','hitboxes'}], 'onHitNext':int,'onMissNext':int,'recovery':int,'fixedRecovery':int} ]
        # }
        self.attack_moves: Dict[int, Dict[int, Dict[str, Any]]] = {}
        try:
            self._load_attack_hitbox_templates()
        except Exception:
            # Fail silently; we will fallback to simple approximations
            self.attack_hitbox_templates = {}
            self.attack_moves = {}

        # Dropped weapon tracking (simulate gravity)
        self.dropped_weapons: List[Dict[str, Any]] = []
        self.prev_player_weapon: int = 0
        self.prev_opp_weapon: int = 0
        self.prev_player_pos: Tuple[float, float] = (0.0, 0.0)
        self.prev_opp_pos: Tuple[float, float] = (0.0, 0.0)
        self.gravity_wu: float = 20.0
        self.dt: float = 1.0 / 30.0
        self.dropped_ttl_frames: int = 350
        # Track initial drop positions to match spawners
        self.drop_initial_positions: List[Tuple[float, float]] = []
    def world_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates using the same logic as camera.gtp()."""
        # Match the camera's coordinate system:
        # - Game world center (0,0) maps to screen center (width/2, height/2)
        # - World bounds: x in [-7.5, 7.5], y in [-5.0, 5.0] (effective field of view with zoom=2.0)
        # - Screen bounds: x in [0, width], y in [0, height]
        
        # Calculate scale factor (similar to camera.scale_gtp())
        world_width = self.world_bounds[1] - self.world_bounds[0]  # 15.0
        world_height = self.world_bounds[3] - self.world_bounds[2]  # 10.0
        scale_x = self.width / world_width
        scale_y = self.height / world_height
        
        # Convert to screen coordinates (center-based)
        screen_x = int(self.width / 2 + x * scale_x)
        screen_y = int(self.height / 2 + y * scale_y)
        
        # Clamp to screen bounds
        screen_x = max(0, min(self.width - 1, screen_x))
        screen_y = max(0, min(self.height - 1, screen_y))
        
        return screen_x, screen_y
    
    def render(self, obs: np.ndarray) -> np.ndarray:
        """
        Create a simplified RGB image from observation data.
        
        Args:
            obs: The observation vector (62 or 63 elements)
            
        Returns:
            RGB image as numpy array of shape (height, width, 3)
        """
        # Create black canvas
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Extract player data
        player_pos = obs[self.PLAYER_POS]
        player_facing = obs[self.PLAYER_FACING]
        player_grounded = obs[self.PLAYER_GROUNDED]
        player_damage = obs[self.PLAYER_DAMAGE]
        player_weapon = obs[self.PLAYER_WEAPON]
        player_stocks = obs[self.PLAYER_STOCKS]
        player_state = obs[self.PLAYER_STATE]
        
        # Extract opponent data
        opp_pos = obs[self.OPP_POS]
        opp_facing = obs[self.OPP_FACING]
        opp_grounded = obs[self.OPP_GROUNDED]
        opp_damage = obs[self.OPP_DAMAGE]
        opp_weapon = obs[self.OPP_WEAPON]
        opp_stocks = obs[self.OPP_STOCKS]
        opp_state = obs[self.OPP_STATE]
        
        # Extract platform data (use player's view)
        platform_pos = obs[self.PLAYER_PLATFORM_POS]
        
        # Draw ground platforms (hardcoded positions from environment)
        self._draw_ground(canvas)
        
        # Draw moving platform
        self._draw_platform(canvas, platform_pos)

        # Detect weapon drops (non-Punch -> Punch) and spawn dropped items at last known positions
        self._detect_weapon_drops(player_pos, int(player_weapon), opp_pos, int(opp_weapon))

        # Update and draw dropped weapons with gravity
        self._update_and_draw_dropped_weapons(platform_pos)
        
        # Draw weapon spawners (check all 4)
        # For dropped weapons, use gravity-simulated positions instead of static obs positions
        matched_weapons = set()  # Track matched dropped weapons to avoid double-matching
        for i, spawner_slice in enumerate([
            self.PLAYER_SPAWNER_1, 
            self.PLAYER_SPAWNER_2,
            self.PLAYER_SPAWNER_3,
            self.PLAYER_SPAWNER_4
        ]):
            spawner_data = obs[spawner_slice]
            if spawner_data[2] > 0:  # weapon_type > 0 means active
                spawner_pos = spawner_data[:2]
                # Check if this spawner matches a dropped weapon by comparing to initial or current positions
                matched_dropped = None
                for dw in self.dropped_weapons:
                    if id(dw) in matched_weapons:
                        continue  # Skip already matched
                    # Try matching by initial position first
                    init_pos = dw.get('initial_pos')
                    if init_pos and abs(init_pos[0] - spawner_pos[0]) < 1.0 and abs(init_pos[1] - spawner_pos[1]) < 1.0:
                        matched_dropped = dw
                        break
                    # If initial doesn't match, try current position (in case obs updates later)
                    if abs(dw['x'] - spawner_pos[0]) < 1.0 and abs(dw['y'] - spawner_pos[1]) < 1.0:
                        matched_dropped = dw
                        break
                # Use gravity-simulated position if matched, otherwise use obs position
                if matched_dropped:
                    matched_weapons.add(id(matched_dropped))
                    self._draw_spawner(canvas, [matched_dropped['x'], matched_dropped['y']], int(spawner_data[2]))
                else:
                    self._draw_spawner(canvas, spawner_pos, int(spawner_data[2]))
        
        # Draw players - distinguish self vs opponent (skip when KO state = 11)
        if int(player_state) != 11:
            self._draw_player(canvas, player_pos, player_facing, player_damage, 
                             player_grounded, player_weapon, player_stocks, player_state,
                             color=(255, 0, 0), label="Self")  # Red for self
        if int(opp_state) != 11:
            self._draw_player(canvas, opp_pos, opp_facing, opp_damage,
                             opp_grounded, opp_weapon, opp_stocks, opp_state,
                             color=(0, 0, 255), label="Opponent")  # Blue for opponent

        if self.draw_hitboxes:
            # Approximate attack hitboxes from obs (no per-frame data available)
            # Do not draw when KO
            if int(player_state) != 11:
                self._draw_attack_hitboxes(canvas, player_pos, player_facing, int(obs[self.PLAYER_MOVE_TYPE]),
                                        int(obs[self.PLAYER_WEAPON]), int(player_state), is_self=True)
            if int(opp_state) != 11:
                self._draw_attack_hitboxes(canvas, opp_pos, opp_facing, int(obs[self.OPP_MOVE_TYPE]),
                                        int(obs[self.OPP_WEAPON]), int(opp_state), is_self=False)
                
        if self.draw_info:
            # Add info overlay
            self._draw_info(canvas, player_damage, player_stocks, 
                        opp_damage, opp_stocks)
        
        return canvas

    def _detect_weapon_drops(self, player_pos: np.ndarray, player_weapon: int,
                              opp_pos: np.ndarray, opp_weapon: int) -> None:
        # player
        if hasattr(self, 'prev_player_weapon') and self.prev_player_weapon in (1, 2) and player_weapon == 0:
            init_pos = (float(self.prev_player_pos[0]), float(self.prev_player_pos[1]))
            self.dropped_weapons.append({
                'x': init_pos[0], 
                'y': init_pos[1], 
                'vy': 0.0, 
                'ttl': self.dropped_ttl_frames,
                'initial_pos': init_pos
            })
            self.drop_initial_positions.append(init_pos)
        # opponent
        if hasattr(self, 'prev_opp_weapon') and self.prev_opp_weapon in (1, 2) and opp_weapon == 0:
            init_pos = (float(self.prev_opp_pos[0]), float(self.prev_opp_pos[1]))
            self.dropped_weapons.append({
                'x': init_pos[0], 
                'y': init_pos[1], 
                'vy': 0.0, 
                'ttl': self.dropped_ttl_frames,
                'initial_pos': init_pos
            })
            self.drop_initial_positions.append(init_pos)
        # update previous
        self.prev_player_weapon = player_weapon
        self.prev_opp_weapon = opp_weapon
        self.prev_player_pos = (float(player_pos[0]), float(player_pos[1]))
        self.prev_opp_pos = (float(opp_pos[0]), float(opp_pos[1]))

    def _update_and_draw_dropped_weapons(self, platform_pos: np.ndarray) -> None:
        new_list: List[Dict[str, Any]] = []
        removed_initial_positions = []
        # Bottom of world (y=5.0) - weapons below this should despawn immediately
        world_bottom = 5.0
        
        for w in self.dropped_weapons:
            # integrate gravity
            w['vy'] += self.gravity_wu * self.dt
            w['y'] += w['vy'] * self.dt
            # collide with support
            ground_y = self._get_supporting_surface_y(w['x'], platform_pos)
            if w['y'] > ground_y:
                w['y'] = ground_y
                w['vy'] = 0.0
            
            # Despawn if weapon falls to bottom of world or below
            if w['y'] >= world_bottom:
                # Track removed initial positions for cleanup
                init_pos = w.get('initial_pos')
                if init_pos:
                    removed_initial_positions.append(init_pos)
                continue  # Skip adding to new_list (despawn)
            
            # ttl
            w['ttl'] -= 1
            if w['ttl'] > 0:
                new_list.append(w)
            else:
                # Track removed initial positions for cleanup
                init_pos = w.get('initial_pos')
                if init_pos:
                    removed_initial_positions.append(init_pos)
        self.dropped_weapons = new_list
        # Clean up initial positions for despawned weapons
        for pos in removed_initial_positions:
            if pos in self.drop_initial_positions:
                self.drop_initial_positions.remove(pos)

    def _get_supporting_surface_y(self, x_wu: float, platform_pos: np.ndarray) -> float:
        px, py = float(platform_pos[0]), float(platform_pos[1])
        if (px - 1.0) <= x_wu <= (px + 1.0):
            return py
        if 2.0 <= x_wu <= 7.0:
            return 0.85
        if -7.0 <= x_wu <= -2.0:
            return 2.85
        return 5.0
    
    def _draw_ground(self, canvas: np.ndarray):
        """Draw simplified ground platforms with white color and walls below."""
        ground_color = (255, 255, 255)  # White
        
        # Ground platforms positioned to match environment setup
        # From environment.py: ground1 = Ground(self.space, 4.5, 1, 10)
        # From environment.py: ground2 = Ground(self.space, -4.5, 3, 10)
        # 
        # Ground vertices define the actual collision bounds:
        # Ground 1 body at (4.5, 1): top surface at y=0.85, spans x=[2.0, 7.0]
        # Ground 2 body at (-4.5, 3): top surface at y=2.85, spans x=[-7.0, -2.0]
        
        # Ground 1: top surface at y=0.85, width=5 (from x=2.0 to x=7.0)
        g1_center = self.world_to_screen(4.5, 0.85)  # Use top surface position
        g1_width_pixels = int(5 * self.width / (self.world_bounds[1] - self.world_bounds[0]))
        
        # Draw ground platform
        cv2.rectangle(canvas,
                     (g1_center[0] - g1_width_pixels//2, g1_center[1] - 3),
                     (g1_center[0] + g1_width_pixels//2, g1_center[1] + 3),
                     ground_color, -1)
        
        # Fill area below ground 1 with white (wall)
        wall_bottom = self.height  # Bottom of screen
        cv2.rectangle(canvas,
                     (g1_center[0] - g1_width_pixels//2, g1_center[1] + 3),
                     (g1_center[0] + g1_width_pixels//2, wall_bottom),
                     ground_color, -1)
        
        # Ground 2: top surface at y=2.85, width=5 (from x=-7.0 to x=-2.0)
        g2_center = self.world_to_screen(-4.5, 2.85)  # Use top surface position
        g2_width_pixels = int(5 * self.width / (self.world_bounds[1] - self.world_bounds[0]))
        
        # Draw ground platform
        cv2.rectangle(canvas,
                     (g2_center[0] - g2_width_pixels//2, g2_center[1] - 3),
                     (g2_center[0] + g2_width_pixels//2, g2_center[1] + 3),
                     ground_color, -1)
        
        # Fill area below ground 2 with white (wall)
        cv2.rectangle(canvas,
                     (g2_center[0] - g2_width_pixels//2, g2_center[1] + 3),
                     (g2_center[0] + g2_width_pixels//2, wall_bottom),
                     ground_color, -1)
    
    def _draw_platform(self, canvas: np.ndarray, pos: np.ndarray):
        """Draw the moving platform with accurate dimensions."""
        screen_pos = self.world_to_screen(pos[0], pos[1])
        platform_color = (0, 255, 0)  # Green
        
        # Actual platform dimensions from environment:
        # Stage is created with width=2, height=1, but collision uses height*0.1+0.1 = 0.2
        # Platform image is rotated 90 degrees when rendering, so:
        # Visual width (horizontal extent) = 2.0
        # Visual height (vertical extent) = 0.2
        platform_width_units = 2.0  # Rotated: original height
        platform_height_units = 0.2  # Rotated: original width
        
        # Convert to screen pixels
        world_width = self.world_bounds[1] - self.world_bounds[0]  # 15.0
        world_height = self.world_bounds[3] - self.world_bounds[2]  # 10.0
        scale_x = self.width / world_width
        scale_y = self.height / world_height
        
        platform_width_pixels = int(platform_width_units * scale_x)
        platform_height_pixels = int(platform_height_units * scale_y)
        
        # Draw platform as rectangle with accurate dimensions
        cv2.rectangle(canvas,
                     (screen_pos[0] - platform_width_pixels//2, screen_pos[1] - platform_height_pixels//2),
                     (screen_pos[0] + platform_width_pixels//2, screen_pos[1] + platform_height_pixels//2),
                     platform_color, -1)
    
    def _draw_spawner(self, canvas: np.ndarray, pos: np.ndarray, weapon_type: int):
        """Draw weapon pickup interaction area (capsule) based on env logic."""
        # In env: pickup capsule width ~1.5 world units, height ~0.83 or image-based; use constant 0.83
        pickup_width_wu = 1.5
        pickup_height_wu = 0.83

        center_px = self.world_to_screen(pos[0], pos[1])

        # Convert size to pixels
        world_width = self.world_bounds[1] - self.world_bounds[0]
        world_height = self.world_bounds[3] - self.world_bounds[2]
        scale_x = self.width / world_width
        scale_y = self.height / world_height
        width_px = max(2, int(pickup_width_wu * scale_x))
        height_px = max(2, int(pickup_height_wu * scale_y))

        # Unified weapon color
        color = (255, 0, 255)

        self._draw_outlined_capsule(canvas, center_px, width_px, height_px, color)
    
    def _draw_player(self, canvas: np.ndarray, pos: np.ndarray, facing: float,
                    damage: float, grounded: float, weapon: float, stocks: float,
                    state: float, color: Tuple[int, int, int], label: str):
        """Draw a player with accurate hurtbox and hitbox areas matching the actual environment."""
        screen_pos = self.world_to_screen(pos[0], pos[1])
        
        # Convert world units to screen pixels
        world_width = self.world_bounds[1] - self.world_bounds[0]  # 15.0
        world_height = self.world_bounds[3] - self.world_bounds[2]  # 10.0
        scale_x = self.width / world_width
        scale_y = self.height / world_height
        
        # Actual player hurtbox dimensions in world units (0.928 x 1.024)
        player_width_units = 0.928
        player_height_units = 1.024
        
        # Convert to screen pixels
        player_width_pixels = int(player_width_units * scale_x)
        player_height_pixels = int(player_height_units * scale_y)
        
        # Draw hurtbox as a simple circle (vulnerable area) - this is what shows where agents can be hit
        # Player 0 (Self): Red (255,0,0), Player 1 (Opponent): Blue (0,0,255)
        hurtbox_color = color
        hurtbox_radius = max(player_width_pixels, player_height_pixels) // 2
        cv2.circle(canvas, screen_pos, hurtbox_radius, hurtbox_color, 2)

    def _draw_attack_hitboxes(self, canvas: np.ndarray, pos: np.ndarray, facing: float,
                               move_type: int, weapon_type: int, state_index: int, is_self: bool):
        """Draw attack hitboxes using constant JSON templates when available; fallback to nothing.
        Without frame index in obs, we render all hitboxes defined in the first cast(s) that specify them.
        """
        # Determine if attack should be considered active based on obs
        # AttackState index (from environment.state_mapping) is 8
        is_active = (state_index == 8 and move_type > 1)

        key = 'self' if is_self else 'opp'
        hb = self._hb_state[key]

        # Update debounce/linger
        if is_active:
            if not hb['last_active']:
                hb['debounce'] = self.hitbox_debounce_frames
                hb['linger'] = 0
            else:
                hb['linger'] = 0
            if hb['debounce'] > 0:
                hb['debounce'] -= 1
                hb['visible'] = False
            else:
                hb['visible'] = True
        else:
            if hb['last_active']:
                hb['linger'] = self.hitbox_linger_frames
            if hb['linger'] > 0:
                hb['linger'] -= 1
                hb['visible'] = True
            else:
                hb['visible'] = False

        hb['last_active'] = is_active

        # Maintain per-agent attack timeline (elapsed frames since observed attack start)
        if not hasattr(self, '_attack_state'):
            self._attack_state = {
                'self': {'move': 0, 'weapon': 0, 'elapsed': 0, 'active': False, 'lock_frames': 0,
                         'power_idx': 0, 'cast_idx': 0, 'frame_idx': 0, 'recovery_left': 0},
                'opp': {'move': 0, 'weapon': 0, 'elapsed': 0, 'active': False, 'lock_frames': 0,
                        'power_idx': 0, 'cast_idx': 0, 'frame_idx': 0, 'recovery_left': 0},
            }

        atk = self._attack_state[key]
        if is_active:
            # If a new move is reported mid-attack but the previous is still within its locked timeline,
            # keep rendering the previous move until its timeline completes.
            if atk['active']:
                if atk['elapsed'] < atk.get('lock_frames', 0) and (atk['move'] != move_type or atk['weapon'] != weapon_type):
                    # Ignore switch; continue previous
                    atk['elapsed'] += 1
                    atk['frame_idx'] += 1
                else:
                    if atk['move'] == move_type and atk['weapon'] == weapon_type:
                        atk['elapsed'] += 1
                        atk['frame_idx'] += 1
                    else:
                        # Start a new move timeline and compute lock duration
                        atk['move'] = move_type
                        atk['weapon'] = weapon_type
                        atk['elapsed'] = 0
                        atk['lock_frames'] = self._estimate_move_total_frames(weapon_type, move_type)
                        atk['power_idx'] = self._get_initial_power_index(weapon_type, move_type)
                        atk['cast_idx'] = 0
                        atk['frame_idx'] = 0
                        atk['recovery_left'] = 0
            else:
                atk['move'] = move_type
                atk['weapon'] = weapon_type
                atk['elapsed'] = 0
                atk['lock_frames'] = self._estimate_move_total_frames(weapon_type, move_type)
                atk['power_idx'] = self._get_initial_power_index(weapon_type, move_type)
                atk['cast_idx'] = 0
                atk['frame_idx'] = 0
                atk['recovery_left'] = 0
            atk['active'] = True
        else:
            atk['active'] = False
            atk['elapsed'] = 0
            atk['recovery_left'] = 0

        if not hb['visible']:
            return

        # Scaling from world to pixels
        world_width = self.world_bounds[1] - self.world_bounds[0]
        world_height = self.world_bounds[3] - self.world_bounds[2]
        scale_x = self.width / world_width
        scale_y = self.height / world_height

        # Determine which hitboxes are active using an exact per-frame tracker mirroring env
        hitboxes: List[Dict[str, Any]] = []
        # Use locked move/weapon to select hitboxes (prevents early switching during prior move)
        locked_move = atk['move'] if atk['active'] else move_type
        locked_weapon = atk['weapon'] if atk['active'] else weapon_type
        move_data = self.attack_moves.get(locked_weapon, {}).get(locked_move)
        if move_data:
            hitboxes = self._advance_and_get_hitboxes(atk, move_data)
        else:
            # Fallback to geometry-only template (draw all at once if no timing info)
            templates_by_weapon = self.attack_hitbox_templates.get(weapon_type)
            hitboxes = templates_by_weapon.get(move_type) if templates_by_weapon else []
            if not atk['active']:
                hitboxes = []

        dir_sign = 1 if facing >= 0.5 else -1
        # Requested colors: yellow for red agent, cyan for blue agent
        color = (255, 255, 0) if is_self else (0, 255, 255)

        for hb in hitboxes:
            x_off = hb.get('xOffset', 0)
            y_off = hb.get('yOffset', 0)
            w = hb.get('width', 0)
            h = hb.get('height', 0)

            # Convert to world units (mirror environment's Capsule helpers):
            # world_offset = value * 2 * BRAWL_TO_UNITS; size similarly
            offset_x_wu = (x_off * 2.0 * self.BRAWL_TO_UNITS) * dir_sign
            offset_y_wu = (y_off * 2.0 * self.BRAWL_TO_UNITS)
            width_wu = (w * 2.0 * self.BRAWL_TO_UNITS)
            height_wu = (h * 2.0 * self.BRAWL_TO_UNITS)

            center_world_x = pos[0] + offset_x_wu
            center_world_y = pos[1] + offset_y_wu
            center_px = self.world_to_screen(center_world_x, center_world_y)
            width_px = max(2, int(width_wu * scale_x))
            height_px = max(2, int(height_wu * scale_y))
            self._draw_outlined_capsule(canvas, center_px, width_px, height_px, color)

    def _estimate_move_total_frames(self, weapon_type: int, move_type: int) -> int:
        """Estimate total frames of a move (casts startup+attack + recovery), halved like env.
        Follows onMiss chain for a single hop to approximate chained powers.
        """
        md = self.attack_moves.get(weapon_type, {}).get(move_type)
        if not md:
            return 0
        total = 0
        visited = 0
        idx = int(md.get('initialPowerIndex', 0))
        while 0 <= idx < len(md['powers']) and visited < 2:
            p = md['powers'][idx]
            for c in p.get('casts', []):
                total += int(c.get('startupFrames', 0)) + int(c.get('attackFrames', 0))
            total += int(p.get('fixedRecovery', p.get('recovery', 0))) // 2
            nxt = int(p.get('onMissNextPowerIndex', -1))
            if nxt == -1:
                break
            idx = nxt
            visited += 1
        return total if total > 0 else 0

    def _get_initial_power_index(self, weapon_type: int, move_type: int) -> int:
        md = self.attack_moves.get(weapon_type, {}).get(move_type)
        if not md:
            return 0
        return int(md.get('initialPowerIndex', 0))

    def _advance_and_get_hitboxes(self, atk: Dict[str, Any], move_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        hitboxes: List[Dict[str, Any]] = []
        p_idx = atk['power_idx']
        if not (0 <= p_idx < len(move_data['powers'])):
            return []
        power = move_data['powers'][p_idx]
        casts = power.get('casts', [])
        # Handle recovery
        if atk['recovery_left'] > 0:
            atk['recovery_left'] -= 1
            return []
        # If cast index out of range, enter recovery
        if not (0 <= atk['cast_idx'] < len(casts)):
            rec = int(power.get('fixedRecovery', power.get('recovery', 0))) // 2
            atk['recovery_left'] = max(0, rec)
            if atk['recovery_left'] == 0:
                next_idx = int(power.get('onMissNextPowerIndex', -1))
                if 0 <= next_idx < len(move_data['powers']):
                    atk['power_idx'] = next_idx
                    atk['cast_idx'] = 0
                    atk['frame_idx'] = 0
            return []
        # Current cast timing
        c = casts[atk['cast_idx']]
        startup = int(c.get('startupFrames', 0))
        attack = int(c.get('attackFrames', 0))
        idx = atk['frame_idx']
        in_startup = idx < startup
        in_attack = idx < (startup + attack) and not in_startup
        if in_startup:
            atk['frame_idx'] += 1
            return []
        if in_attack:
            hitboxes = c.get('hitboxes', []) or []
            atk['frame_idx'] += 1
            return hitboxes
        # Move to next cast
        atk['cast_idx'] += 1
        atk['frame_idx'] = 0
        if atk['cast_idx'] >= len(casts):
            rec = int(power.get('fixedRecovery', power.get('recovery', 0))) // 2
            atk['recovery_left'] = max(0, rec)
            if atk['recovery_left'] == 0:
                next_idx = int(power.get('onMissNextPowerIndex', -1))
                if 0 <= next_idx < len(move_data['powers']):
                    atk['power_idx'] = next_idx
                    atk['cast_idx'] = 0
                    atk['frame_idx'] = 0
        return []

    def _load_attack_hitbox_templates(self) -> None:
        """Load per-move hitbox templates from environment JSONs for unarmed, spear, hammer.
        Build both geometry templates and timing-aware cast timelines from the first relevant power.
        weapon_type: 0 Punch, 1 Spear, 2 Hammer (matches environment's mapping).
        """
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        env_base = os.path.abspath(os.path.join(base, 'environment'))

        # Map move_type integer to canonical filename tail
        move_to_tail = {
            2: 'NLight.json',
            3: 'DLight.json',
            4: 'SLight.json',
            5: 'NSig.json',
            6: 'DSig.json',
            7: 'SSig.json',
            8: 'NAir.json',
            9: 'DAir.json',
            10: 'SAir.json',
            11: 'Recovery.json',
            12: 'Groundpound.json',
        }

        def load_dir(prefix: str, weapon_type: int) -> Tuple[Dict[int, List[Dict[str, Any]]], Dict[int, Dict[str, Any]]]:
            res_geo: Dict[int, List[Dict[str, Any]]] = {}
            res_move: Dict[int, Dict[str, Any]] = {}
            for mt, tail in move_to_tail.items():
                path = os.path.join(env_base, prefix, f"{prefix[:-9].capitalize()} {tail}") if prefix.endswith('_attacks') else os.path.join(env_base, prefix, f"{tail}")
                # Handle different folder/file naming:
                # unarmed_attacks -> files start with 'Unarmed'
                if prefix == 'unarmed_attacks':
                    path = os.path.join(env_base, prefix, f"Unarmed {tail}")
                elif prefix == 'spear_attacks':
                    path = os.path.join(env_base, prefix, f"Spear {tail}")
                elif prefix == 'hammer_attacks':
                    path = os.path.join(env_base, prefix, f"Hammer {tail}")

                if not os.path.exists(path):
                    continue
                try:
                    with open(path, 'r') as f:
                        data = json.load(f)
                    # Choose the first power that contains any hitboxes in any cast (closest to what env uses)
                    chosen_power = None
                    for power in data.get('powers', []):
                        if any('hitboxes' in c for c in power.get('casts', [])):
                            chosen_power = power
                            break
                    if chosen_power is None and data.get('powers'):
                        chosen_power = data['powers'][0]

                    # Build full move structure with initialPowerIndex and all powers/casts
                    move_struct: Dict[str, Any] = {
                        'initialPowerIndex': int(data.get('move', {}).get('initialPowerIndex', 0)),
                        'powers': []
                    }
                    for power in data.get('powers', []):
                        p_entry = {
                            'onHitNextPowerIndex': int(power.get('onHitNextPowerIndex', -1)),
                            'onMissNextPowerIndex': int(power.get('onMissNextPowerIndex', -1)),
                            'recovery': int(power.get('recovery', 0)),
                            'fixedRecovery': int(power.get('fixedRecovery', 0)),
                            'casts': []
                        }
                        for cast in power.get('casts', []):
                            p_entry['casts'].append({
                                'startupFrames': int(cast.get('startupFrames', 0)) // 2,
                                'attackFrames': int(cast.get('attackFrames', 0)) // 2,
                                'hitboxes': cast.get('hitboxes', []) or []
                            })
                        move_struct['powers'].append(p_entry)

                    if move_struct['powers']:
                        res_move[mt] = move_struct
                        # Geometry template: union of all hitboxes in all casts of first chosen power (for fallback)
                        geo: List[Dict[str, Any]] = []
                        for p in move_struct['powers']:
                            for c in p['casts']:
                                geo.extend(c['hitboxes'])
                        if geo:
                            res_geo[mt] = geo
                except Exception:
                    pass
            return res_geo, res_move

        # weapon_type mapping: 0 Punch (unarmed), 1 Spear, 2 Hammer
        templates_geo: Dict[int, Dict[int, List[Dict[str, Any]]]] = {}
        move_data_map: Dict[int, Dict[int, Dict[str, Any]]] = {}
        geo0, mv0 = load_dir('unarmed_attacks', 0)
        geo1, mv1 = load_dir('spear_attacks', 1)
        geo2, mv2 = load_dir('hammer_attacks', 2)
        templates_geo[0], move_data_map[0] = geo0, mv0
        templates_geo[1], move_data_map[1] = geo1, mv1
        templates_geo[2], move_data_map[2] = geo2, mv2
        self.attack_hitbox_templates = templates_geo
        self.attack_moves = move_data_map
    
    def _draw_capsule(self, canvas: np.ndarray, center: tuple, width: int, height: int, color: tuple):
        """Draw a capsule (rounded rectangle) to match the actual hurtbox collision."""
        x, y = center
        
        # Calculate radius for rounded ends
        radius = min(width, height) // 2
        
        # Draw the main rectangular body
        if width > height:
            # Horizontal capsule: rectangle + two circles on sides
            rect_x = x - width//2 + radius
            rect_y = y - height//2
            rect_width = width - 2 * radius
            rect_height = height
            
            # Draw rectangle
            cv2.rectangle(canvas, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), color, -1)
            
            # Draw left circle
            cv2.circle(canvas, (x - width//2 + radius, y), radius, color, -1)
            # Draw right circle
            cv2.circle(canvas, (x + width//2 - radius, y), radius, color, -1)
        else:
            # Vertical capsule: rectangle + two circles on top/bottom
            rect_x = x - width//2
            rect_y = y - height//2 + radius
            rect_width = width
            rect_height = height - 2 * radius
            
            # Draw rectangle
            cv2.rectangle(canvas, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), color, -1)
            
            # Draw top circle
            cv2.circle(canvas, (x, y - height//2 + radius), radius, color, -1)
            # Draw bottom circle
            cv2.circle(canvas, (x, y + height//2 - radius), radius, color, -1)
    
    def _draw_outlined_capsule(self, canvas: np.ndarray, center: tuple, width: int, height: int, color: tuple):
        """Draw an outlined capsule (rounded rectangle) exactly matching environment's draw_hithurtbox algorithm."""
        x, y = center
        
        # Follow the exact same logic as environment's draw_hithurtbox method
        if width < height:
            # Vertical Capsule
            radius = width // 2
            half_height = height // 2
            circle_height = half_height - radius
            
            # Arc centers
            top_center = (x, y - circle_height)
            bottom_center = (x, y + circle_height)
            
            # Draw arcs (AA for smooth joins)
            cv2.ellipse(canvas, top_center, (radius, radius), 180, 0, 180, color, 2, lineType=cv2.LINE_AA)
            cv2.ellipse(canvas, bottom_center, (radius, radius), 0, 0, 180, color, 2, lineType=cv2.LINE_AA)
            
            # Draw vertical lines spanning from top_center.y to bottom_center.y
            cv2.line(canvas, (x - radius, top_center[1]), (x - radius, bottom_center[1]), color, 2, lineType=cv2.LINE_AA)
            cv2.line(canvas, (x + radius, top_center[1]), (x + radius, bottom_center[1]), color, 2, lineType=cv2.LINE_AA)
            
        elif width == height:
            # Circular Capsule (matching environment line 4760)
            cv2.circle(canvas, (x, y), width // 2, color, 2)
            
        else:
            # Horizontal Capsule
            radius = height // 2
            half_width = width // 2
            circle_width = half_width - radius
            
            # Arc centers
            right_center = (x + circle_width, y)
            left_center = (x - circle_width, y)
            
            # Draw arcs (AA for smooth joins)
            cv2.ellipse(canvas, right_center, (radius, radius), 270, 0, 180, color, 2, lineType=cv2.LINE_AA)
            cv2.ellipse(canvas, left_center, (radius, radius), 90, 0, 180, color, 2, lineType=cv2.LINE_AA)
            
            # Draw horizontal lines spanning from left_center.x to right_center.x
            cv2.line(canvas, (left_center[0], y - radius), (right_center[0], y - radius), color, 2, lineType=cv2.LINE_AA)
            cv2.line(canvas, (left_center[0], y + radius), (right_center[0], y + radius), color, 2, lineType=cv2.LINE_AA)
    
    def _draw_info(self, canvas: np.ndarray, p1_damage: float, p1_stocks: float,
                  p2_damage: float, p2_stocks: float):
        """Draw damage and stock information."""
        # Convert normalized damage back to percentage
        p1_dmg_pct = p1_damage * 500
        p2_dmg_pct = p2_damage * 500
        
        # Draw P1 info (top left)
        cv2.putText(canvas, f"P1: {p1_dmg_pct:.1f}%", 
                   (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
        cv2.putText(canvas, f"Lives: {int(p1_stocks)}", 
                   (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
        
        # Draw P2 info (top right)
        cv2.putText(canvas, f"P2: {p2_dmg_pct:.1f}%", 
                   (self.width - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 191, 255), 2)
        cv2.putText(canvas, f"Lives: {int(p2_stocks)}", 
                   (self.width - 150, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 191, 255), 2)


def render_observation(obs: np.ndarray, width=720, height=480) -> np.ndarray:
    """
    Render an observation as an RGB image.
    
    Args:
        obs: Observation vector from the environment
        width: Output image width
        height: Output image height
    
    Returns:
        RGB numpy array of shape (height, width, 3)
    """
    renderer = ObservationRenderer(width=width, height=height)
    return renderer.render(obs)

