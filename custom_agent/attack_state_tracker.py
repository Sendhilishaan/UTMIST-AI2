# attack_state_tracker.py
# Utility to reconstruct and track attack timing (powers/casts/frames) from observations only

import os
import json
from typing import Dict, Any, List, Tuple

import numpy as np


class AttackStateTracker:
    """
    Tracks per-agent attack timing by mirroring environment power/cast/frame logic using only observations.
    Loads move JSONs (unarmed/spear/hammer), applies frame-halving, and advances a per-agent tracker each step.
    """
    def __init__(self, repo_root: str) -> None:
        self.attack_moves: Dict[int, Dict[int, Dict[str, Any]]] = {}
        self._load_moves(repo_root)
        # Per-agent trackers keyed by 'self' or 'opp'
        self.trackers: Dict[str, Dict[str, Any]] = {
            'self': self._empty_tracker(),
            'opp': self._empty_tracker(),
        }

    def _empty_tracker(self) -> Dict[str, Any]:
        return {
            'move': 0,
            'weapon': 0,
            'active': False,
            'power_idx': 0,
            'cast_idx': 0,
            'frame_idx': 0,
            'recovery_left': 0,
        }

    def reset(self) -> None:
        self.trackers['self'] = self._empty_tracker()
        self.trackers['opp'] = self._empty_tracker()

    def update_from_obs(self, who: str, obs: np.ndarray,
                        idxs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Update tracker for 'self' or 'opp' using obs indices and return active hitboxes for this frame.
        idxs must include keys: move_type, weapon, state (AttackState==8).
        """
        tr = self.trackers[who]
        move_type = int(obs[idxs['move_type']])
        weapon_type = int(obs[idxs['weapon']])
        state_index = int(obs[idxs['state']])
        in_attack_state = (state_index == 8 and move_type > 1)

        if not in_attack_state:
            tr.update({'active': False, 'move': 0, 'weapon': 0, 'power_idx': 0, 'cast_idx': 0, 'frame_idx': 0, 'recovery_left': 0})
            return []

        # If new attack (move/weapon changed or tracker inactive), initialize per JSON
        if (not tr['active']) or tr['move'] != move_type or tr['weapon'] != weapon_type:
            tr['active'] = True
            tr['move'] = move_type
            tr['weapon'] = weapon_type
            tr['power_idx'] = self._get_initial_power_index(weapon_type, move_type)
            tr['cast_idx'] = 0
            tr['frame_idx'] = 0
            tr['recovery_left'] = 0

        # Advance and compute active hitboxes this frame
        md = self.attack_moves.get(weapon_type, {}).get(move_type)
        if not md:
            return []
        return self._advance_and_get_hitboxes(tr, md)

    # ---- internals ----

    def _get_initial_power_index(self, weapon_type: int, move_type: int) -> int:
        md = self.attack_moves.get(weapon_type, {}).get(move_type)
        if not md:
            return 0
        return int(md.get('initialPowerIndex', 0))

    def _advance_and_get_hitboxes(self, tr: Dict[str, Any], move_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        hitboxes: List[Dict[str, Any]] = []
        p_idx = tr['power_idx']
        if not (0 <= p_idx < len(move_data['powers'])):
            return []
        power = move_data['powers'][p_idx]
        casts = power.get('casts', [])

        # Recovery phase
        if tr['recovery_left'] > 0:
            tr['recovery_left'] -= 1
            if tr['recovery_left'] == 0:
                next_idx = int(power.get('onMissNextPowerIndex', -1))
                if 0 <= next_idx < len(move_data['powers']):
                    tr['power_idx'] = next_idx
                    tr['cast_idx'] = 0
                    tr['frame_idx'] = 0
            return []

        # Out of casts -> enter recovery immediately
        if not (0 <= tr['cast_idx'] < len(casts)):
            rec = int(power.get('fixedRecovery', power.get('recovery', 0))) // 2
            tr['recovery_left'] = max(0, rec)
            return []

        c = casts[tr['cast_idx']]
        startup = int(c.get('startupFrames', 0))
        attack = int(c.get('attackFrames', 0))
        idx = tr['frame_idx']

        in_startup = idx < startup
        in_attack = idx < (startup + attack) and not in_startup

        if in_startup:
            tr['frame_idx'] += 1
            return []
        if in_attack:
            hitboxes = c.get('hitboxes', []) or []
            tr['frame_idx'] += 1
            return hitboxes

        # Advance to next cast
        tr['cast_idx'] += 1
        tr['frame_idx'] = 0
        if tr['cast_idx'] >= len(casts):
            rec = int(power.get('fixedRecovery', power.get('recovery', 0))) // 2
            tr['recovery_left'] = max(0, rec)
        return []

    def _load_moves(self, repo_root: str) -> None:
        env_base = os.path.abspath(os.path.join(repo_root, 'environment'))
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
        folders = {
            0: 'unarmed_attacks',
            1: 'spear_attacks',
            2: 'hammer_attacks',
        }
        for wtype, folder in folders.items():
            self.attack_moves[wtype] = {}
            for mt, tail in move_to_tail.items():
                if folder == 'unarmed_attacks':
                    path = os.path.join(env_base, folder, f"Unarmed {tail}")
                elif folder == 'spear_attacks':
                    path = os.path.join(env_base, folder, f"Spear {tail}")
                else:
                    path = os.path.join(env_base, folder, f"Hammer {tail}")
                if not os.path.exists(path):
                    continue
                try:
                    with open(path, 'r') as f:
                        data = json.load(f)
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
                        self.attack_moves[wtype][mt] = move_struct
                except Exception:
                    continue
