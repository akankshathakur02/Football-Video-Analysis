"""
DirectionAnalyzer
=================
Classifies each frame-to-frame movement into Forward / Backward / Lateral
based on the heading angle, using pitch coordinates.

Convention (adapts to pitch orientation):
  - The principal axis of play is assumed to be along the X-axis of the
    pitch coordinate system (as set up by ViewTransformer).
  - Heading angles within ±45° of the +X direction → Forward
  - Heading angles within ±45° of the –X direction → Backward
  - All remaining angles                            → Lateral

The thresholds can be overridden at construction time.
"""

import numpy as np
from typing import Dict, List, Any, Tuple

# Angle cone for forward / backward classification (degrees from axis)
FORWARD_BACKWARD_HALF_CONE_DEG: float = 45.0


class DirectionAnalyzer:
    """
    Analyse movement directions from pitch-transformed coordinates.

    Parameters
    ----------
    cone_deg : float
        Half-angle (degrees) used to classify forward vs. lateral movement.
        Default 45° means:  |angle| ≤ 45° → forward, > 135° → backward.
    """

    def __init__(self, cone_deg: float = FORWARD_BACKWARD_HALF_CONE_DEG) -> None:
        self.cone_rad = np.deg2rad(cone_deg)

    def analyze(
        self, tracks: Dict[str, List[Dict[int, Any]]]
    ) -> Dict[int, Dict[str, float]]:
        """
        Compute Forward / Backward / Lateral movement percentages per player.

        Parameters
        ----------
        tracks : dict
            Full tracks dict with 'position_transformed' entries.

        Returns
        -------
        dict  {player_id: {"Forward": pct, "Backward": pct, "Lateral": pct}}
        """
        player_tracks = tracks.get("players", [])
        n_frames      = len(player_tracks)

        # Build ordered position lists
        player_positions: Dict[int, List[Tuple[int, list]]] = {}
        for frame_num in range(n_frames):
            for pid, info in player_tracks[frame_num].items():
                pos = info.get("position_transformed")
                if pos is not None:
                    player_positions.setdefault(pid, []).append((frame_num, pos))

        direction_results: Dict[int, Dict[str, float]] = {}

        for pid, frames_pos in player_positions.items():
            frames_pos.sort(key=lambda x: x[0])

            forward = backward = lateral = 0

            for i in range(1, len(frames_pos)):
                _, p0 = frames_pos[i - 1]
                _, p1 = frames_pos[i]

                dx = p1[0] - p0[0]
                dy = p1[1] - p0[1]

                # Skip effectively stationary frames
                if abs(dx) < 1e-4 and abs(dy) < 1e-4:
                    continue

                angle = np.arctan2(dy, dx)  # in [-π, π]

                if abs(angle) <= self.cone_rad:
                    forward += 1
                elif abs(angle) >= (np.pi - self.cone_rad):
                    backward += 1
                else:
                    lateral += 1

            total = forward + backward + lateral
            if total == 0:
                direction_results[pid] = {"Forward": 0.0, "Backward": 0.0, "Lateral": 0.0}
                continue

            direction_results[pid] = {
                "Forward":  round(forward  / total * 100, 2),
                "Backward": round(backward / total * 100, 2),
                "Lateral":  round(lateral  / total * 100, 2),
            }

        return direction_results
