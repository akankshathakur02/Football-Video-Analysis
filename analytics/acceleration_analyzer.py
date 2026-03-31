"""
AccelerationAnalyzer
====================
Computes per-player velocity, acceleration, sprint detection,
and explosive-acceleration / deceleration event counting.

All physics is done over the pitch-coordinate positions (metres),
so results are in real-world units:
  - speed        : km/h
  - acceleration : m/s²
"""

import numpy as np
from typing import Dict, List, Any


# ── Tuneable thresholds ────────────────────────────────────────────────────────
SPRINT_SPEED_KMH        = 25.0      # km/h  – FIFA zone 5 boundary
EXPLOSIVE_ACCEL_MS2     = 3.0       # m/s²  – burst start threshold
DECEL_THRESHOLD_MS2     = -2.5      # m/s²  – braking threshold
# ──────────────────────────────────────────────────────────────────────────────


class AccelerationAnalyzer:
    """
    Analyse acceleration and sprint dynamics from tracked positions.

    Parameters
    ----------
    frame_rate : float
        Video frame rate (fps).  Default 24.
    frame_window : int
        Number of frames between successive speed samples (matches the
        SpeedAndDistance_Estimator window so metrics are comparable).
    """

    def __init__(self, frame_rate: float = 24.0, frame_window: int = 5) -> None:
        self.frame_rate   = frame_rate
        self.frame_window = frame_window

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_acceleration_metrics(
        self, tracks: Dict[str, List[Dict[int, Any]]]
    ) -> Dict[int, Dict[str, Any]]:
        """
        Compute acceleration, sprint, and event counts for every player.

        Parameters
        ----------
        tracks : dict
            The full tracks dictionary produced by Tracker.

        Returns
        -------
        dict  {player_id: {metric_name: value, ...}}

        Keys per player
        ---------------
        velocities          list[float]   – speed at each window step (km/h)
        accelerations       list[float]   – acceleration at each window step (m/s²)
        max_acceleration    float
        max_deceleration    float
        sprint_count        int
        explosive_accel_count int
        decel_event_count   int
        """
        player_tracks = tracks.get("players", [])
        n_frames      = len(player_tracks)
        dt            = self.frame_window / self.frame_rate          # seconds per window

        # ── collect per-player ordered (frame, position) pairs ──────────────
        player_positions: Dict[int, List[tuple]] = {}
        for frame_num in range(n_frames):
            for pid, info in player_tracks[frame_num].items():
                pos = info.get("position_transformed")
                if pos is None:
                    continue
                player_positions.setdefault(pid, []).append((frame_num, pos))

        # ── compute metrics ──────────────────────────────────────────────────
        results: Dict[int, Dict[str, Any]] = {}

        for pid, frames_pos in player_positions.items():
            if len(frames_pos) < 2:
                continue

            frames_pos.sort(key=lambda x: x[0])

            velocities:    List[float] = []   # km/h
            accels:        List[float] = []   # m/s²
            prev_speed_ms: float | None = None

            for i in range(1, len(frames_pos)):
                f0, p0 = frames_pos[i - 1]
                f1, p1 = frames_pos[i]

                elapsed_frames = f1 - f0
                if elapsed_frames == 0:
                    continue
                elapsed_s = elapsed_frames / self.frame_rate

                dist_m      = float(np.linalg.norm(np.array(p1) - np.array(p0)))
                speed_ms    = dist_m / elapsed_s
                speed_kmh   = speed_ms * 3.6

                velocities.append(speed_kmh)

                if prev_speed_ms is not None:
                    a = (speed_ms - prev_speed_ms) / elapsed_s
                    accels.append(a)

                prev_speed_ms = speed_ms

            if not velocities:
                continue

            vel_arr   = np.array(velocities, dtype=np.float64)
            accel_arr = np.array(accels,     dtype=np.float64) if accels else np.array([0.0])

            # sprint events: leading edge of speed > threshold
            sprinting      = vel_arr > SPRINT_SPEED_KMH
            sprint_count   = int(np.sum(np.diff(sprinting.astype(int)) == 1))

            # explosive acceleration events
            exp_accel_count = int(np.sum(accel_arr > EXPLOSIVE_ACCEL_MS2))

            # deceleration events (leading edge of strong braking)
            decelerating   = accel_arr < DECEL_THRESHOLD_MS2
            decel_count    = int(np.sum(np.diff(decelerating.astype(int)) == 1)) \
                             if len(accel_arr) > 1 else int(decelerating[0])

            results[pid] = {
                "velocities":              velocities,
                "accelerations":           accel_arr.tolist(),
                "max_acceleration":        float(accel_arr.max()),
                "max_deceleration":        float(accel_arr.min()),
                "sprint_count":            sprint_count,
                "explosive_accel_count":   exp_accel_count,
                "decel_event_count":       decel_count,
            }

        # ── write sprint flag back into tracks ──────────────────────────────
        self._annotate_tracks(tracks, results)

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _annotate_tracks(
        self,
        tracks:  Dict[str, List[Dict[int, Any]]],
        results: Dict[int, Dict[str, Any]],
    ) -> None:
        """Append sprint boolean to every player frame entry."""
        sprint_pids = {
            pid for pid, m in results.items() if m["sprint_count"] > 0
        }
        for frame_dict in tracks.get("players", []):
            for pid, info in frame_dict.items():
                spd = info.get("speed", 0.0)
                info["is_sprinting"] = (spd > SPRINT_SPEED_KMH) and (pid in sprint_pids)
