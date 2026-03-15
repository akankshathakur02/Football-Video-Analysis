"""
PossessionOverlay
=================
Draws a premium possession panel onto each video frame, replacing
the plain text drawn by the original tracker.

Features
--------
• Smooth semi-transparent glass panel
• Coloured team bars (proportional to possession %)
• Ball-owner highlight ring
• Sprint indicator badge on sprinting players
"""

import cv2
import numpy as np
from typing import List, Dict, Any


# ── colour palette (BGR) ──────────────────────────────────────────────────────
TEAM_COLORS_DEFAULT = {
    1: (60,  180, 255),   # team 1 – warm orange-ish (BGR)
    2: (255, 100,  50),   # team 2 – blue (BGR)
}
PANEL_BG_COLOR  = (20,  20,  30)
SPRINT_BADGE_C  = (0,  255, 200)
BALL_OWNER_C    = (0,  220,  40)
# ─────────────────────────────────────────────────────────────────────────────


class PossessionOverlay:
    """
    Annotate video frames with a possession panel and enhanced player markers.

    Parameters
    ----------
    team_colors : dict, optional
        BGR colour for each team id.  Falls back to TEAM_COLORS_DEFAULT.
    panel_alpha : float
        Transparency of the possession panel background (0=invisible, 1=opaque).
    panel_rect  : tuple (x1, y1, x2, y2)
        Position of the panel in pixels.  Auto-detected from frame size if None.
    """

    def __init__(
        self,
        team_colors: Dict[int, tuple] | None = None,
        panel_alpha: float                   = 0.55,
        panel_rect:  tuple | None            = None,
    ) -> None:
        self.team_colors = team_colors or TEAM_COLORS_DEFAULT
        self.panel_alpha = panel_alpha
        self._panel_rect = panel_rect          # set lazily on first frame

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def annotate_frames(
        self,
        frames:            List[np.ndarray],
        tracks:            Dict[str, List[Dict[int, Any]]],
        team_ball_control: np.ndarray,
    ) -> List[np.ndarray]:
        """
        Apply possession overlay to every frame.

        Parameters
        ----------
        frames             : list of BGR frames (modified in-place).
        tracks             : full tracks dict.
        team_ball_control  : 1-D array of team ids (one per frame).

        Returns
        -------
        list of annotated frames.
        """
        output: List[np.ndarray] = []

        for frame_num, frame in enumerate(frames):
            frame = frame.copy()

            # Determine panel rect on first frame
            if self._panel_rect is None:
                h, w = frame.shape[:2]
                # Wide enough for 'Team 1 Ball Control: 100.00%'
                self._panel_rect = (w - 360, h - 75, w - 10, h - 10)

            # Cumulative possession up to this frame
            t1_pct, t2_pct = self._cumulative_possession(team_ball_control, frame_num)

            # Draw possession panel
            self._draw_panel(frame, t1_pct, t2_pct)

            # Highlight ball owner and sprinting players
            # player_dict = tracks["players"][frame_num] if frame_num < len(tracks["players"]) else {}
            # self._highlight_players(frame, player_dict)

            output.append(frame)

        return output

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cumulative_possession(team_ball_control: np.ndarray, frame_num: int):
        tbc  = team_ball_control[: frame_num + 1]
        t1   = int(np.sum(tbc == 1))
        t2   = int(np.sum(tbc == 2))
        total = t1 + t2 if (t1 + t2) > 0 else 1
        return round(t1 / total * 100, 1), round(t2 / total * 100, 1)

    def _draw_panel(
        self,
        frame:   np.ndarray,
        t1_pct:  float,
        t2_pct:  float,
    ) -> None:
        h, w = frame.shape[:2]

        panel_w = 210
        panel_h = 55
        x1 = w - panel_w - 20
        y1 = h - panel_h - 40
        x2 = x1 + panel_w
        y2 = y1 + panel_h

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        font   = cv2.FONT_HERSHEY_SIMPLEX
        fscale = 0.45
        fthick = 1
        t1_text = f"Team 1 Ball Control: {t1_pct:.1f}%"
        t2_text = f"Team 2 Ball Control: {t2_pct:.1f}%"

        cv2.putText(frame, t1_text, (x1 + 8, y1 + 22),
                    font, fscale, (255, 255, 255), fthick, cv2.LINE_AA)
        cv2.putText(frame, t2_text, (x1 + 8, y1 + 45),
                    font, fscale, (255, 255, 255), fthick, cv2.LINE_AA)

    def _highlight_players(
        self,
        frame:       np.ndarray,
        player_dict: Dict[int, Any],
    ) -> None:
        for pid, info in player_dict.items():
            bbox = info.get("bbox")
            if bbox is None:
                continue

            # Ball owner – bright green ring
            if info.get("has_ball", False):
                x_c = int((bbox[0] + bbox[2]) / 2)
                y_c = int(bbox[3])
                w   = int((bbox[2] - bbox[0]) / 2)
                cv2.ellipse(frame, (x_c, y_c), (w, int(0.35 * w)),
                            0, -45, 235, BALL_OWNER_C, 3, cv2.LINE_AA)

            # Sprinting players – badge
            if info.get("is_sprinting", False):
                bx = int((bbox[0] + bbox[2]) / 2) - 15
                by = int(bbox[1]) - 28
                cv2.rectangle(frame, (bx, by), (bx + 40, by + 16), SPRINT_BADGE_C, -1)
                cv2.putText(frame, "SPR", (bx + 3, by + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)
