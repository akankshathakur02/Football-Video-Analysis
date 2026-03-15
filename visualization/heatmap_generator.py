"""
HeatmapGenerator
================
Generates per-player pitch heatmaps from homography-transformed coordinates.

Dependencies: matplotlib, seaborn, numpy
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend – safe for server/CI
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import Dict, List, Any, Optional



# Pitch dimensions (metres) – must match ViewTransformer
PITCH_LENGTH_M: float = 105.0   # full pitch for display; data may cover a section
PITCH_WIDTH_M:  float = 68.0


class HeatmapGenerator:
    """
    Generate per-player heatmaps on a stylised football pitch.

    Parameters
    ----------
    output_dir : str
        Folder where PNG heatmaps are saved (created if absent).
    pitch_length_m : float
        Pitch length used for axis scaling.
    pitch_width_m : float
        Pitch width used for axis scaling.
    """

    def __init__(
        self,
        output_dir:      str   = "heatmaps",
        pitch_length_m:  float = PITCH_LENGTH_M,
        pitch_width_m:   float = PITCH_WIDTH_M,
    ) -> None:
        self.output_dir     = output_dir
        self.pitch_length_m = pitch_length_m
        self.pitch_width_m  = pitch_width_m
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_all(
        self,
        tracks:     Dict[str, List[Dict[int, Any]]],
        team_info:  Optional[Dict[int, int]] = None,
    ) -> None:
        """
        Generate a heatmap PNG for every tracked player.

        Parameters
        ----------
        tracks     : full tracks dict with 'position_transformed' entries.
        team_info  : optional {player_id: team_id} mapping for colouring.
        """
        player_positions = self._collect_positions(tracks)
        team_info = team_info or {}

        for pid, positions in player_positions.items():
            if len(positions) < 5:          # too few points to be meaningful
                continue
            team_id    = team_info.get(pid, 0)
            save_path  = os.path.join(self.output_dir, f"player_{pid}_heatmap.png")
            self._plot_heatmap(pid, positions, team_id, save_path)

        print(f"[HeatmapGenerator] {len(player_positions)} heatmaps saved to '{self.output_dir}/'")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _collect_positions(
        self, tracks: Dict[str, List[Dict[int, Any]]]
    ) -> Dict[int, List[tuple]]:
        """Extract (x, y) pitch coordinates for every player."""
        player_tracks  = tracks.get("players", [])
        player_positions: Dict[int, List[tuple]] = {}

        for frame_dict in player_tracks:
            for pid, info in frame_dict.items():
                pos = info.get("position_transformed")
                if pos is not None:
                    player_positions.setdefault(pid, []).append(tuple(pos))

        return player_positions

    def _plot_heatmap(
        self,
        player_id: int,
        positions: List[tuple],
        team_id:   int,
        save_path: str,
    ) -> None:
        """Render and save one player's heatmap."""
        xs = np.array([p[0] for p in positions], dtype=np.float64)
        ys = np.array([p[1] for p in positions], dtype=np.float64)

        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#16213e")

        # ── pitch outline ────────────────────────────────────────────────────
        self._draw_pitch(ax)

        # ── KDE heatmap ──────────────────────────────────────────────────────
        cmap = "Reds" if team_id == 1 else "Blues" if team_id == 2 else "Greens"
        try:
            sns.kdeplot(
                x=xs,
                y=ys,
                fill=True,
                cmap=cmap,
                alpha=0.65,
                bw_adjust=0.5,
                levels=20,
                ax=ax,
            )
        except Exception:
            # Fallback to scatter if KDE fails (too few unique points)
            ax.scatter(xs, ys, c="yellow", s=10, alpha=0.5)

        # ── scatter overlay ──────────────────────────────────────────────────
        ax.scatter(xs, ys, c="white", s=4, alpha=0.25, zorder=5)

        # ── labels ───────────────────────────────────────────────────────────
        team_label = f"Team {team_id}" if team_id in (1, 2) else "Unknown"
        ax.set_title(
            f"Player {player_id} – {team_label} | Position Heatmap",
            color="white",
            fontsize=14,
            fontweight="bold",
            pad=12,
        )
        ax.set_xlabel("Pitch Length (m)", color="#a0a0b0", fontsize=10)
        ax.set_ylabel("Pitch Width (m)",  color="#a0a0b0", fontsize=10)
        ax.tick_params(colors="#a0a0b0")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")

        # legend
        patch = mpatches.Patch(color="white", label=f"n={len(positions)} frames")
        ax.legend(handles=[patch], facecolor="#222244", labelcolor="white",
                  fontsize=9, loc="upper right")

        plt.tight_layout()
        plt.savefig(save_path, dpi=120, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)

    def _draw_pitch(self, ax: plt.Axes) -> None:
        """Draw a minimal pitch outline (lines only, scaled to data coords)."""
        lw    = 1.2
        col   = "#3a3a5c"
        L     = self.pitch_length_m
        W     = self.pitch_width_m

        # Outer boundary
        rect = plt.Rectangle((0, 0), L, W, linewidth=lw, edgecolor=col, facecolor="none", zorder=2)
        ax.add_patch(rect)

        # Halfway line
        ax.plot([L / 2, L / 2], [0, W], color=col, linewidth=lw, zorder=2)

        # Centre circle – radius 9.15 m
        circle = plt.Circle((L / 2, W / 2), 9.15, linewidth=lw, edgecolor=col, facecolor="none", zorder=2)
        ax.add_patch(circle)

        ax.set_xlim(-2, L + 2)
        ax.set_ylim(-2, W + 2)
        ax.set_aspect("equal")
