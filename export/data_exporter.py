"""
DataExporter
============
Writes all structured analytics to CSV and JSON files.

Output files
------------
  speeds.csv          – avg/max speed, total distance per player
  acceleration.csv    – max accel, decel count, sprint count per player
  speed_zones.csv     – % time in each FIFA zone per player
  full_summary.csv    – combined summary table
  direction_analysis.csv – forward / backward / lateral percentages
  metrics.json        – all metrics in one JSON document
"""

import csv
import json
import os
import sys
from typing import Dict, Any



class DataExporter:
    """
    Export player analytics to CSV and JSON.

    Parameters
    ----------
    output_dir : str
        Directory where all CSV/JSON files are written.
    """

    def __init__(self, output_dir: str = "output_analytics") -> None:
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _path(self, filename: str) -> str:
        return os.path.join(self.output_dir, filename)

    @staticmethod
    def _write_csv(filepath: str, fieldnames: list, rows: list) -> None:
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    # ── public API ────────────────────────────────────────────────────────────

    def export_speeds(
        self, tracks: Dict[str, Any]
    ) -> None:
        """
        Export speeds.csv  (Player_ID | Avg_Speed_kmh | Max_Speed_kmh | Total_Distance_m).
        Reads 'speed' and 'distance' fields annotated by SpeedAndDistance_Estimator.
        """
        player_tracks = tracks.get("players", [])

        # Gather all speed samples and the last distance reading per player
        speed_samples: Dict[int, list]  = {}
        last_distance:  Dict[int, float] = {}

        for frame_dict in player_tracks:
            for pid, info in frame_dict.items():
                spd  = info.get("speed")
                dist = info.get("distance")
                if spd is not None:
                    speed_samples.setdefault(pid, []).append(spd)
                if dist is not None:
                    last_distance[pid] = dist

        rows = []
        for pid in sorted(speed_samples):
            speeds = speed_samples[pid]
            rows.append({
                "Player_ID":        pid,
                "Avg_Speed_kmh":    round(sum(speeds) / len(speeds), 3),
                "Max_Speed_kmh":    round(max(speeds), 3),
                "Total_Distance_m": round(last_distance.get(pid, 0.0), 3),
            })

        self._write_csv(
            self._path("speeds.csv"),
            ["Player_ID", "Avg_Speed_kmh", "Max_Speed_kmh", "Total_Distance_m"],
            rows,
        )

    def export_acceleration(
        self, accel_metrics: Dict[int, Dict[str, Any]]
    ) -> None:
        """
        Export acceleration.csv
        (Player_ID | Max_Acceleration | Deceleration_Count | Sprint_Count).
        """
        rows = []
        for pid in sorted(accel_metrics):
            m = accel_metrics[pid]
            rows.append({
                "Player_ID":          pid,
                "Max_Acceleration":   round(m.get("max_acceleration", 0.0), 4),
                "Sprint_Count":       m.get("sprint_count", 0),
            })

        self._write_csv(
            self._path("acceleration.csv"),
            ["Player_ID", "Max_Acceleration", "Sprint_Count"],
            rows,
        )

    def export_speed_zones(
        self, zone_stats: Dict[int, Dict[str, float]]
    ) -> None:
        """
        Export speed_zones.csv
        (Player_ID | Z1_percent | Z2_percent | Z3_percent | Z4_percent | Z5_percent).
        """
        rows = []
        for pid in sorted(zone_stats):
            z = zone_stats[pid]
            rows.append({
                "Player_ID":  pid,
                "Z1_percent": z.get("Z1", 0.0),
                "Z2_percent": z.get("Z2", 0.0),
                "Z3_percent": z.get("Z3", 0.0),
                "Z4_percent": z.get("Z4", 0.0),
                "Z5_percent": z.get("Z5", 0.0),
            })

        self._write_csv(
            self._path("speed_zones.csv"),
            ["Player_ID", "Z1_percent", "Z2_percent", "Z3_percent", "Z4_percent", "Z5_percent"],
            rows,
        )

    def export_direction_analysis(
        self, direction_stats: Dict[int, Dict[str, float]]
    ) -> None:
        """
        Export direction_analysis.csv
        (Player_ID | Forward_percent | Backward_percent | Lateral_percent).
        """
        rows = []
        for pid in sorted(direction_stats):
            d = direction_stats[pid]
            rows.append({
                "Player_ID":        pid,
                "Forward_percent":  d.get("Forward",  0.0),
                "Backward_percent": d.get("Backward", 0.0),
                "Lateral_percent":  d.get("Lateral",  0.0),
            })

        self._write_csv(
            self._path("direction_analysis.csv"),
            ["Player_ID", "Forward_percent", "Backward_percent", "Lateral_percent"],
            rows,
        )

    def export_full_summary(
        self,
        tracks:          Dict[str, Any],
        accel_metrics:   Dict[int, Dict[str, Any]],
        zone_stats:      Dict[int, Dict[str, float]],
    ) -> None:
        """
        Export full_summary.csv combining all key metrics.
        """
        player_tracks = tracks.get("players", [])

        speed_samples: Dict[int, list]  = {}
        last_distance:  Dict[int, float] = {}
        for frame_dict in player_tracks:
            for pid, info in frame_dict.items():
                spd  = info.get("speed")
                dist = info.get("distance")
                if spd is not None:
                    speed_samples.setdefault(pid, []).append(spd)
                if dist is not None:
                    last_distance[pid] = dist

        all_pids = sorted(set(speed_samples) | set(accel_metrics) | set(zone_stats))
        rows = []
        for pid in all_pids:
            speeds = speed_samples.get(pid, [0.0])
            m      = accel_metrics.get(pid, {})
            z      = zone_stats.get(pid, {})
            rows.append({
                "Player_ID":        pid,
                "Avg_Speed_kmh":    round(sum(speeds) / len(speeds), 3),
                "Max_Speed_kmh":    round(max(speeds), 3),
                "Total_Distance_m": round(last_distance.get(pid, 0.0), 3),
                "Max_Acceleration": round(m.get("max_acceleration", 0.0), 4),
                "Sprint_Count":     m.get("sprint_count", 0),
                "Z5_percent":  z.get("Z5", 0.0),
            })

        self._write_csv(
            self._path("full_summary.csv"),
            ["Player_ID", "Avg_Speed_kmh", "Max_Speed_kmh", "Total_Distance_m",
             "Max_Acceleration", "Sprint_Count", "Z5_percent"],
            rows,
        )

    def export_metrics_json(
        self,
        tracks:          Dict[str, Any],
        accel_metrics:   Dict[int, Dict[str, Any]],
        zone_stats:      Dict[int, Dict[str, float]],
        direction_stats: Dict[int, Dict[str, float]],
        team_ball_control: Any,
    ) -> None:
        """
        Export metrics.json with all player statistics and possession data.
        """
        import numpy as np

        # speed / distance
        player_tracks = tracks.get("players", [])
        speed_samples: Dict[int, list]  = {}
        last_distance:  Dict[int, float] = {}
        for frame_dict in player_tracks:
            for pid, info in frame_dict.items():
                spd  = info.get("speed")
                dist = info.get("distance")
                if spd is not None:
                    speed_samples.setdefault(pid, []).append(spd)
                if dist is not None:
                    last_distance[pid] = dist

        players_json = {}
        for pid in sorted(set(speed_samples) | set(accel_metrics) | set(zone_stats)):
            speeds = speed_samples.get(pid, [0.0])
            m      = accel_metrics.get(pid, {})
            z      = zone_stats.get(pid, {})
            d      = direction_stats.get(pid, {})

            # strip non-serialisable arrays
            accel_safe = {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in m.items()
            }

            players_json[str(pid)] = {
                "speed": {
                    "avg_kmh":        round(sum(speeds) / len(speeds), 3),
                    "max_kmh":        round(max(speeds), 3),
                    "total_distance": round(last_distance.get(pid, 0.0), 3),
                },
                "acceleration": accel_safe,
                "speed_zones":  z,
                "direction":    d,
            }

        # possession
        tbc = np.array(team_ball_control)
        t1  = int(np.sum(tbc == 1))
        t2  = int(np.sum(tbc == 2))
        tot = t1 + t2 if (t1 + t2) > 0 else 1
        possession = {
            "team_1_percent": round(t1 / tot * 100, 2),
            "team_2_percent": round(t2 / tot * 100, 2),
        }

        payload = {"players": players_json, "possession": possession}

        with open(self._path("metrics.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    # ── convenience: run all exports ─────────────────────────────────────────

    def export_all(
        self,
        tracks:            Dict[str, Any],
        accel_metrics:     Dict[int, Dict[str, Any]],
        zone_stats:        Dict[int, Dict[str, float]],
        direction_stats:   Dict[int, Dict[str, float]],
        team_ball_control,
    ) -> None:
        """Run every export in one call."""
        self.export_speeds(tracks)
        self.export_acceleration(accel_metrics)
        self.export_speed_zones(zone_stats)
        self.export_direction_analysis(direction_stats)
        self.export_full_summary(tracks, accel_metrics, zone_stats)
        self.export_metrics_json(
            tracks, accel_metrics, zone_stats, direction_stats, team_ball_control
        )
        print(f"[DataExporter] All analytics written to '{self.output_dir}/'")
