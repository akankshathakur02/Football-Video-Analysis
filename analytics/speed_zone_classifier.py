"""
SpeedZoneClassifier
===================
Classifies each player's frames into FIFA-standard speed zones and
reports the percentage of time spent in each zone.

Zone definition (km/h)
----------------------
Z1  Walking          0  – 7
Z2  Jogging          7  – 14
Z3  Running         14  – 19
Z4  High-speed run  19  – 25
Z5  Sprinting       >25
"""

import numpy as np
from typing import Dict, List, Any

# Zone boundaries (km/h) — upper bound inclusive except last
ZONE_BOUNDARIES: List[float] = [0, 7, 14, 19, 25, float("inf")]
ZONE_LABELS: List[str]        = ["Z1", "Z2", "Z3", "Z4", "Z5"]


class SpeedZoneClassifier:
    """
    Classify player speeds into FIFA zones.

    Usage
    -----
    >>> clf = SpeedZoneClassifier()
    >>> zone_stats = clf.classify(tracks)
    """

    def classify(
        self, tracks: Dict[str, List[Dict[int, Any]]]
    ) -> Dict[int, Dict[str, float]]:
        """
        Compute zone time percentages for every player.

        Parameters
        ----------
        tracks : dict
            Full tracks dict (with 'speed' annotated by SpeedAndDistance_Estimator).

        Returns
        -------
        dict  {player_id: {"Z1": pct, "Z2": pct, ..., "Z5": pct}}
        """
        player_tracks = tracks.get("players", [])

        # Accumulate speed readings per player
        player_speeds: Dict[int, List[float]] = {}
        for frame_dict in player_tracks:
            for pid, info in frame_dict.items():
                spd = info.get("speed")
                if spd is not None:
                    player_speeds.setdefault(pid, []).append(float(spd))

        zone_results: Dict[int, Dict[str, float]] = {}

        for pid, speeds in player_speeds.items():
            arr   = np.array(speeds, dtype=np.float64)
            total = len(arr)
            if total == 0:
                continue

            zone_counts = {}
            for z, label in enumerate(ZONE_LABELS):
                lo = ZONE_BOUNDARIES[z]
                hi = ZONE_BOUNDARIES[z + 1]
                count = int(np.sum((arr >= lo) & (arr < hi)))
                zone_counts[label] = round(count / total * 100, 2)

            zone_results[pid] = zone_counts

        return zone_results
