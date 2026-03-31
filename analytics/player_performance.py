"""
Player Performance - Analytics Module
=====================================
Tasks:
1. Fix Player Tracking IDs (Restrict to exactly 22 stable players)
2. Generate Football Heatmaps (Pending Task 2)
"""

from typing import Dict, Any

class PlayerPerformanceTracker:
    def __init__(self):
        self.player_mapping_ = {}
        self.max_players = 22

    def filter_and_assign_fixed_ids(self, tracks: Dict[str, Any]) -> Dict[str, Any]:
        """
        Cleans the tracks dictionary to ensure exactly 22 fixed, stable player IDs.
        Replaces raw tracker IDs with 'Player_1' to 'Player_22'.
        Removes all spurious tracks (e.g. tracks that are not part of the top 22).
        """
        player_tracks = tracks.get("players", [])
        if not player_tracks:
            return tracks

        # Step 1: Count frequency of each tracker ID across all frames
        id_counts = {}
        for frame_dict in player_tracks:
            for pid in frame_dict.keys():
                id_counts[pid] = id_counts.get(pid, 0) + 1

        # Step 2: Select the top 22 tracker IDs by duration/frequency
        # This assumes the main 22 players are tracked the longest.
        sorted_ids = sorted(id_counts.keys(), key=lambda x: id_counts[x], reverse=True)
        top_22_ids = set(sorted_ids[:self.max_players])

        # Step 3: Create mapping for stable IDs
        self.player_mapping_ = {}
        for idx, original_id in enumerate(sorted_ids[:self.max_players]):
            new_id = f"Player_{idx + 1}"
            self.player_mapping_[original_id] = new_id

        # Step 4: Re-assign and filter tracks in-place
        for frame_idx, frame_dict in enumerate(player_tracks):
            new_frame_dict = {}
            for pid, info in frame_dict.items():
                if pid in top_22_ids:
                    stable_id = self.player_mapping_[pid]
                    new_frame_dict[stable_id] = info
                # Otherwise, it's a spurious track, so drop it (~47 rows limit fixed)
            
            # Replace the old frame dict with the new one containing at most 22 keys
            tracks["players"][frame_idx] = new_frame_dict

        return tracks

