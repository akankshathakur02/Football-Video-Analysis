"""
main.py – Football Video Analysis (Extended)
=============================================
Runs the full analytics pipeline:

  1. Read video
  2. Object tracking  (YOLO + ByteTrack)
  3. Camera motion compensation
  4. Homography transformation (pitch coords)
  5. Ball interpolation
  6. Speed & distance estimation
  7. Team assignment
  8. Ball possession assignment
  9. Acceleration & sprint analytics          [NEW]
 10. Speed zone classification                [NEW]
 11. Directional movement analysis            [NEW]
 12. Heatmap generation                       [NEW]
 13. Structured data export (CSV + JSON)      [NEW]
 14. Annotated video output (with new panel)  [NEW]

Usage
-----
    python main.py
"""

import numpy as np
import cv2

# ── existing modules ──────────────────────────────────────────────────────────
from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator

# ── new analytics modules ─────────────────────────────────────────────────────
from analytics import AccelerationAnalyzer, SpeedZoneClassifier, DirectionAnalyzer
<<<<<<< HEAD
from analytics.player_performance import PlayerPerformanceTracker
=======
>>>>>>> 97fe184166cc0685367faade91030cdceb76d801
from export import DataExporter
from visualization import HeatmapGenerator, PossessionOverlay


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
VIDEO_PATH        = "input_videos/08fd33_4.mp4"
OUTPUT_VIDEO_PATH = "output_videos/output_video.mp4"    # MP4 for wider compatibility
ANALYTICS_DIR     = "output_analytics"
HEATMAP_DIR       = "heatmaps"
FRAME_RATE        = 30      # fps of the source video (30 fps confirmed)


def read_video_limited(video_path: str, max_frames: int) -> list:
    """
    Read only `max_frames` frames from the video to stay within RAM.
    Returns a list of BGR numpy arrays.
    """
    import cv2
    cap = cv2.VideoCapture(video_path)
    frames = []
    count  = 0
    while count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        count += 1
    cap.release()
    return frames


def build_team_info(tracks: dict) -> dict:
    """Return {player_id: team_id} from the last annotated frame available."""
    team_info: dict = {}
    for frame_dict in tracks["players"]:
        for pid, info in frame_dict.items():
            if "team" in info:
                team_info[pid] = info["team"]
    return team_info


def main() -> None:
    print("=" * 60)
    print("  Football Video Analysis – Extended Pipeline")
    print("=" * 60)

    # ── 1. Read video ─────────────────────────────────────────────────────────
    print("\n[1/14] Reading video …")
    # Determine how many frames the tracking stub covers so we don't
    # load the entire ~5 500-frame video into RAM at once.
    import pickle as _pkl
    _stub_path = "stubs/track_stubs.pkl"
    with open(_stub_path, "rb") as _f:
        _stub = _pkl.load(_f)
    max_frames = len(_stub.get("players", []))
    del _stub
    video_frames = read_video_limited(VIDEO_PATH, max_frames)
    print(f"       Loaded {len(video_frames)} frames (capped to stub size {max_frames})")

    # ── 2. Object tracking ───────────────────────────────────────────────────
    print("[2/14] Running YOLO + ByteTrack …")
    tracker = Tracker("models/best.pt")
    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=True,
        stub_path="stubs/track_stubs.pkl",
    )
<<<<<<< HEAD
    
    # Task 1: Fix Player Tracking IDs
    print("       Filtering to 22 fixed player IDs (Player_1 to Player_22) …")
    perf_tracker = PlayerPerformanceTracker()
    tracks = perf_tracker.filter_and_assign_fixed_ids(tracks)
    
=======
>>>>>>> 97fe184166cc0685367faade91030cdceb76d801
    tracker.add_position_to_tracks(tracks)

    # ── 3. Camera motion compensation ────────────────────────────────────────
    print("[3/14] Estimating camera motion …")
    cam_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = cam_estimator.get_camera_movement(
        video_frames,
        read_from_stub=True,
        stub_path="stubs/camera_movement_stub.pkl",
    )
    cam_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # ── 4. Perspective transformation ────────────────────────────────────────
    print("[4/14] Applying homography transformation …")
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # ── 5. Ball interpolation ────────────────────────────────────────────────
    print("[5/14] Interpolating ball positions …")
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # ── 6. Speed & distance ──────────────────────────────────────────────────
    print("[6/14] Computing speed and distance …")
    speed_estimator = SpeedAndDistance_Estimator()
    speed_estimator.add_speed_and_distance_to_tracks(tracks)

    # ── 7. Team assignment ───────────────────────────────────────────────────
    print("[7/14] Assigning player teams via jersey-colour clustering …")
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks["players"][0])

    for frame_num, player_track in enumerate(tracks["players"]):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(
                video_frames[frame_num], track["bbox"], player_id
            )
            tracks["players"][frame_num][player_id]["team"]       = team
            tracks["players"][frame_num][player_id]["team_color"] = (
                team_assigner.team_colors[team]
            )

    # ── 8. Ball possession assignment ────────────────────────────────────────
    print("[8/14] Assigning ball possession …")
    player_assigner   = PlayerBallAssigner()
    team_ball_control = []

    for frame_num, player_track in enumerate(tracks["players"]):
        ball_bbox = tracks["ball"][frame_num][1]["bbox"]
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks["players"][frame_num][assigned_player]["has_ball"] = True
            team_ball_control.append(tracks["players"][frame_num][assigned_player]["team"])
        else:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 1)

    team_ball_control = np.array(team_ball_control)

    # ── 9. Acceleration & sprint analytics ──────────────────────────────────
    print("[9/14] Computing acceleration and sprint metrics …")
    accel_analyzer = AccelerationAnalyzer(frame_rate=FRAME_RATE)
    accel_metrics  = accel_analyzer.compute_acceleration_metrics(tracks)

    # ── 10. Speed zone classification ─────────────────────────────────────────
    print("[10/14] Classifying speed zones …")
    zone_classifier = SpeedZoneClassifier()
    zone_stats      = zone_classifier.classify(tracks)

    # ── 11. Directional movement analysis ────────────────────────────────────
    print("[11/14] Analysing movement directions …")
    dir_analyzer    = DirectionAnalyzer()
    direction_stats = dir_analyzer.analyze(tracks)

    # ── 12. Heatmap generation ───────────────────────────────────────────────
    print("[12/14] Generating player heatmaps …")
    team_info         = build_team_info(tracks)
    heatmap_generator = HeatmapGenerator(output_dir=HEATMAP_DIR)
    heatmap_generator.generate_all(tracks, team_info=team_info)

    # ── 13. Structured data export ───────────────────────────────────────────
    print("[13/14] Exporting analytics (CSV + JSON) …")
    exporter = DataExporter(output_dir=ANALYTICS_DIR)
    exporter.export_all(
        tracks,
        accel_metrics,
        zone_stats,
        direction_stats,
        team_ball_control,
    )

    # ── 14. Annotated video output ───────────────────────────────────────────
    print("[14/14] Rendering annotated video …")

    # Base annotations (ellipses, IDs, triangles)
    output_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    # Camera movement overlay
    output_frames = cam_estimator.draw_camera_movement(output_frames, camera_movement_per_frame)

    # Speed & distance text
    output_frames = speed_estimator.draw_speed_and_distance(output_frames, tracks)

    # Premium possession panel + player highlights
    possession_overlay = PossessionOverlay(
        team_colors={
            t: tuple(int(c) for c in color)
            for t, color in team_assigner.team_colors.items()
        }
    )
    output_frames = possession_overlay.annotate_frames(
        output_frames, tracks, team_ball_control
    )

    # Save video (MP4) — write frame-by-frame to avoid double RAM usage
    import os as _os
    _os.makedirs("output_videos", exist_ok=True)
    
    # Explicitly get original session resolution as requested
    _cap = cv2.VideoCapture(VIDEO_PATH)
    _w = int(_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    _h = int(_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    _cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, FRAME_RATE, (_w, _h))
    for frame in output_frames:
        out.write(frame)
    out.release()
    del output_frames   # free RAM

    print(f"\n✔  Annotated video  → {OUTPUT_VIDEO_PATH}")
    print(f"✔  Analytics CSVs   → {ANALYTICS_DIR}/")
    print(f"✔  Heatmaps         → {HEATMAP_DIR}/")
    print("\nPipeline complete.")


if __name__ == "__main__":
    main()