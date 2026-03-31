import pickle
import numpy as np
from trackers import Tracker
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator

def check_metrics():
    with open('stubs/track_stubs.pkl', 'rb') as f:
        tracks = pickle.load(f)
    
    tracker = Tracker('models/best.pt')
    tracks = tracker._clean_tracks(tracks)
    # tracker.add_position_to_tracks(tracks) # already done in cleaning? no

    # Re-run full preprocessing
    tracker.add_position_to_tracks(tracks)
    
    with open('stubs/camera_movement_stub.pkl', 'rb') as f:
        camera_movement = pickle.load(f)
    
    cam_estimator = CameraMovementEstimator(np.zeros((10,10,3), dtype=np.uint8))
    cam_estimator.add_adjust_positions_to_tracks(tracks, camera_movement)
    
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)
    
    speed_estimator = SpeedAndDistance_Estimator()
    speed_estimator.add_speed_and_distance_to_tracks(tracks)
    
    player_count = 0
    with_speed = 0
    with_pos_trans = 0
    for frame_num in range(len(tracks['players'])):
        for pid, info in tracks['players'][frame_num].items():
            player_count += 1
            if 'speed' in info:
                with_speed += 1
            if 'position_transformed' in info and info['position_transformed'] is not None:
                with_pos_trans += 1
                
    print(f"Total player entries: {player_count}")
    print(f"With position_transformed: {with_pos_trans}")
    print(f"With speed: {with_speed}")

if __name__ == "__main__":
    check_metrics()
