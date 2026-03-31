import traceback
try:
    import pickle
    import numpy as np
    from trackers import Tracker
    from camera_movement_estimator import CameraMovementEstimator
    from view_transformer import ViewTransformer
    from speed_and_distance_estimator import SpeedAndDistance_Estimator

    f = open('stubs/track_stubs.pkl', 'rb')
    tracks = pickle.load(f)

    tracker = Tracker('models/best.pt')
    tracks = tracker._clean_tracks(tracks)
    tracks = tracker._smooth_tracks(tracks)
    tracker.add_position_to_tracks(tracks)

    f2 = open('stubs/camera_movement_stub.pkl', 'rb')
    cam_movement = pickle.load(f2)
    cam_estimator = CameraMovementEstimator(np.zeros((10,10,3)))
    cam_estimator.add_adjust_positions_to_tracks(tracks, cam_movement)

    vt = ViewTransformer()
    vt.add_transformed_position_to_tracks(tracks)

    s = SpeedAndDistance_Estimator()
    s.add_speed_and_distance_to_tracks(tracks)

    c = sum(1 for p in tracks['players'] for track_id, t in p.items() if 'speed' in t)
    print(f'Total player items with speed: {c}')
except Exception as e:
    with open('error.txt', 'w') as f:
        traceback.print_exc(file=f)
