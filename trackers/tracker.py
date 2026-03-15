from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
import sys
import os
from typing import TypedDict, List, Dict, Tuple, Any, cast, Optional

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import get_center_of_bbox, get_bbox_width, get_foot_position

# Fully dynamic types to suppress IDE noise
TracksDict = Any

class Tracker:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path) 
        self.tracker = sv.ByteTrack()

    def add_position_to_tracks(self, tracks: Any):
        for player_tracks in tracks["players"]:
            for track_id, track_info in player_tracks.items():
                track_info['position'] = get_foot_position(track_info['bbox'])
        
        for ball_tracks in tracks["ball"]:
            for track_id, track_info in ball_tracks.items():
                track_info['position'] = get_center_of_bbox(track_info['bbox'])

        for ref_tracks in tracks["referees"]:
            for track_id, track_info in ref_tracks.items():
                track_info['position'] = get_foot_position(track_info['bbox'])

    def interpolate_ball_positions(self,ball_positions):
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def detect_frames(self, frames: Any):
        batch_size=20 
        detections = [] 
        for i in range(0,len(frames),batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.5)
            detections += detections_batch
        return detections

    def _clean_tracks(self, tracks: Any) -> Any:
        # 1. Removing false positives based on area/aspect
        for frame_num in range(len(tracks["players"])):
            player_dict = tracks["players"][frame_num]
            tracks["players"][frame_num] = {
                tid: info for tid, info in player_dict.items()
                if (int(info["bbox"][2]) - int(info["bbox"][0])) * (int(info["bbox"][3]) - int(info["bbox"][1])) >= 150 # type: ignore
                and (int(info["bbox"][3]) - int(info["bbox"][1])) >= 0.8 * (int(info["bbox"][2]) - int(info["bbox"][0])) # type: ignore
            }
        
        # 2. Tracking consistency: discard tracks with streaks < 3 
        consecutive_lengths = {}
        last_frame_seen = {}

        for frame_num in range(len(tracks["players"])):
            for track_id in tracks["players"][frame_num].keys(): # type: ignore
                if track_id not in last_frame_seen or frame_num != (last_frame_seen[track_id] + 1):
                    consecutive_lengths[track_id] = 1
                else:
                    consecutive_lengths[track_id] += 1
                last_frame_seen[track_id] = frame_num

        for frame_num in range(len(tracks["players"])):
            player_dict = tracks["players"][frame_num]
            tracks["players"][frame_num] = {
                tid: info for tid, info in player_dict.items()
                if consecutive_lengths.get(tid, 0) >= 3
            }

        # 3. Distance jump filtering
        for track_id, max_streak in consecutive_lengths.items():
            if max_streak < 3: continue
            prev_bbox = None
            for frame_num in range(len(tracks["players"])):
                if track_id in tracks["players"][frame_num]:
                    curr_bbox = tracks["players"][frame_num][track_id]["bbox"]
                    if prev_bbox is not None:
                        cx_p, cy_p = get_center_of_bbox(prev_bbox)
                        cx_c, cy_c = get_center_of_bbox(curr_bbox)
                        if ((cx_p - cx_c)**2 + (cy_p - cy_c)**2)**0.5 > 50:
                            # Build a new dict without this tid for this frame
                            tracks["players"][frame_num] = {
                                tid: info for tid, info in tracks["players"][frame_num].items()
                                if tid != track_id
                            }
                            continue
                    prev_bbox = curr_bbox
        return tracks

    def _smooth_tracks(self, tracks: Any) -> Any:
        # Apply temporal smoothing to player positions: smoothed_position = 0.75 * previous + 0.25 * current
        track_ids: set = set()
        for f_idx in range(len(tracks["players"])):
            track_ids.update(tracks["players"][f_idx].keys())
        
        for tid in track_ids:
            prev_bbox = None
            for f_idx in range(len(tracks["players"])):
                if tid in tracks["players"][f_idx]:
                    curr_track = tracks["players"][f_idx][tid]
                    curr_bbox = curr_track["bbox"]
                    if prev_bbox is not None:
                        # Smoothing formula: 0.75*prev + 0.25*curr
                        p_bbox = cast(List[float], prev_bbox)
                        c_bbox = cast(List[float], curr_bbox)
                        new_bbox = [(0.75 * p_bbox[i] + 0.25 * c_bbox[i]) for i in range(len(p_bbox))]
                        curr_track["bbox"] = cast(List[float], new_bbox)
                    prev_bbox = cast(List[float], curr_track["bbox"])
        return tracks

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            return self._clean_tracks(tracks)

        detections = self.detect_frames(frames)

        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        for frame_num, detection in enumerate(detections):
            cls_names = cast(Dict[int, str], detection.names)
            cls_names_inv = {v: k for k, v in cls_names.items()}

            # Covert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            valid_mask = []
            for i, bbox in enumerate(detection_supervision.xyxy):
                conf = detection_supervision.confidence[i]
                cls_id = detection_supervision.class_id[i]
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                area = w * h
                
                # 1. Filtering detections with low confidence (keep >= 0.5)
                if conf < 0.5:
                    valid_mask.append(False)
                    continue
                
                if cls_id == cls_names_inv.get("player") or cls_names.get(int(cls_id)) == "goalkeeper":
                    # 2. Removing false positives: area and aspect ratio
                    if area < 150 or h < 0.8 * w:
                        valid_mask.append(False)
                        continue
                        
                valid_mask.append(True)
                
            detection_supervision = detection_supervision[np.array(valid_mask)]

            # Convert GoalKeeper to player object
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names.get(int(class_id)) == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cast(int, cls_names_inv.get("player", 0))

            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({}) # type: ignore
            tracks["referees"].append({}) # type: ignore
            tracks["ball"].append({}) # type: ignore

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = int(frame_detection[4])

                if cls_id == cls_names_inv.get('player'):
                    tracks["players"][frame_num][int(track_id)] = {"bbox":bbox} # type: ignore
                
                if cls_id == cls_names_inv.get('referee'):
                    tracks["referees"][frame_num][int(track_id)] = {"bbox":bbox} # type: ignore
            
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv.get('ball'):
                    tracks["ball"][frame_num][1] = {"bbox": bbox} # type: ignore

        tracks = self._clean_tracks(tracks)
        tracks = self._smooth_tracks(tracks)

        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f)

        return tracks
    
    def draw_ellipse(self,frame,bbox,color,track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color = color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height=20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2- rectangle_height//2) +15
        y2_rect = (y2+ rectangle_height//2) +15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect),int(y1_rect) ),
                          (int(x2_rect),int(y2_rect)),
                          color,
                          cv2.FILLED)
            
            x1_text = x1_rect+12
            if track_id > 99:
                x1_text -=10
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text),int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )

        return frame

    def draw_traingle(self,frame,bbox,color):
        y= int(bbox[1])
        x,_ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20],
        ])
        cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2)

        return frame

    def draw_team_ball_control(self,frame,frame_num,team_ball_control):
        # Draw a semi-transparent rectaggle 
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900,970), (255,255,255), -1 )
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        # Get the number of time each team had ball control
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
        team_1 = team_1_num_frames/(team_1_num_frames+team_2_num_frames)
        team_2 = team_2_num_frames/(team_1_num_frames+team_2_num_frames)

        cv2.putText(frame, f"Team 1 Ball Control: {team_1*100:.2f}%",(1400,900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2*100:.2f}%",(1400,950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control):
        """
        Professional broadcast-style overlay.
        Per player:
          • Thin circle marker (r=12) at ground-anchor (feet)
          • Three-line label placed ABOVE the marker
          • Label collision avoidance: shift upward +12px, handle overlapping text
        """
        FONT       = cv2.FONT_HERSHEY_SIMPLEX
        FSCALE     = 0.4    # professional small font
        THICK      = 1
        LINE_AA    = cv2.LINE_AA
        LINE_GAP   = 14    # vertical gap between text lines (px)

        # Fixed team colours for broadcast-style palette (BGR)
        TEAM_COLORS = {
            1: (255, 255, 255),   # Team A – white
            2: (144, 238, 144),   # Team B – light green
        }

        output_video_frames = []

        # Previous stable values for speed and distance to prevent spikes (id: (speed, dist))
        stable_metrics = {}

        for frame_num, frame_img in enumerate(video_frames):
            frame = frame_img.copy()

            player_dict  = tracks["players"][frame_num]
            ball_dict    = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # collect label rects for collision avoidance
            drawn_rects = []

            for track_id, player in player_dict.items():
                bbox = player["bbox"]

                team_id = player.get("team", 1)
                color   = TEAM_COLORS.get(team_id, (255, 255, 255))

                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

                # Anchor text to player foot position
                px = int((x1 + x2) / 2)
                py = int(y2)

                MARKER_R = 12

                # Team info
                team_id    = player.get("team", 1)
                team_color = player.get("team_color")
                # Fallback to hardcoded if dict doesn't have it
                if team_color is None:
                    team_color = TEAM_COLORS.get(team_id, (255, 255, 255))
                
                color = team_color

                # Build label lines
                speed    = player.get("speed")
                distance = player.get("distance")

                # Handle tracking noise by favoring previous values
                if speed is None or speed > 40 or speed < 0: # 40km/h is superhuman
                    if track_id in stable_metrics:
                        speed, distance = stable_metrics[track_id] # type: ignore
                    else:
                        speed, distance = 0.0, 0.0
                else: # Update stable metrics with valid values
                    stable_metrics[track_id] = (speed, distance)

                lines = [str(track_id)]
                if speed is not None:
                    lines.append(f"{speed:.2f} km/h")
                if distance is not None:
                    lines.append(f"{distance:.2f} m")

                # measure each line
                sizes = [cv2.getTextSize(l, FONT, FSCALE, THICK)[0] for l in lines]
                max_w = max(s[0] for s in sizes)
                block_h = int(len(lines) * LINE_GAP)

                # default top-left of text block (above the circle)
                ty = int(py - MARKER_R - 6 - block_h)
                tx = int(px - max_w // 2)

                # collision avoidance
                rect = [int(tx), int(ty), int(tx + max_w), int(ty + block_h)]
                overlap = False
                for _ in range(4): # retry up to 4 times
                    overlap = False
                    for r in drawn_rects:
                        if not (rect[2] < r[0] or rect[0] > r[2] or
                                rect[3] < r[1] or rect[1] > r[3]):
                            overlap = True
                            break
                    if overlap:
                        ty = int(ty) - 12   # shift UPWARD 12 px # type: ignore
                        rect = [int(tx), int(ty), int(tx + max_w), int(ty + block_h)] # type: ignore
                    else:
                        break
                
                # If still overlapping after 4 tries (crowded area), simplify to ID only and reduce marker size
                if overlap:
                    MARKER_R = 10 
                    lines = [str(track_id)]
                    sizes = [cv2.getTextSize(l, FONT, FSCALE, THICK)[0] for l in lines]
                    max_w = int(sizes[0][0])
                    block_h = int(LINE_GAP)
                    ty = int(py - MARKER_R - 6 - block_h)
                    tx = int(px - max_w // 2)
                    rect = [int(tx), int(ty), int(tx + max_w), int(ty + block_h)]
                
                drawn_rects.append(rect)

                # Thin circle at feet 
                cv2.circle(frame, (px, py), MARKER_R, color, 2, LINE_AA)

                # Render each line (black shadow + white/colored text) 
                curr_y = int(ty) + int(LINE_GAP) # baseline starting point
                for i, line in enumerate(lines):
                    w_px = int(sizes[i][0])
                    lx = int(px) - int(w_px) // 2
                    
                    # black outline for readability
                    cv2.putText(frame, str(line), (int(lx), int(curr_y)), int(FONT), float(FSCALE), (0, 0, 0), int(THICK) + 2, int(LINE_AA)) # type: ignore
                    # foreground text 
                    font_color = color if i == 0 else (255, 255, 255) 
                    cv2.putText(frame, str(line), (int(lx), int(curr_y)), int(FONT), float(FSCALE), font_color, int(THICK), int(LINE_AA)) # type: ignore
                    
                    curr_y += int(LINE_GAP)

                # Red triangle above ball-owner 
                if player.get("has_ball", False):
                    frame = self.draw_traingle(frame, bbox, (0, 0, 255))

            # Referees 
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))

            # Ball (green triangle) 
            for _, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["bbox"], (0, 255, 0))

            output_video_frames.append(frame)

        return output_video_frames

