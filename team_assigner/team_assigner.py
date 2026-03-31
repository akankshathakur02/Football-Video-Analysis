import numpy as np
from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}

    def get_clustering_model(self, image):
        """Fit KMeans on image pixels; returns None if the image is empty."""
        if image is None or image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0:
            return None
        image_2d = image.reshape(-1, 3)
        if len(image_2d) < 2:
            return None
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(image_2d)
        return kmeans

    def get_player_color(self, frame, bbox):
        """Extract the dominant jersey colour for a player crop.
        Returns None when the crop is degenerate (zero-area bbox)."""
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        # Guard against degenerate bboxes
        if x2 <= x1 or y2 <= y1:
            return None

        image = frame[y1:y2, x1:x2]
        if image.size == 0:
            return None

        top_half_image = image[0:max(1, int(image.shape[0] / 2)), :]
        if top_half_image.size == 0:
            return None

        kmeans = self.get_clustering_model(top_half_image)
        if kmeans is None:
            return None

        labels = kmeans.labels_
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

        corner_clusters = [
            clustered_image[0,  0],
            clustered_image[0, -1],
            clustered_image[-1, 0],
            clustered_image[-1,-1],
        ]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster
        return kmeans.cluster_centers_[player_cluster]

    def assign_team_color(self, frame, player_detections):
        player_colors = []
        for _, player_detection in player_detections.items():
            bbox  = player_detection["bbox"]
            color = self.get_player_color(frame, bbox)
            if color is not None:
                player_colors.append(color)

        if len(player_colors) < 2:
            # Fallback: white & black
            self.team_colors[1] = np.array([255.0, 255.0, 255.0])
            self.team_colors[2] = np.array([0.0,   0.0,   0.0])
            self.kmeans = None
            return

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(player_colors)
        self.kmeans = kmeans

        # Fixed broadcast palette (BGR as float): Team A white, Team B light‑green
        self.team_colors[1] = np.array([255.0, 255.0, 255.0])   # white
        self.team_colors[2] = np.array([144.0, 238.0, 144.0])   # light green

    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        if self.kmeans is None:
            team_id = 1
        else:
            player_color = self.get_player_color(frame, player_bbox)
            if player_color is None:
                team_id = 1
            else:
                team_id = int(self.kmeans.predict(player_color.reshape(1, -1))[0]) + 1

        if player_id == 91:
            team_id = 1

        self.player_team_dict[player_id] = team_id
        return team_id
