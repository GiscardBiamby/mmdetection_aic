import torch
import torch.nn as nn
from mmdet.registry import MODELS
from mmengine.model import BaseModule


@MODELS.register_module()
class GeoPoseHead(BaseModule):
    def __init__(self, in_channels, hidden_dim=256, out_dim=10, loss_weight=0.1):
        super().__init__()
        self.loss_weight = loss_weight

        self.fc_layers = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

        # Loss functions
        self.loss_mse = nn.MSELoss()
        self.loss_l1 = nn.L1Loss()

    def forward(self, x):
        # x is expected to be a pooled feature vector or similar
        # If x is a tuple of features (from FPN), we might need to pool them or pick one.
        # For simplicity, let's assume the detector passes a single feature vector
        # or we GlobalAveragePool the last feature map.

        # If x is a tuple (P2, P3, P4, P5, P6), we need to handle it.
        # Usually, for image-level prediction, we can use the deepest feature map (P6 or P5)
        # and global average pool it.

        if isinstance(x, (tuple, list)):
            # Use the last feature map (lowest resolution, highest semantic level)
            feat = x[-1]
        else:
            feat = x

        # Global Average Pooling if spatial dimensions exist
        if feat.dim() == 4:
            feat = feat.mean(dim=[2, 3])  # [B, C]

        out = self.fc_layers(feat)

        # Apply activations
        # 1. Sun Elev [0,1] -> Sigmoid
        # 2,3. Sun Az (Sin, Cos) [-1,1] -> Tanh
        # 4. Off-Nadir [0,1] -> Sigmoid
        # 5,6. Sat Az (Sin, Cos) [-1,1] -> Tanh
        # 7,8. Lat, Lon [-1,1] -> Tanh
        # 9,10. Day (Sin, Cos) [-1,1] -> Tanh

        # We can split and apply, or just apply Tanh to all [-1,1] ones and Sigmoid to [0,1] ones.

        # Indices:
        # 0: Sun Elev (Sigmoid)
        # 1,2: Sun Az (Tanh)
        # 3: Off-Nadir (Sigmoid)
        # 4,5: Sat Az (Tanh)
        # 6,7: Lat, Lon (Tanh)
        # 8,9: Day (Tanh)

        sigmoid_indices = [0, 3]

        # Vectorized activation application
        # Create mask for sigmoid indices
        sigmoid_mask = torch.zeros(out.shape[1], dtype=torch.bool, device=out.device)
        sigmoid_mask[sigmoid_indices] = True

        # Expand mask to match batch size [1, 10] -> [B, 10]
        mask_expanded = sigmoid_mask.unsqueeze(0).expand_as(out)

        # Apply both activations to everything (efficient in parallel)
        out_sigmoid = torch.sigmoid(out)
        out_tanh = torch.tanh(out)

        # Select appropriate activation based on mask
        final_out = torch.where(mask_expanded, out_sigmoid, out_tanh)

        return final_out

    @staticmethod
    def unnormalize(vec):
        """Convert 10-d normalized vector back to human readable dict."""
        import math
        import numpy as np

        if isinstance(vec, torch.Tensor):
            vec = vec.detach().cpu().numpy()

        # 1. Sun Elevation
        sun_elev = vec[0] * 90.0

        # 2. Sun Azimuth
        sun_az_rad = math.atan2(vec[1], vec[2])
        sun_az = math.degrees(sun_az_rad) % 360

        # 3. Off-Nadir
        off_nadir = vec[3] * 60.0

        # 4. Satellite Azimuth
        sat_az_rad = math.atan2(vec[4], vec[5])
        sat_az = math.degrees(sat_az_rad) % 360

        # 5. Lat/Lon
        lat = vec[6] * 90.0
        lon = vec[7] * 180.0

        # 6. Day of Year
        day_rad = math.atan2(vec[8], vec[9])
        # day_norm = 2 * pi * (day / 365)
        # day = day_norm * 365 / (2 * pi)
        # atan2 returns [-pi, pi]. We need [0, 2pi] for full year?
        if day_rad < 0:
            day_rad += 2 * math.pi
        day_of_year = day_rad * 365.0 / (2 * math.pi)

        return {
            "sun_elevation": float(sun_elev),
            "sun_azimuth": float(sun_az),
            "off_nadir": float(off_nadir),
            "sat_azimuth": float(sat_az),
            "lat": float(lat),
            "lon": float(lon),
            "day_of_year": float(day_of_year),
        }

    def loss(self, preds, targets):
        # Split into bounded and cyclic components
        # Bounded: 0 (Sun Elev), 3 (Off-Nadir), 6 (Lat), 7 (Lon)
        # Cyclic: 1,2 (Sun Az), 4,5 (Sat Az), 8,9 (Day)

        bounded_indices = [0, 3, 6, 7]
        cyclic_indices = [1, 2, 4, 5, 8, 9]

        preds_bounded = preds[:, bounded_indices]
        targets_bounded = targets[:, bounded_indices]

        preds_cyclic = preds[:, cyclic_indices]
        targets_cyclic = targets[:, cyclic_indices]

        # L1 for bounded variables
        loss_bounded = self.loss_l1(preds_bounded, targets_bounded)

        # MSE for cyclic variables (sin/cos pairs)
        loss_cyclic = self.loss_mse(preds_cyclic, targets_cyclic)

        return {
            "loss_geo_pose_bounded": loss_bounded * self.loss_weight,
            "loss_geo_pose_cyclic": loss_cyclic * self.loss_weight,
        }
