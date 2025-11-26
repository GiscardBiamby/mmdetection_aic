import torch
import torch.nn as nn
from mmdet.registry import MODELS
from mmengine.model import BaseModule

# FPN feature indices: P2=0, P3=1, P4=2, P5=3, P6=4
GEOPOSE_FPN_LEVELS = (3,)  # Change this to (4,), (3, 4), (2, 3, 4), etc.


@MODELS.register_module()
class GeoPoseHead(BaseModule):
    def __init__(self, in_channels, hidden_dim=256, out_dim=10, loss_weight=0.1):
        super().__init__()
        self.loss_weight = loss_weight

        self.fpn_levels = GEOPOSE_FPN_LEVELS
        # THre 3 is because of the cat of three vectors in `_moment_pool:`
        fc_in_channels = in_channels * 3 * len(self.fpn_levels)

        self.fc_layers = nn.Sequential(
            nn.Linear(fc_in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

        # Loss functions
        self.loss_mse = nn.MSELoss()
        self.loss_l1 = nn.L1Loss()

    def _moment_pool(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Compute pooled features that preserve horizontal or vertical flip-sensitive orientation
        cues.

        x_grid and y_grid are just coordinate ramps in [-1, 1]:
            * x_grid varies across width (left→right): -1 ... +1
            * y_grid varies across height (top→bottom): -1 ... +1

        When you multiply the feature map feat[B,C,H,W] by x_grid[1,1,1,W], broadcasting produces
        [B,C,H,W]. Taking the mean over (H,W) gives [B,C].

        Intuition: this computes the first spatial moment about the image center (a left-right or
        top-bottom "imbalance" summary). Under a horizontal flip, the feature content is mirrored
        but x_grid is fixed in image coordinates, so the x-weighted statistic changes sign (and
        similarly for y under a vertical flip).

        Returns a [B, 3C] vector made of:
          - m0: global mean (flip-invariant)
          - mx: x-weighted mean (changes sign under horizontal flip)
          - my: y-weighted mean (changes sign under vertical flip)
        """
        # feat: [B, C, H, W]
        B, C, H, W = feat.shape

        m0 = feat.mean(dim=[2, 3])  # [B, C]

        # Ensure grid dtype matches feat dtype for mixed precision training
        x_grid = torch.linspace(-1, 1, W, device=feat.device, dtype=feat.dtype).view(1, 1, 1, W)
        y_grid = torch.linspace(-1, 1, H, device=feat.device, dtype=feat.dtype).view(1, 1, H, 1)

        mx = (feat * x_grid).mean(dim=[2, 3])  # [B, C]
        my = (feat * y_grid).mean(dim=[2, 3])  # [B, C]

        return torch.cat([m0, mx, my], dim=1)  # [B, 3C]

    @staticmethod
    def _l2_normalize_pairs(
        vec: torch.Tensor, pairs: list[tuple[int, int]], eps: float = 1e-6
    ) -> torch.Tensor:
        """L2-normalize specified (sin, cos) pairs so each lies on the unit circle."""
        original_dtype = vec.dtype
        out = vec.clone().float()  # Work in float32 for numerical stability
        for i, j in pairs:
            pair = out[:, [i, j]]  # [B, 2]
            denom = torch.sqrt((pair * pair).sum(dim=1, keepdim=True)).clamp_min(eps)  # [B, 1]
            out[:, [i, j]] = pair / denom
        return out.to(original_dtype)  # Convert back to original dtype

    def forward(self, x):
        # x is expected to be a pooled feature vector or similar
        # If x is a tuple of features (from FPN), we might need to pool them or pick one.
        # For simplicity, let's assume the detector passes a single feature vector
        # or we GlobalAveragePool the last feature map.

        # If x is a tuple (P2, P3, P4, P5, P6), we need to handle it.
        # Usually, for image-level prediction, we can use the deepest feature map (P6 or P5)
        # and global average pool it.

        if isinstance(x, (tuple, list)):
            # Use selected FPN levels and moment-pool each, then concatenate
            pooled = []
            for lvl in self.fpn_levels:
                pooled.append(self._moment_pool(x[lvl]))  # [B, 3C]
            feat = torch.cat(pooled, dim=1)  # [B, 3C * NUM_GEOPOSE_FPN_LEVELS]
        elif x.dim() == 4:
            # Behave like the single-level FPN case: moment-pool the 4D map directly
            feat = self._moment_pool(x)  # [B, 3C]
        else:
            raise ValueError(
                f"Expected x to be a tuple or a tensor with shape [B,C,H,W] but got {x.shape}"
            )

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
        """
        L1 for Bounded (linear) vars like sun elevation, lat, lon:

            * L1 is more robust to outliers than MSE
            * These values have straightforward linear relationships (no wraparound)
            * L1 provides consistent gradients regardless of error magnitude

        MSE for Cyclic (sin/cos pairs) vars encoded as (sin θ, cos θ) pairs:

            * MSE penalizes the Euclidean distance between predicted and target unit vectors
            * This is geometrically meaningful: minimizing (sin_pred - sin_gt)² + (cos_pred - cos_gt)²
              is equivalent to minimizing the chord distance between two points on the unit circle
            * MSE's squared penalty discourages large deviations in either component, helping maintain
              the implicit unit circle constraint
        """
        # Split into bounded and cyclic components
        # Bounded: 0 (Sun Elev), 3 (Off-Nadir), 6 (Lat), 7 (Lon)
        # Cyclic: 1,2 (Sun Az), 4,5 (Sat Az), 8,9 (Day)

        bounded_indices = [0, 3, 6, 7]
        cyclic_indices = [1, 2, 4, 5, 8, 9]

        # L2-normalize (sin, cos) pairs before loss/logging
        cyclic_pairs = [(1, 2), (4, 5), (8, 9)]
        preds = preds.clone()
        targets = targets.clone()
        preds = self._l2_normalize_pairs(preds, cyclic_pairs)
        targets = self._l2_normalize_pairs(targets, cyclic_pairs)

        preds_bounded = preds[:, bounded_indices]
        targets_bounded = targets[:, bounded_indices]

        preds_cyclic = preds[:, cyclic_indices]
        targets_cyclic = targets[:, cyclic_indices]

        # L1 for bounded variables
        loss_bounded = self.loss_l1(preds_bounded, targets_bounded)

        # MSE for cyclic variables (sin/cos pairs)
        loss_cyclic = self.loss_mse(preds_cyclic, targets_cyclic)

        losses = {
            "loss_geo_pose_bounded": loss_bounded * self.loss_weight,
            "loss_geo_pose_cyclic": loss_cyclic * self.loss_weight,
        }

        # * mmdet uses the values that start with "loss_" in backprop, any values that don't start
        # *  with "loss_" will only be logged and not affect the model training.

        # Per-component logging (NOT used in training - no 'loss_' prefix)
        with torch.no_grad():
            # Bounded components - L1 (matching actual loss)
            bounded_names = ["sun_elev", "off_nadir", "lat", "lon"]
            per_bounded_l1 = torch.abs(preds_bounded - targets_bounded).mean(dim=0)
            for i, name in enumerate(bounded_names):
                losses[f"geo_{name}_l1"] = per_bounded_l1[i]

            # Cyclic components - MSE for these to match actual loss computed above:
            cyclic_names = [
                "sun_az_sin",
                "sun_az_cos",
                "sat_az_sin",
                "sat_az_cos",
                "day_sin",
                "day_cos",
            ]
            per_cyclic_mse = ((preds_cyclic - targets_cyclic) ** 2).mean(dim=0)
            for i, name in enumerate(cyclic_names):
                losses[f"geo_{name}_mse"] = per_cyclic_mse[i]

            # Human-readable angle/value errors (degrees/units)
            # Sun elevation error (degrees, range 0-90)
            losses["geo_sun_elev_deg_err"] = per_bounded_l1[0] * 90.0

            # Off-nadir error (degrees, range 0-60)
            losses["geo_off_nadir_deg_err"] = per_bounded_l1[1] * 60.0

            # Lat error (degrees, range -90 to 90)
            losses["geo_lat_deg_err"] = per_bounded_l1[2] * 90.0

            # Lon error (degrees, range -180 to 180)
            losses["geo_lon_deg_err"] = per_bounded_l1[3] * 180.0

            # Angular errors for azimuth/cyclic using arccos of dot product
            # Sun azimuth angular error
            sun_az_dot = (preds[:, 1] * targets[:, 1] + preds[:, 2] * targets[:, 2]).clamp(-1, 1)
            losses["geo_sun_az_deg_err"] = torch.acos(sun_az_dot).mean() * (180.0 / 3.14159)

            # Satellite azimuth angular error
            sat_az_dot = (preds[:, 4] * targets[:, 4] + preds[:, 5] * targets[:, 5]).clamp(-1, 1)
            losses["geo_sat_az_deg_err"] = torch.acos(sat_az_dot).mean() * (180.0 / 3.14159)

            # Day of year angular error (convert to days out of 365)
            day_dot = (preds[:, 8] * targets[:, 8] + preds[:, 9] * targets[:, 9]).clamp(-1, 1)
            losses["geo_day_err"] = torch.acos(day_dot).mean() * (365.0 / (2 * 3.14159))

        return losses
