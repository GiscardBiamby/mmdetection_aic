import math
import numpy as np
import torch
from dateutil import parser as date_parser
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadGeoPose(BaseTransform):
    """Load and process geometric properties from catalog_info.

    Produces a 10-d vector:
    1. Sun Elevation (normalized)
    2. Sun Azimuth Sin
    3. Sun Azimuth Cos
    4. Off-Nadir (normalized)
    5. Sat Azimuth Sin
    6. Sat Azimuth Cos
    7. Lat (normalized)
    8. Lon (normalized)
    9. Day of Year Sin
    10. Day of Year Cos
    """

    def __init__(self) -> None:
        super().__init__()

    def transform(self, results: dict) -> dict:
        catalog_info = results.get("catalog_info", {})
        if not catalog_info:
            raise ValueError(
                f"Missing 'catalog_info' img_id: {results.get('img_path', 'unknown')}."
            )

        data = catalog_info.get("data", {})
        properties = data.get("properties", {})

        # 1. Sun Elevation
        sun_elev = properties.get("view:sun_elevation", 0.0)
        feat_1 = sun_elev / 90.0

        # 2. Sun Azimuth
        sun_az = properties.get("view:sun_azimuth", 0.0)
        sun_az_rad = math.radians(sun_az)
        feat_2 = math.sin(sun_az_rad)
        feat_3 = math.cos(sun_az_rad)

        # 3. Off-Nadir
        off_nadir = properties.get("view:off_nadir", 0.0)
        # Fallback to off_nadir_avg if view:off_nadir is missing (though they are usually same)
        if off_nadir == 0.0:
            off_nadir = properties.get("off_nadir_avg", 0.0)
        feat_4 = off_nadir / 60.0

        # 4. Satellite Azimuth
        sat_az = properties.get("view:azimuth", 0.0)
        sat_az_rad = math.radians(sat_az)
        feat_5 = math.sin(sat_az_rad)
        feat_6 = math.cos(sat_az_rad)

        # 5. Geographic Location (Lat/Lon)
        # Try to get from bbox in data
        bbox = data.get("bbox", [])
        if len(bbox) >= 4:
            # bbox is [min_lon, min_lat, max_lon, max_lat] usually in GeoJSON
            # Example: [54.389324, 24.328633, 54.534094, 24.453395]
            # Lat is index 1 and 3. Lon is index 0 and 2.
            min_lon, min_lat, max_lon, max_lat = bbox[0], bbox[1], bbox[2], bbox[3]
            lat_centroid = (min_lat + max_lat) / 2.0
            lon_centroid = (min_lon + max_lon) / 2.0
        else:
            # Fallback if bbox missing
            lat_centroid = 0.0
            lon_centroid = 0.0

        feat_7 = lat_centroid / 90.0
        feat_8 = lon_centroid / 180.0

        # 6. Day of Year
        dt_str = properties.get("datetime", "")
        if dt_str:
            try:
                dt = date_parser.parse(dt_str)
                day_of_year = dt.timetuple().tm_yday
            except:
                day_of_year = 1
        else:
            day_of_year = 1

        day_norm = 2 * math.pi * (day_of_year / 365.0)
        feat_9 = math.sin(day_norm)
        feat_10 = math.cos(day_norm)

        geo_pose = torch.tensor(
            [feat_1, feat_2, feat_3, feat_4, feat_5, feat_6, feat_7, feat_8, feat_9, feat_10],
            dtype=torch.float32,
        )

        results["gt_geo_pose"] = geo_pose
        return results
