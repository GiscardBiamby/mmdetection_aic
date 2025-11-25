import math
import torch
from dateutil import parser as date_parser
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS
from mmdet.datasets.transforms import PackDetInputs


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
        sun_elev = float(properties["view:sun_elevation"])
        feat_1 = sun_elev / 90.0

        # 2. Sun Azimuth
        sun_az = float(properties["view:sun_azimuth"])
        sun_az_rad = math.radians(sun_az)
        feat_2 = math.sin(sun_az_rad)
        feat_3 = math.cos(sun_az_rad)

        # 3. Off-Nadir
        off_nadir = properties.get("view:off_nadir", None)
        # Fallback to off_nadir_avg if view:off_nadir is missing (though they are usually same)
        if off_nadir is None:
            off_nadir = properties["off_nadir_avg"]
        feat_4 = float(off_nadir) / 60.0

        # 4. Satellite Azimuth
        sat_az = float(properties["view:azimuth"])
        sat_az_rad = math.radians(sat_az)
        feat_5 = math.sin(sat_az_rad)
        feat_6 = math.cos(sat_az_rad)

        # 5. Geographic Location (Lat/Lon)
        bbox = data.get("bbox", [])
        if len(bbox) >= 4:
            # This bounding box defines the rectangular extent that encloses the entire footprint of
            # the satellite image on the Earth's surface.
            # bbox is [min_lon, min_lat, max_lon, max_lat] usually in GeoJSON
            # Example: [54.389324, 24.328633, 54.534094, 24.453395]
            # Lat is index 1 and 3. Lon is index 0 and 2.
            min_lon, min_lat, max_lon, max_lat = map(float, bbox[:4])
            lat_centroid = (min_lat + max_lat) / 2.0
            lon_centroid = (min_lon + max_lon) / 2.0
        else:
            # Fallback if bbox missing
            lat_centroid = 0.0
            lon_centroid = 0.0
        # Latitude can range from -90 to +90.
        feat_7 = lat_centroid / 90.0
        # Longitude can range from -180 to +180.
        feat_8 = lon_centroid / 180.0

        # 6. Day of Year
        dt_str = properties["datetime"]
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

        geo_vec = [
            feat_1,  # "view:sun_elevation"
            feat_2,  # sin("view:sun_azimuth")
            feat_3,  # cost("view:sun_azimuth")
            feat_4,  # "view:off_nadir" / 60.0
            feat_5,  # sin("view:azimuth")
            feat_6,  # cost("view:azimuth")
            feat_7,  # lat_centroid / 90.0
            feat_8,  # lon_centroid / 180.0
            feat_9,  # sin(day_of_year("datetime"))
            feat_10,  # cos(day_of_year("datetime"))
        ]

        results["gt_geo_pose"] = torch.tensor(geo_vec, dtype=torch.float32)
        return results


@TRANSFORMS.register_module()
class FlipGeoPose(BaseTransform):
    """Adjust geopose metadata based on the flip augmentation applied to the image.

    Must be placed after RandomFlip in the pipeline.

    * Azimuth Convention: We assume standard compass angles where North is 0°, East is 90°.
    * Cosine Component (Y-axis): Represents the North-South direction. A vertical flip mirrors this
      axis.
    * Sine Component (X-axis): Represents the East-West direction. A horizontal flip mirrors this
      axis.
    * Invariant Features: Elevation, Off-Nadir angle, Latitude, Longitude, and Day of Year are
      unaffected by simple image flips.
    """

    def transform(self, results: dict) -> dict:
        # If no flip was applied, do nothing
        if not results.get("flip", False):
            return results

        gt_geo_pose = results.get("gt_geo_pose")
        if gt_geo_pose is None:
            return results

        direction = results.get("flip_direction")
        if direction is None:
            return results

        # Clone to ensure we don't modify shared memory if that's an issue,
        # though usually pipeline results are unique per worker.
        gt_geo_pose = gt_geo_pose.clone()


        # Vector Indices:
        # 0: Sun Elev
        # 1: Sun Az Sin, 2: Sun Az Cos
        # 3: Off Nadir
        # 4: Sat Az Sin, 5: Sat Az Cos
        # ...

        # Horizontal Flip (Left-Right): Negate Sin (y-axis reflection in unit circle if 0 is North)
        if direction == "horizontal":
            gt_geo_pose[1] *= -1  # Sun Az Sin
            gt_geo_pose[4] *= -1  # Sat Az Sin

        # Vertical Flip (Up-Down): Negate Cos (x-axis reflection)
        elif direction == "vertical":
            gt_geo_pose[2] *= -1  # Sun Az Cos
            gt_geo_pose[5] *= -1  # Sat Az Cos

        # Diagonal Flip (Both): Negate Both
        elif direction == "diagonal":
            gt_geo_pose[1] *= -1
            gt_geo_pose[4] *= -1
            gt_geo_pose[2] *= -1
            gt_geo_pose[5] *= -1
        else:
            raise ValueError(f"Unknown flip_direction: {direction}")

        results["gt_geo_pose"] = gt_geo_pose
        return results


@TRANSFORMS.register_module()
class PackGeoPoseInputs(PackDetInputs):
    """Pack the inputs data for the detection task and include gt_geo_pose."""

    def transform(self, results: dict) -> dict:
        packed_results = super().transform(results)
        data_sample = packed_results["data_samples"]

        if "gt_geo_pose" in results:
            data_sample.gt_geo_pose = results["gt_geo_pose"]

        return packed_results
