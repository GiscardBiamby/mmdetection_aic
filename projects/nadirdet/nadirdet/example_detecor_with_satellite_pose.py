from mmdet.models.detectors import (
    FasterRCNN as TwoStageDetector,  # example base detector
    SingleStageDetector,
)
from mmdet.registry import MODELS


@MODELS.register_module()
class MyDetectorWithSensorPose(SingleStageDetector):
    """Example detector that uses satellite sensor pose information."""

    def loss(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        batch_inputs,
        batch_data_samples,
        **_kwargs,
    ):
        """Calculate losses from a batch of inputs and data samples."""
        # Extract sensor_pose for each image
        # Depending on where you stored it, it might be in .get or metainfo
        sensor_poses = []
        for ds in batch_data_samples:
            # case 1: we stored it under 'sensor_pose' (data field)
            sp = getattr(ds, "sensor_pose", None)
            if sp is None and "sensor_pose" in ds.metainfo:
                # case 2: stored in metainfo
                sp = ds.metainfo["sensor_pose"]
            sensor_poses.append(sp)

        # Now call the parent loss or your own loss, passing sensor_poses
        # Example: maybe your head.loss takes sensor_poses as extra arg
        feats = self.extract_feat(batch_inputs)
        losses = self.bbox_head.loss(feats, batch_data_samples, sensor_poses=sensor_poses)

        return losses
