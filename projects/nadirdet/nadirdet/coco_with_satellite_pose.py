from mmdet.datasets import CocoDataset
from mmdet.registry import DATASETS

# THis is just an example and was never tested. Assume this doesn't work.


@DATASETS.register_module()
class CocoDatasetWithSensorPose(CocoDataset):
    """CocoDataset that also loads image-level sensor_pose from the JSON.

    Assumes sensor_pose is a key in the image dict of the COCO annotation file.
    """

    def load_data_list(self) -> list[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``"""
        # Let the parent do all the heavy lifting
        data_list = super().load_data_list()

        # Now inject sensor_pose from self.coco.imgs[img_id]
        # into each sample's 'data' dict.
        for sample in data_list:
            img_id = sample["img_id"]  # this is set by CocoDataset
            img_info = self.coco.imgs[img_id]

            # If present, attach it; otherwise set to None
            sensor_pose = img_info.get("sensor_pose", None)
            # Put in 'data'
            sample.setdefault("data", {})
            sample["data"]["sensor_pose"] = sensor_pose

        return data_list
