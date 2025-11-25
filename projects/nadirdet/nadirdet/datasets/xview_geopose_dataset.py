from mmdet.datasets.coco import CocoDataset
from mmdet.registry import DATASETS


@DATASETS.register_module()
class XViewGeoPoseDataset(CocoDataset):
    def __init__(self, geo_properties: list[str] | tuple | None = None, **kwargs):
        self.geo_properties = geo_properties or (
            "view:sun_elevation",
            "view:sun_azimuth",
            "view:off_nadir",
            "view:azimuth",
            "datetime",
            "bbox",
        )
        super().__init__(**kwargs)

    def parse_data_info(self, raw_data_info: dict) -> dict:
        """Parse raw annotation to target format.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``

        Returns
        -------
            Union[dict, List[dict]]: Parsed annotation.
        """
        data_info = super().parse_data_info(raw_data_info)

        # Extract catalog_info from raw_data_info (which corresponds to an image entry in COCO json)
        # In some MMDetection versions/configs, raw_data_info wraps the image info in 'raw_img_info'
        catalog_info = raw_data_info.get("catalog_info")
        if catalog_info is None and "raw_img_info" in raw_data_info:
            catalog_info = raw_data_info["raw_img_info"].get("catalog_info")

        if catalog_info is None:
            raise ValueError(
                f"Missing 'catalog_info' in raw_data_info for img_id: {data_info.get('img_id', 'unknown')}."
            )

        data_info["catalog_info"] = catalog_info
        data_info["geo_properties"] = self.geo_properties

        return data_info
