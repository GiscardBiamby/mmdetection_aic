from mmdet.datasets import CocoDataset
from mmdet.registry import DATASETS
import torch

@DATASETS.register_module()
class CocoDatasetGSD(CocoDataset):
    def __init__(self, *args, **kwargs):
        super(CocoDatasetGSD, self).__init__(*args, **kwargs)
        
    def __getitem__(self, idx: int) -> dict:
        # coco dataset returns a dict
        # {
        #   "inputs": ...
        #   "data_samples": ... 
        # }
        data = super(CocoDatasetGSD, self).__getitem__(idx)
        data["input_res"] = torch.tensor(0.3)
        # if isinstance(inputs, dict):
        #     print(inputs.keys())
        # elif isinstance(inputs, list | dict):
        #     print(len(inputs))
        # else:
        #     print(type(inputs))
        return data