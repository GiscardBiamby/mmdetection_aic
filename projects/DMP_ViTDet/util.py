from mmcv.transforms import BaseTransform

from numpy import random

class CheckAddGSD(BaseTransform):
    """Add the input gsd of the image if it is missing or < 0."""
    def transform(self, results):
        if "input_res" not in results:
            raise NotImplementedError("need gsd avg")
            results["input_res"] = 0
        elif results["input_res"] < 0:
            raise NotImplementedError("need gsd avg")
            results["input_res"] = 0
        return results

class GSDDropout(BaseTransform):
    """Randomly set the input gsd of the image."""
    def __init__(self, prob=0.1):
        self.prob = prob

    def transform(self, results):
        val = random.uniform(0, 1)
        if val < self.prob:
            raise NotImplementedError("need gsd avg")
            results["input_res"] = 0.3
        return results