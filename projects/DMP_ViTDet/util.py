from mmcv.transforms import BaseTransform

from numpy import random

class CheckAddGSD(BaseTransform):
    """Add the input gsd of the image if it is missing or < 0."""
    def __init__(self, gsd=0.45608300352667663):
        self.gsd = gsd
    
    def transform(self, results):
        if "input_res" not in results:
            results["input_res"] = self.gsd
        elif results["input_res"] < 0:
            results["input_res"] = self.gsd
        return results

class GSDDropout(BaseTransform):
    """Randomly set the input gsd of the image."""
    def __init__(self, prob=0.1, gsd=0.45608300352667663):
        self.prob = prob
        self.gsd = gsd

    def transform(self, results):
        val = random.uniform(0, 1)
        if val < self.prob:
            results["input_res"] = self.gsd
        return results