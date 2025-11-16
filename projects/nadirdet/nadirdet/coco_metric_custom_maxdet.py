from mmdet.datasets.api_wrappers import COCOeval
from mmdet.evaluation import CocoMetric
from mmdet.registry import METRICS


@METRICS.register_module()
class XViewCocoMetric(CocoMetric):
    def compute_metrics(self, results):
        # copy CocoMetric.compute_metrics but add:
        # coco_eval = COCOeval(...) as usual, then:
        # coco_eval.params.maxDets = [300, 1000]  # or [300] if you like
        # then proceed as in the original implementation
        ...
        return super().compute_metrics(results)
