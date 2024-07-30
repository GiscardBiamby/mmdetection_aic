from mmcv.transforms import BaseTransform
from torch.nn import LayerNorm as LN


class AddInputRes(BaseTransform):
    """ """

    def transform(self, results):
        results["input_res"] = -1
        return results


def _layernorm(dim):
    return LN(dim, eps=1e-6)
