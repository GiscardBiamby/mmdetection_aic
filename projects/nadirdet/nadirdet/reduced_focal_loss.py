import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS


@MODELS.register_module()
class ReducedFocalLoss(nn.Module):
    """Reduced Focal Loss.

    A variant of focal loss where the focal modulation ( (1 - p_t)^gamma )
    is only applied when p_t < p_thresh. For confident predictions
    (p_t >= p_thresh) it reduces to standard cross-entropy.

    Args:
        use_sigmoid (bool): If True, treat inputs as logits for binary
            classification with sigmoid. If False, treat inputs as logits
            for multi-class classification with softmax.
        gamma (float): Focusing parameter.
        alpha (float or None): Class weighting factor. If None, no alpha
            weighting is applied.
        p_thresh (float): Probability threshold below which focal
            modulation is active.
        reduction (str): 'none', 'mean' or 'sum'.
        loss_weight (float): Scalar multiplier.

    This is a simplified implementation intended to match the high-level
    description from the xView paper (reduced focal loss with probability
    thresholds).
    """

    def __init__(
        self,
        use_sigmoid: bool = True,
        gamma: float = 2.0,
        alpha: float | None = 0.25,
        p_thresh: float = 0.5,
        reduction: str = "mean",
        loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        assert reduction in ("none", "mean", "sum")
        # init the mmdet logger so we can log from this class:
        raise NotImplementedError(
            "ReducedFocalLoss is not yet implemented. The class exists and was written by ChatGPT but not tested yet, and I haven't looekd at the code to see if it is correct. This class is just a placeholder at the moment."
        )
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.p_thresh = p_thresh
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor | None = None,
        avg_factor: float | None = None,
        reduction_override: str | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute loss.

        Args:
            pred: Raw logits. Shape:
                - sigmoid: (N, C) or (N,) for binary
                - softmax: (N, C) for multi-class
            target: Class targets.
                - sigmoid: 0/1 or same shape as pred
                - softmax: int64 class indices, shape (N,)
            weight: Optional per-sample weight.
            avg_factor: Optional normalization factor from sampler.
        """
        reduction = reduction_override if reduction_override else self.reduction

        if self.use_sigmoid:
            loss = self._binary_reduced_focal_loss(pred, target)
        else:
            loss = self._softmax_reduced_focal_loss(pred, target)

        if weight is not None:
            # broadcast if necessary
            if weight.shape != loss.shape:
                weight = weight.view(-1, *[1] * (loss.dim() - 1))
            loss = loss * weight

        if reduction == "mean":
            if avg_factor is not None:
                loss = loss.sum() / (avg_factor + 1e-6)
            else:
                loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()
        # else 'none'

        return loss * self.loss_weight

    def _binary_reduced_focal_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Binary reduced focal loss with sigmoid."""
        # pred: (N, C) or (N,)
        if pred.dim() == 1:
            pred = pred.unsqueeze(1)
        if target.dim() == 1:
            target = target.unsqueeze(1).float()
        else:
            target = target.float()

        # standard BCE loss (per element, no reduction)
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")

        # p_t = p if y=1 else 1-p
        prob = torch.sigmoid(pred)
        p_t = prob * target + (1 - prob) * (1 - target)

        # focal modulation only when p_t < p_thresh
        mod_factor = torch.where(
            p_t < self.p_thresh,
            (1 - p_t).pow(self.gamma),
            torch.ones_like(p_t),
        )

        loss = mod_factor * bce_loss

        # alpha weighting
        if self.alpha is not None:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            loss = alpha_t * loss

        # reduce per-sample (sum over classes if needed)
        return loss.view(loss.size(0), -1).sum(dim=1)

    def _softmax_reduced_focal_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Multi-class reduced focal loss with softmax."""
        # pred: (N, C), target: (N,)
        if target.dim() != 1:
            target = target.view(-1).long()

        log_prob = F.log_softmax(pred, dim=1)
        prob = log_prob.exp()  # (N, C)

        # p_t: probability of the true class
        pt = prob.gather(1, target.view(-1, 1)).view(-1)
        ce_loss = F.nll_loss(log_prob, target, reduction="none")

        # focal modulation only when p_t < p_thresh
        mod_factor = torch.where(
            pt < self.p_thresh,
            (1 - pt).pow(self.gamma),
            torch.ones_like(pt),
        )

        loss = mod_factor * ce_loss

        if self.alpha is not None:
            # per-class alpha, here we just do binary-style:
            # alpha for foreground, (1-alpha) for background
            # you can refine this if needed
            alpha_t = torch.full_like(loss, 1.0)
            # simplistic: class 0 as background
            fg_mask = target != 0
            alpha_t[fg_mask] = self.alpha
            alpha_t[~fg_mask] = 1 - self.alpha
            loss = alpha_t * loss

        return loss
