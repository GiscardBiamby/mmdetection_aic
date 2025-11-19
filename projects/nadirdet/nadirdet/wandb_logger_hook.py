from mmdet.registry import HOOKS
from mmengine.hooks import Hook
from mmengine.runner import Runner

try:
    import wandb
except ImportError:
    wandb = None


@HOOKS.register_module()
class WandbEpochLoggerHook(Hook):
    """
    Logs the current epoch to Weights & Biases.
    This allows using 'epoch' as a custom x-axis in the W&B UI.
    """

    def __init__(self):
        if wandb is None:
            raise RuntimeError('wandb is not installed, please install it with "pip install wandb"')
        self.wandb = wandb

    def before_run(self, runner: Runner) -> None:
        """Define epoch as a metric in WandB so it appears as an x-axis option."""
        # only log if a run is active (this is true on rank 0 when WandbVisBackend is enabled)
        if self.wandb.run is None:
            return
        # This registers 'epoch' as a distinct metric in WandB
        self.wandb.define_metric("epoch")

    def after_val_epoch(self, runner, metrics: dict[str, float] | None = None) -> None:
        """
        Called after the validation epoch.
        Logs the current epoch number.
        """
        super().after_val_epoch(runner, metrics)

        # only log if a run is active (this is true on rank 0 when WandbVisBackend is enabled)
        if self.wandb.run is None:
            return
        # MMDetection epochs are 1-based, but we might want to log at epoch 0 for initial
        # validation. runner.epoch is the current *completed* epoch count.
        current_epoch = runner.epoch
        self.wandb.log({"epoch": current_epoch}, step=runner.iter)
