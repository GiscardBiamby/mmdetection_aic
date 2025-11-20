import math
import torch
import torch.distributed as dist
from mmengine.hooks import Hook
from mmengine.logging import MMLogger
from mmengine.registry import HOOKS


@HOOKS.register_module()
class SqrtLRScalingHook(Hook):
    def __init__(self, base_batch_size):
        self.base_batch_size = base_batch_size

    def before_run(self, runner):
        """
        Executes before the training loop starts.
        Calculates the global batch size and scales the LR.
        """
        # 1. Get Local Batch Size (Samples per GPU)
        # runner.train_dataloader is already built by this stage
        samples_per_gpu = None
        if (
            hasattr(runner.train_dataloader, "batch_size")
            and runner.train_dataloader.batch_size is not None
        ):
            samples_per_gpu = runner.train_dataloader.batch_size
        elif hasattr(runner.train_dataloader, "batch_sampler") and hasattr(
            runner.train_dataloader.batch_sampler, "batch_size"
        ):
            samples_per_gpu = runner.train_dataloader.batch_sampler.batch_size

        if samples_per_gpu is None:
            # Fallback or error
            raise ValueError(
                "Could not determine batch_size from train_dataloader. "
                "Please ensure batch_size is set or batch_sampler has batch_size."
            )

        # 2. Determine World Size (Number of GPUs)
        # This logic handles both Distributed (DDP) and Single-GPU (standard) modes.
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            mode = "Distributed (DDP)"
        else:
            # If running tools/train.py on 1 GPU without launcher, this block runs.
            world_size = 1
            mode = "Single-GPU"

        global_batch_size = samples_per_gpu * world_size

        # 3. Calculate Scaling Factor (Square Root)
        if self.base_batch_size is None:
            return

        scaling_factor = math.sqrt(global_batch_size / self.base_batch_size)

        # 4. Scale the Learning Rate and Log
        logger = MMLogger.get_current_instance()
        optimizer = runner.optim_wrapper.optimizer

        logger.info("=" * 40)
        logger.info(f"[SqrtLRScalingHook] Mode: {mode}")
        logger.info(f"Base Batch Size:   {self.base_batch_size}")
        logger.info(
            f"Current Global BS: {global_batch_size} "
            f"({samples_per_gpu} img/gpu * {world_size} gpus)"
        )
        logger.info(
            f"Scaling Factor: {scaling_factor:.4f} "
            f"(sqrt({global_batch_size}/{self.base_batch_size}))"
        )

        # Loop through ALL param_groups (Backbone, Neck, Head might be separate)
        for i, group in enumerate(optimizer.param_groups):
            old_lr = group["lr"]
            new_lr = old_lr * scaling_factor
            group["lr"] = new_lr

            # Log every group to be safe
            logger.info(f" > Param Group {i}: LR scaled {old_lr:.2e} -> {new_lr:.2e}")

        logger.info("=" * 40)
