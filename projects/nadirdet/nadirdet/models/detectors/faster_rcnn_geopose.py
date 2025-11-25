import copy
import numpy as np
import torch
from mmdet.models.detectors import FasterRCNN
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from torch import Tensor


@MODELS.register_module()
class FasterRCNNGeoPose(FasterRCNN):
    def __init__(self, geo_pose_head, **kwargs):
        """Initialize FasterRCNNGeoPose.

        Args:
            geo_pose_head (dict): Config dict for geo pose head.
            **kwargs: Other arguments for FasterRCNN.
        """
        super().__init__(**kwargs)
        self.geo_pose_head = MODELS.build(geo_pose_head)

    def _forward(self, batch_inputs: Tensor, batch_data_samples: SampleList) -> tuple:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.

        Returns:
            tuple: A tuple of features from ``rpn_head`` and ``roi_head``
            forward.
        """
        results = ()
        x = self.extract_feat(batch_inputs)

        if self.with_rpn:
            rpn_results_list = self.rpn_head.predict(x, batch_data_samples, rescale=False)
        else:
            assert batch_data_samples[0].get("proposals", None) is not None
            rpn_results_list = [data_sample.proposals for data_sample in batch_data_samples]
        roi_outs = self.roi_head.forward(x, rpn_results_list, batch_data_samples)

        # GeoPose forward
        geo_preds = self.geo_pose_head(x)

        results = results + (roi_outs, geo_preds)
        return results

    def loss(self, batch_inputs: Tensor, batch_data_samples: SampleList) -> dict | list:  # type: ignore
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """
        # Extract features ONCE
        x = self.extract_feat(batch_inputs)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get("rpn_proposal", self.test_cfg.rpn)
            rpn_data_samples = copy.deepcopy(batch_data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = torch.zeros_like(data_sample.gt_instances.labels)

            rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
                x, rpn_data_samples, proposal_cfg=proposal_cfg
            )
            # avoid get same name with roi_head loss
            losses.update(rpn_losses)
        else:
            # Copy from TwoStageDetector.loss() if need to implement
            raise NotImplementedError("RPN is required for FasterRCNNGeoPose.")

        # ROI forward and loss
        roi_losses = self.roi_head.loss(x, rpn_results_list, batch_data_samples)
        losses.update(roi_losses)

        # Run Geo Pose Head
        # Forward geo head
        geo_preds = self.geo_pose_head(x)

        # Get targets
        # batch_data_samples is a list of DetDataSample
        # We need to extract 'gt_geo_pose' from them.
        # In MMDetection 3.x, custom data is usually in `metainfo` or `gt_instances`?
        # Wait, `LoadGeoPose` adds `gt_geo_pose` to `results`.
        # `PackDetInputs` packs `results` into `data_sample`.
        # We need to make sure `gt_geo_pose` is packed.
        # We might need to modify `PackDetInputs` or use a custom one,
        # OR just access it if `PackDetInputs` puts unknown keys into `metainfo` or similar.

        # Usually `PackDetInputs` puts keys in `meta_keys` into `metainfo`.
        # We need to update the config to include `gt_geo_pose` in `meta_keys` of `PackDetInputs`.

        gt_geo_poses = []
        for data_sample in batch_data_samples:
            # Assuming it's in metainfo (if we configure PackDetInputs correctly)
            # Or maybe we need to check where it ends up.
            # If we add it to `results` in transform, and add it to `meta_keys` in `PackDetInputs`,
            # it will be in `data_sample.metainfo`.
            if hasattr(data_sample, "gt_geo_pose"):
                gt_geo_poses.append(data_sample.gt_geo_pose)
            elif "gt_geo_pose" in data_sample.metainfo:
                gt_geo_poses.append(data_sample.metainfo["gt_geo_pose"])
            else:
                # Debug info
                # available_keys = list(data_sample.metainfo.keys())
                # print(f"Missing gt_geo_pose. Available keys: {available_keys}")
                raise ValueError(
                    f"Missing 'gt_geo_pose' for sample {data_sample.metainfo.get('img_path', 'unknown')}"
                )

        if not gt_geo_poses:
            raise ValueError("No geopose ground truth found in batch.")

        # Convert to tensor if numpy (LoadGeoPose returns numpy array)
        gt_geo_poses = [
            torch.from_numpy(p) if isinstance(p, np.ndarray) else p for p in gt_geo_poses
        ]
        gt_geo_poses = torch.stack(gt_geo_poses).to(geo_preds.device)

        # Calculate loss
        geo_loss = self.geo_pose_head.loss(geo_preds, gt_geo_poses)
        losses.update(geo_loss)

        return losses

    def predict(
        self, batch_inputs: Tensor, batch_data_samples: SampleList, rescale: bool = True
    ) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Return the detection results of the
            input images. The returns value is DetDataSample,
            which usually contain 'pred_instances'. And the
            ``pred_instances`` usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """
        if not self.with_bbox:
            raise ValueError("Bbox head must be implemented.")
        x = self.extract_feat(batch_inputs)

        # If there are no pre-defined proposals, use RPN to get proposals
        if batch_data_samples[0].get("proposals", None) is None:
            rpn_results_list = self.rpn_head.predict(x, batch_data_samples, rescale=False)
        else:
            rpn_results_list = [data_sample.proposals for data_sample in batch_data_samples]

        results_list = self.roi_head.predict(
            x, rpn_results_list, batch_data_samples, rescale=rescale
        )

        batch_data_samples = self.add_pred_to_datasample(batch_data_samples, results_list)

        # Run Geo Pose Head
        geo_preds = self.geo_pose_head(x)

        # Attach to results
        for data_sample, geo_pred in zip(batch_data_samples, geo_preds, strict=True):
            data_sample.pred_geo_pose = geo_pred

        return batch_data_samples
