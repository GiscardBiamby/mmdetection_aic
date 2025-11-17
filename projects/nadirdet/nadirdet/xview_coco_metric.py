import itertools
import os.path as osp
import tempfile
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence
import numpy as np
from mmdet.datasets.api_wrappers import COCO, COCOevalMP
from mmdet.evaluation import CocoMetric
from mmdet.registry import METRICS
from mmengine.evaluator import BaseMetric
from mmengine.fileio import dump, get_local_path, load
from mmengine.hooks import checkpoint_hook
from mmengine.logging import MMLogger
from pycocotools.cocoeval import COCOeval, StatKey, StatKeyPerClass
from terminaltables import AsciiTable


@METRICS.register_module()
class XViewCocoMetric(CocoMetric):
    """COCO evaluation metric.

    Evaluate AR, AP, and mAP for detection tasks including proposal/box
    detection and instance segmentation. Please refer to
    https://cocodataset.org/#detection-eval for more details.

    Args:
        ann_file (str, optional): Path to the coco format annotation file.
            If not specified, ground truth annotations from the dataset will
            be converted to coco format. Defaults to None.
        metric (str | List[str]): Metrics to be evaluated. Valid metrics
            include 'bbox', 'segm', 'proposal', and 'proposal_fast'.
            Defaults to 'bbox'.
        classwise (bool): Whether to evaluate the metric class-wise.
            Defaults to False. Setting this to True is unsupported for xview because the cocobetter COCOEval already computes per-class metrics.
        max_dets (Sequence[int]): Numbers of proposals to be evaluated.
            Defaults to (500, 1000, 10000).
        iou_thrs (float | List[float], optional): IoU threshold to compute AP
            and AR. If not specified, IoUs from 0.25 to 0.95 will be used.
            Defaults to None.
        area_range_labels (): List[str] | str, optional): Area range labels for the values in `area_ranges`. 
        area_ranges (): List[str] | str, optional): Area ranges for evaluating AR and AP.
        metric_items (List[str], optional): Metric result names to be
            recorded in the evaluation result. Defaults to None. Only used in the proposal eval code path (so not used for COCO eval of xview detections/seg).
        format_only (bool): Format the output results without perform
            evaluation. It is useful when you want to format the result
            to a specific format and submit it to the test server.
            Defaults to False.
        outfile_prefix (str, optional): The prefix of json files. It includes
            the file path and the prefix of filename, e.g., "a/b/prefix".
            If not specified, a temp file will be created. Defaults to None.
        file_client_args (dict, optional): Arguments to instantiate the
            corresponding backend in mmdet <= 3.0.0rc6. Defaults to None.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
        sort_categories (bool): Whether sort categories in annotations. Only
            used for `Objects365V1Dataset`. Defaults to False.
        use_mp_eval (bool): Whether to use mul-processing evaluation
    """

    default_prefix: str | None = ""

    def __init__(
        self,
        ann_file: str | None = None,
        metric: str | list[str] = "bbox",
        classwise: bool = False,
        # TODO: Check un-chipped xview to see max objects per un-chipped image
        max_dets: Sequence[int] = (500, 1000, 10000),
        summary_ious: float | Sequence[float] | None = [0.25, 0.50, 0.75],
        iou_thrs: float | Sequence[float] | None = None,
        area_range_labels: str | Sequence[str] | None = ["all", "small", "medium", "large"],
        area_ranges: list[int] | list[list[int]] | None = None,
        metric_items: Sequence[str] | None = None,
        format_only: bool = False,
        outfile_prefix: str | None = None,
        file_client_args: dict | None = None,
        backend_args: dict | None = None,
        collect_device: str = "cpu",
        prefix: str | None = None,
        sort_categories: bool = False,
        use_mp_eval: bool = False,
    ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        # coco evaluation metrics
        self.metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ["bbox", "segm", "proposal", "proposal_fast"]
        for metric in self.metrics:
            if metric not in allowed_metrics:
                raise KeyError(
                    "metric should be one of 'bbox', 'segm', 'proposal', "
                    f"'proposal_fast', but got {metric}."
                )

        # do class wise evaluation, default False
        self.classwise = classwise
        # whether to use multi processing evaluation, default False
        self.use_mp_eval = use_mp_eval

        # max_dets used to compute recall or precision.
        self.max_dets = list(max_dets)
        self.summary_ious = summary_ious
        self.area_range_labels = area_range_labels

        if area_ranges is None:
            area_ranges = [
                [0**2, 1e5**2],
                [0**2, 32**2],
                [32**2, 96**2],
                [96**2, 1e5**2],
            ]
        self.area_ranges = area_ranges
        # iou_thrs used to compute recall or precision.
        if iou_thrs is None:
            iou_thrs = np.linspace(
                0.25, 0.95, int(np.round((0.95 - 0.25) / 0.05)) + 1, endpoint=True
            )
        self.iou_thrs = iou_thrs
        self.metric_items = metric_items
        self.format_only = format_only
        if self.format_only:
            assert outfile_prefix is not None, "outfile_prefix must be not"
            "None when format_only is True, otherwise the result files will"
            "be saved to a temp directory which will be cleaned up at the end."

        self.outfile_prefix = outfile_prefix

        self.backend_args = backend_args
        if file_client_args is not None:
            raise RuntimeError(
                "The `file_client_args` is deprecated, "
                "please use `backend_args` instead, please refer to"
                "https://github.com/open-mmlab/mmdetection/blob/main/configs/_base_/datasets/coco_detection.py"  # noqa: E501
            )

        # if ann_file is not specified,
        # initialize coco api with the converted dataset
        if ann_file is not None:
            with get_local_path(ann_file, backend_args=self.backend_args) as local_path:
                self._coco_api = COCO(local_path)
                if sort_categories:
                    # 'categories' list in objects365_train.json and
                    # objects365_val.json is inconsistent, need sort
                    # list(or dict) before get cat_ids.
                    cats = self._coco_api.cats
                    sorted_cats = {i: cats[i] for i in sorted(cats)}
                    self._coco_api.cats = sorted_cats
                    categories = self._coco_api.dataset["categories"]
                    sorted_categories = sorted(categories, key=lambda i: i["id"])
                    self._coco_api.dataset["categories"] = sorted_categories
        else:
            self._coco_api = None

        # handle dataset lazy init
        self.cat_ids = None
        self.img_ids = None

    def compute_metrics(self, results: list) -> dict[str, float]:
        """
        Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        # split gt and prediction list
        gts, preds = zip(*results)

        tmp_dir = None
        if self.outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            outfile_prefix = osp.join(tmp_dir.name, "results")
        else:
            outfile_prefix = self.outfile_prefix

        if self._coco_api is None:
            # use converted gt json file to initialize coco api
            logger.info("Converting ground truth to coco format...")
            coco_json_path = self.gt_to_coco_json(gt_dicts=gts, outfile_prefix=outfile_prefix)
            self._coco_api = COCO(coco_json_path)

        # handle lazy init
        if self.cat_ids is None:
            self.cat_ids = self._coco_api.get_cat_ids(cat_names=self.dataset_meta["classes"])
        if self.img_ids is None:
            self.img_ids = self._coco_api.get_img_ids()

        # convert predictions to coco format and dump to json file
        result_files = self.results2json(preds, outfile_prefix)

        eval_results = OrderedDict()
        if self.format_only:
            logger.info(f"results are saved in {osp.dirname(outfile_prefix)}")
            return eval_results

        for metric in self.metrics:
            logger.info(f"Evaluating {metric}...")

            # TODO: May refactor fast_eval_recall to an independent metric?
            # fast eval recall
            if metric == "proposal_fast":
                ar = self.fast_eval_recall(preds, self.max_dets, self.iou_thrs, logger=logger)
                log_msg = []
                for i, num in enumerate(self.max_dets):
                    eval_results[f"AR@{num}"] = ar[i]
                    log_msg.append(f"\nAR@{num}\t{ar[i]:.4f}")
                log_msg = "".join(log_msg)
                logger.info(log_msg)
                continue

            # evaluate proposal, bbox and segm
            iou_type = "bbox" if metric == "proposal" else metric
            if metric not in result_files:
                raise KeyError(f"{metric} is not in results")
            try:
                predictions = load(result_files[metric])
                if iou_type == "segm":
                    # Refer to https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py#L331  # noqa
                    # When evaluating mask AP, if the results contain bbox,
                    # cocoapi will use the box area instead of the mask area
                    # for calculating the instance area. Though the overall AP
                    # is not affected, this leads to different
                    # small/medium/large mask AP results.
                    for x in predictions:
                        x.pop("bbox")
                coco_dt = self._coco_api.loadRes(predictions)

            except IndexError:
                logger.error("The testing results of the whole dataset is empty.")
                break

            if self.use_mp_eval:
                raise NotImplementedError("XViewCocoMetric does not support use_mp_eval=True")
                coco_eval = COCOevalMP(self._coco_api, coco_dt, iou_type)
            else:
                coco_eval = COCOeval(self._coco_api, coco_dt, iou_type)

            coco_eval.params.catIds = self.cat_ids
            coco_eval.params.imgIds = self.img_ids
            coco_eval.params.maxDets = list(self.max_dets)
            coco_eval.params.iouThrs = self.iou_thrs
            coco_eval.params.summaryIous = self.summary_ious
            coco_eval.params.areaRngLbl = self.area_range_labels
            coco_eval.params.areaRng = self.area_ranges

            if metric == "proposal":
                raise NotImplementedError(
                    "Custom XView coco metric does not support 'proposal' metric"
                )
                coco_eval.params.useCats = 0
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
                metric_items = self.metric_items
                if metric_items is None:
                    metric_items = [
                        "AR@100",
                        "AR@300",
                        "AR@1000",
                        "AR_s@1000",
                        "AR_m@1000",
                        "AR_l@1000",
                    ]

                for item in metric_items:
                    val = float(f"{coco_eval.stats[coco_metric_names[item]]:.3f}")
                    eval_results[item] = val
            else:
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
                if self.classwise:  # Compute per-category AP
                    raise NotImplementedError("XViewCocoMetric classwise=True is untested")
                    # Compute per-category AP
                    # from https://github.com/facebookresearch/detectron2/
                    precisions = coco_eval.eval["precision"]
                    # precision: (iou, recall, cls, area range, max dets)
                    assert len(self.cat_ids) == precisions.shape[2]

                    results_per_category = []
                    for idx, cat_id in enumerate(self.cat_ids):
                        t = []
                        # area range index 0: all area ranges
                        # max dets index -1: typically 100 per image
                        nm = self._coco_api.loadCats(cat_id)[0]
                        precision = precisions[:, :, idx, 0, -1]
                        precision = precision[precision > -1]
                        if precision.size:
                            ap = np.mean(precision)
                        else:
                            ap = float("nan")
                        t.append(f"{nm['name']}")
                        t.append(f"{round(ap, 3)}")
                        eval_results[f"{nm['name']}_precision"] = round(ap, 3)

                        # indexes of IoU  @50 and @75
                        for iou in [0, 5]:
                            precision = precisions[iou, :, idx, 0, -1]
                            precision = precision[precision > -1]
                            if precision.size:
                                ap = np.mean(precision)
                            else:
                                ap = float("nan")
                            t.append(f"{round(ap, 3)}")

                        # indexes of area of small, median and large
                        for area in [1, 2, 3]:
                            precision = precisions[:, :, idx, area, -1]
                            precision = precision[precision > -1]
                            if precision.size:
                                ap = np.mean(precision)
                            else:
                                ap = float("nan")
                            t.append(f"{round(ap, 3)}")
                        results_per_category.append(tuple(t))

                    num_columns = len(results_per_category[0])
                    results_flatten = list(itertools.chain(*results_per_category))
                    headers = ["category", "mAP", "mAP_50", "mAP_75", "mAP_s", "mAP_m", "mAP_l"]
                    results_2d = itertools.zip_longest(
                        *[results_flatten[i::num_columns] for i in range(num_columns)]
                    )
                    table_data = [headers]
                    table_data += [result for result in results_2d]
                    table = AsciiTable(table_data)
                    logger.info("\n" + table.table)

                for k, v in coco_eval.stats_dict.items():
                    key = f"metric_{k.metric}-iou_{k.iou}-area_{k.area}-max_dets={k.max_dets}"
                    prefix = "xview_coco_" + ("precision" if k.metric == "AP" else "@AR")
                    key = prefix + "/" + key
                    eval_results[key] = float(f"{round(v, 3)}")
                    logger.info(f"  {key}: {round(v, 3)}")
                for k, v in coco_eval.stats_dict_per_class.items():
                    key = f"metric_{k.metric}-iou_{k.iou}-area_{k.area}-max_dets={k.max_dets}-cat_id_{k.cat_id}-name={k.name}"
                    prefix = "xview_coco_" + ("precision" if k.metric == "AP" else "@AR")
                    prefix += "_per_class"
                    key = prefix + "/" + key
                    eval_results[key] = float(f"{round(v, 3)}")

                MAP_IOU = "{:0.2f}:{:0.2f}".format(self.iou_thrs[0], self.iou_thrs[-1])
                copy_paste_metrics = [StatKey("AP", MAP_IOU, "all", self.max_dets[-1])]
                for iou in self.summary_ious:
                    copy_paste_metrics.append(
                        StatKey("AP", f"{iou:0.2f}", "all", self.max_dets[-1])
                    )
                for area_label in self.area_range_labels:
                    copy_paste_metrics.append(StatKey("AP", MAP_IOU, area_label, self.max_dets[-1]))

                ap = [coco_eval.stats_dict[sk] for sk in copy_paste_metrics]
                logger.info(
                    f"{metric}_mAP_copypaste: {ap[0]:.3f} "
                    f"{ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} "
                    f"{ap[4]:.3f} {ap[5]:.3f}"
                )

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results
