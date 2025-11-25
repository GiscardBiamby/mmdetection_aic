import argparse
import projects.nadirdet.nadirdet  # Register modules  # noqa: F401
import torch
from mmdet.apis import inference_detector, init_detector
from mmdet.registry import MODELS
from projects.nadirdet.nadirdet.models.heads.geo_pose_head import GeoPoseHead


def main():
    parser = argparse.ArgumentParser(description="Inference for FasterRCNNGeoPose")
    parser.add_argument("config", help="Path to config file")
    parser.add_argument("checkpoint", help="Path to checkpoint file")
    parser.add_argument("img", help="Path to image file")
    parser.add_argument("--device", default="cuda:0", help="Device used for inference")
    args = parser.parse_args()

    # Build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)

    # Test a single image
    result = inference_detector(model, args.img)

    # result is a DetDataSample (or list of them if batch inference, but inference_detector usually
    # returns one for one image) However, inference_detector might return a list if input is a list.
    # Here input is single string.

    if isinstance(result, list):
        result = result[0]

    # Extract geopose prediction
    if hasattr(result, "pred_geo_pose"):
        pred_geo_pose = result.pred_geo_pose
        print("\nRaw Predicted GeoPose Vector:")
        print(pred_geo_pose)

        # Unnormalize
        readable_pose = GeoPoseHead.unnormalize(pred_geo_pose)
        print("\nPredicted Geometric Properties:")
        for k, v in readable_pose.items():
            print(f"{k}: {v}")
    else:
        print(
            "No 'pred_geo_pose' found in result. Make sure the model is FasterRCNNGeoPose and weights are loaded."
        )


if __name__ == "__main__":
    main()
