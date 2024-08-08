import torch
from mmdet.apis.det_inferencer import DetInferencer
from copy import deepcopy

model_60_class = DetInferencer("projects/ViTDet_67_class/vit_L_gsd_det_rcnn_xview_67.py", weights="~/models/vit-L-gsd_det-rcnn-xview_67/epoch_100.pth", device="cpu")
model_67_class = DetInferencer("projects/ViTDet_67_class/vit_L_gsd_det_rcnn_xview_67.py", device="cpu")

sd = model_60_class.model.state_dict()
new_sd = deepcopy(sd)
# size mismatch for roi_head.bbox_head.fc_cls.weight: copying a param with shape torch.Size([61, 1024]) from checkpoint, the shape in current model is torch.Size([68, 1024]).
# size mismatch for roi_head.bbox_head.fc_cls.bias: copying a param with shape torch.Size([61]) from checkpoint, the shape in current model is torch.Size([68]).
# size mismatch for roi_head.bbox_head.fc_reg.weight: copying a param with shape torch.Size([240, 1024]) from checkpoint, the shape in current model is torch.Size([268, 1024]).
# size mismatch for roi_head.bbox_head.fc_reg.bias: copying a param with shape torch.Size([240]) from checkpoint, the shape in current model is torch.Size([268]).
# size mismatch for roi_head.mask_head.conv_logits.weight: copying a param with shape torch.Size([60, 256, 1, 1]) from checkpoint, the shape in current model is torch.Size([67, 256, 1, 1]).
# size mismatch for roi_head.mask_head.conv_logits.bias: copying a param with shape torch.Size([60]) from checkpoint, the shape in current model is torch.Size([67]).
category_id = -1
CLASSES = ('Fixed-wing Aircraft', 'Small Aircraft', 'Cargo Plane', 'Helicopter', 'Passenger Vehicle', 'Small Car', 'Bus', 'Pickup Truck', 'Utility Truck', 'Truck', 'Cargo Truck', 'Truck w/Box', 'Truck Tractor', 'Trailer', 'Truck w/Flatbed', 'Truck w/Liquid', 'Crane Truck', 'Railway Vehicle', 'Passenger Car', 'Cargo Car', 'Flat Car', 'Tank car', 'Locomotive', 'Maritime Vessel', 'Motorboat', 'Sailboat', 'Tugboat', 'Barge', 'Fishing Vessel', 'Ferry', 'Yacht', 'Container Ship', 'Oil Tanker', 'Engineering Vehicle', 'Tower crane', 'Container Crane', 'Reach Stacker', 'Straddle Carrier', 'Mobile Crane', 'Dump Truck', 'Haul Truck', 'Scraper/Tractor', 'Front loader/Bulldozer', 'Excavator', 'Cement Mixer', 'Ground Grader', 'Hut/Tent', 'Shed', 'Building', 'Aircraft Hangar', 'Damaged Building', 'Facility', 'Construction Site', 'Vehicle Lot', 'Helipad', 'Storage Tank', 'Shipping container lot', 'Shipping Container', 'Pylon', 'Tower', 'Passenger Car1', 'Passenger Car2', 'Passenger Car3', 'Passenger Car4', 'Passenger Car5', 'Passenger Car6', 'Passenger Car7')
category_id = 5


fc_cls_weight = sd["roi_head.bbox_head.fc_cls.weight"]
fc_cls_bias = sd["roi_head.bbox_head.fc_cls.bias"]
car_weights = fc_cls_weight[category_id:category_id+1]
car_bias = fc_cls_bias[category_id:category_id+1]

fc_reg_weight = sd["roi_head.bbox_head.fc_reg.weight"]
fc_reg_bias = sd["roi_head.bbox_head.fc_reg.bias"]
new_fc_reg_weight = fc_reg_weight[category_id*4:(category_id+1)*4].repeat(268 // 4,1)
new_fc_reg_bias = fc_reg_bias[category_id*4:(category_id+1)*4].repeat(268 // 4)
new_sd["roi_head.bbox_head.fc_reg.weight"] = new_fc_reg_weight
new_sd["roi_head.bbox_head.fc_reg.bias"] = new_fc_reg_bias

mask_head_weight = sd["roi_head.mask_head.conv_logits.weight"]
mask_head_bias = sd["roi_head.mask_head.conv_logits.bias"]
new_mask_head_weight = mask_head_weight[category_id:category_id+1].repeat(67,1,1,1)
new_mask_head_bias = mask_head_bias[category_id:category_id+1].repeat(67)
new_sd["roi_head.mask_head.conv_logits.weight"] = new_mask_head_weight
new_sd["roi_head.mask_head.conv_logits.bias"] = new_mask_head_bias

model_67_class.model.load_state_dict(new_sd)
torch.save(model_67_class.model.state_dict(), "/home/mlavery/models/vit-L-gsd_det-rcnn-xview_67/epoch_100_Passenger_Car.pth")