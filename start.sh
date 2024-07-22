### INTSTRUCTIONS:
# make datadir in data and symlink coco dataset folder inside second data folder
# data -> data -> coco (sym link)



cd ../data
echo "moving to data dir."
pwd
echo "starting training"
# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,8 ../mmdetection/tools/dist_train.sh ../mmdetection/configs/dino/dino-4scale_r50_8xb2-12e_coco.py \
#    8
# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,8 ../mmdetection/tools/dist_train.sh ../mmdetection/projects/ViTDINO-FSDP/ViTDINO-conf.py \
#      8
# ../mmdetection/tools/dist_train.sh ../small-object-detection-benchmark/mmdet_configs/tood/tood_r50_fpn_1x_coco.py \
#      8
# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,8 ../mmdetection/tools/dist_train.sh ../mmdetection/projects/CO-DETR/configs/codino/scale_co_dino.py \
#      8
# ../mmdetection/tools/dist_train.sh /home/mlavery/scalemae_docker/mmdetection/projects/ViTDet/configs/vitdet_yolox.py \
#      5
# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,8 ../mmdetection/tools/dist_train.sh /home/mlavery/scalemae_docker/mmdetection/projects/ViTDet/configs/vitdet_mask-rcnn_vit-b-mae_lsj-100e.py \
#      8
# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,8 ../mmdetection/tools/dist_train.sh /home/mlavery/scalemae_docker/mmdetection/projects/ViTDet/configs/vitdet_rcnn_coco.py \
#      8
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,8 ../mmdetection/tools/dist_train.sh /home/mlavery/scalemae_docker/mmdetection/projects/ViTDet/configs/vitdet_rcnn_xview.py \
     8
