### INTSTRUCTIONS:
# make datadir in data and symlink coco dataset folder inside second data folder
# data -> data -> coco (sym link)



cd ../data
echo "moving to data dir."
pwd
echo "starting training"
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,8 ../mmdetection/tools/dist_train.sh ../mmdetection/projects/ViTDINO-FSDP/ViTDINO-conf.py \
    8