model[1]="rotated_retinanet_obb_swin_tiny_fpn_1x_dota_le90"
model[2]="rotated_retinanet_obb_swin_small_fpn_1x_dota_le90"
model[3]="rotated_retinanet_obb_swin_base_fpn_1x_dota_le90"
model[4]="rotated_retinanet_obb_vmamba_tiny_fpn_1x_dota_le90"
model[5]="rotated_retinanet_obb_vmamba_small_fpn_1x_dota_le90"
model[6]="rotated_retinanet_obb_vmamba_base_fpn_1x_dota_le90"
model[7]="rotated_faster_rcnn_swin_tiny_fpn_1x_dota_le90"
model[8]="rotated_faster_rcnn_swin_small_fpn_1x_dota_le90"
model[9]="rotated_faster_rcnn_swin_base_fpn_1x_dota_le90"
model[10]="rotated_faster_rcnn_vmamba_tiny_fpn_1x_dota_le90"
model[11]="rotated_faster_rcnn_vmamba_small_fpn_1x_dota_le90"
model[12]="rotated_faster_rcnn_vmamba_base_fpn_1x_dota_le90"

for model_i in ${model[*]}
do
    # echo "./configs/_rotated_faster_rcnn_/${model_i}.py"
    # CUDA_VISIBLE_DEVICES=3,4,5,6 ./tools/dist_train.sh ./configs/_rotated_retinanet_/${model_i}.py 4
    # CUDA_VISIBLE_DEVICES=3,4,5,6 ./tools/dist_test.sh "./configs/_rotated_retinanet_/${model_i}.py" "./work_dirs/${model_i}/epoch_12.pth" 4 --format-only --eval-options submission_dir="./work_dirs/Task1_${model_i}_epoch_12/"
    CUDA_VISIBLE_DEVICES=3,4,5,6 ./tools/dist_train.sh ./configs/_rotated_faster_rcnn_/${model_i}.py 4
    CUDA_VISIBLE_DEVICES=3,4,5,6 ./tools/dist_test.sh "./configs/_rotated_faster_rcnn_/${model_i}.py" "./work_dirs/${model_i}/epoch_12.pth" 4 --format-only --eval-options submission_dir="./work_dirs/Task1_${model_i}_epoch_12/"
    python "../DOTA_devkit-master/dota_evaluation_task1.py" --mergedir "./work_dirs/Task1_${model_i}_epoch_12/" --imagesetdir "./data/DOTA/val/" --use_07_metric True
    # epoch=1
    # while [ $epoch -le 11 ]
    # do
    #     # echo "./work_dirs/${model_i}/epoch_${epoch}.pth"
    #     rm "./work_dirs/${model_i}/epoch_${epoch}.pth"
    #     epoch=`expr ${epoch} + 1`
    # done
done
