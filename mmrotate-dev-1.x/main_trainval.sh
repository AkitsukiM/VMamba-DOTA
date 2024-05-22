model[1]="oriented-rcnn-le90_vmamba-base_fpn_1x_dota"
model[2]="oriented-rcnn-le90_vmamba-small_fpn_1x_dota"
model[3]="oriented-rcnn-le90_vmamba-tiny_fpn_1x_dota"

for model_i in ${model[*]}
do
    # echo "./configs/_oriented_rcnn_/${model_i}.py"
    CUDA_VISIBLE_DEVICES=3,4,5,6 ./tools/dist_train.sh ./configs/_oriented_rcnn_/${model_i}.py 4
    CUDA_VISIBLE_DEVICES=3,4,5,6 ./tools/dist_test.sh "./configs/_oriented_rcnn_/${model_i}.py" "./work_dirs/${model_i}/epoch_12.pth" 4
    python "../DOTA_devkit-master/dota_evaluation_task1.py" --mergedir "./work_dirs/Task1_${model_i}/" --imagesetdir "./data/DOTA/val/" --use_07_metric True
    # epoch=1
    # while [ $epoch -le 11 ]
    # do
    #     # echo "./work_dirs/${model_i}/epoch_${epoch}.pth"
    #     rm "./work_dirs/${model_i}/epoch_${epoch}.pth"
    #     epoch=`expr ${epoch} + 1`
    # done
done
