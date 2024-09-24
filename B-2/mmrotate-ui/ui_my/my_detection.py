from mmdet.apis import init_detector, inference_detector
from mmrotate.apis.inference import inference_detector_by_patches

from ui_my.transforms_rotated import poly_to_rotated_box_np
from ui_my.my_image import imshow_det_rbboxes

import os
import cv2
import numpy as np
import mmcv
import torch


dirname = os.path.dirname(cv2.__file__)
plugin_path = os.path.join(dirname, 'plugins', 'platforms')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path


CLASSES = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
           'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
           'basketball-court', 'storage-tank', 'soccer-ball-field',
           'roundabout', 'harbor', 'swimming-pool', 'helicopter')

PALETTE = [(165, 42, 42), (189, 183, 107), (0, 255, 0), (255, 0, 0),
           (138, 43, 226), (255, 128, 0), (255, 0, 255), (0, 255, 255),
           (255, 193, 193), (0, 51, 153), (255, 250, 205), (0, 139, 139),
           (255, 255, 0), (147, 116, 116), (0, 0, 255)]


class Detection(object):
    def __init__(self, method='model_0', score_thr=0.5):
        super(Detection, self).__init__()
        assert method in ['model_0', 'model_1', 'model_2'], 'only support model_0, model_1, model_2'

        if method == 'model_0':
            config_path = './configs/rotated_faster_rcnn/rotated_faster_rcnn_r50_fpn_1x_dota_le90.py'
            checkpoint_path = './work_dirs/rotated_faster_rcnn_r50_fpn_1x_dota_le90/rotated_faster_rcnn_r50_fpn_1x_dota_le90-0393aa5c.pth'
        elif method == 'model_1':
            config_path = ''
            checkpoint_path = ''
        elif method == 'model_2':
            config_path = ''
            checkpoint_path = ''
        else:
            pass
        # self.device = 'cpu'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = init_detector(config_path, checkpoint_path, device=self.device)
        self.classes = self.model.CLASSES

        self.method = method
        self.score_thr = score_thr


    def parse_gt(self, filename):
        """

        :param filename: ground truth file to parse
        :return: all instances in a picture
        """
        # objects = []
        objects_label = []
        objects_bboxs = []
        with  open(filename, 'r') as f:
            while True:
                line = f.readline()
                if line:
                    splitlines = line.strip().split(' ')
                    # object_struct = {}
                    if (len(splitlines) < 9):
                        continue
                    # object_struct['name'] = splitlines[8]
                    object_name = splitlines[8]
                    objects_label.append(self.classes.index(object_name))

                    # if (len(splitlines) == 9):
                    #     object_struct['difficult'] = 0
                    # elif (len(splitlines) == 10):
                    #     object_struct['difficult'] = int(splitlines[9])
                    object_bboxs = [float(splitlines[0]),
                                    float(splitlines[1]),
                                    float(splitlines[2]),
                                    float(splitlines[3]),
                                    float(splitlines[4]),
                                    float(splitlines[5]),
                                    float(splitlines[6]),
                                    float(splitlines[7]),
                                    1.0]
                    objects_bboxs.append(object_bboxs)
                else:
                    break

        labels = np.array(objects_label)
        bboxes_scores = np.array(objects_bboxs)
        scores = bboxes_scores[:, -1:]
        bboxes = bboxes_scores[:, :-1]
        bboxes = poly_to_rotated_box_np(bboxes)
        bboxes = np.hstack((bboxes, scores))
        return labels, bboxes


    def gt_img(self, img_file, show=False, out_file=None):
        img = mmcv.imread(img_file)
        labelTxt_path = img_file[:-4] + ".txt"
        if not os.path.exists(labelTxt_path): return img
        labels, bboxes = self.parse_gt(labelTxt_path)

        # print(img.shape)
        # print(bboxes)
        print("gt:", labels)
        # print(self.classes)

        final_img = imshow_det_rbboxes(
            img=img,
            bboxes=bboxes,
            labels=labels,
            segms=None,
            class_names=CLASSES,
            score_thr=self.score_thr,
            bbox_color='green',
            text_color=None,
            mask_color=None,
            thickness=8,
            font_size=0,
            win_name='',
            show=show,
            wait_time=0,
            out_file=out_file)

        return final_img


    def predict_img(self, img_file, show=False, out_file=None):
        img = mmcv.imread(img_file)
        bbox_result = inference_detector_by_patches(
            model = self.model,
            img = img_file,
            sizes = [1024],
            steps = [824],
            ratios = [1.0],
            merge_iou_thr = 0.1,
            bs=1)

        bboxes = np.vstack(bbox_result)
        labels = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(bbox_result)]
        labels = np.concatenate(labels)   

        # print(img.shape)
        # print(bboxes)
        print("pred:", labels)
        # print(self.classes)

        final_img = imshow_det_rbboxes(
            img=img,
            bboxes=bboxes,
            labels=labels,
            segms=None,
            class_names=CLASSES,
            score_thr=self.score_thr,
            bbox_color=PALETTE,
            text_color=None,
            mask_color=None,
            thickness=8,
            font_size=0,
            win_name='',
            show=show,
            wait_time=0,
            out_file=out_file)

        return final_img


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_file", type = str, default = "data_test/aaaa.png", help = "img file")
    parser.add_argument("--method", type = str, default = "model_0", help = "detection method")
    parser.add_argument("--show", type = bool, default = False, help = "show result img or not")
    parser.add_argument("--out_file", type = str, default = "data_test/aaaa_out.png", help = "out file")
    opt = parser.parse_args()

    detect_model = Detection(method=opt.method)
    result = detect_model.predict_img(img_file=opt.img_file, show=opt.show, out_file=opt.out_file)

