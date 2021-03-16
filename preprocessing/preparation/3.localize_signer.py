import glob
import os
import sys
from argparse import ArgumentParser

import cv2
import matplotlib.pyplot as plt
import mmcv
import numpy as np
import torch
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from torchvision.transforms import transforms
from tqdm import tqdm

sys.path.insert(1, os.path.join(sys.path[0], 'preparation'))
from Pose_HRNet.SimpleHRNet import SimpleHRNet2


class HRNetProcess():
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def trans(self, img):
        img = cv2.resize(img, (288, 384), interpolation=cv2.INTER_CUBIC)
        img = self.transform(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img = torch.stack([img], dim=0)
        return img


def visualize(img, result, pts, save):
    img = cv2.rectangle(img, tuple(result[:2]), tuple(result[2:]), (0, 0, 255), thickness=3)
    for i in range(pts.shape[0]):
        pt = pts[i]*511
        if i in [11, 12]:
            color = (0, 0, 255)
        else:
            color = (0, 125, 125) 
        img = cv2.circle(img, center=(int(pt[1]), int(pt[0])), radius=2, color=(0, 0, 255), thickness=-1)
    cv2.imwrite(save, img)


def get_box(base, task, model):
    with open(f'preparation/pre_data/file_{task}.txt', 'r') as f:
        data = f.readlines()
        data = [x.split(',')[0] for x in data]
    boxes = []
    f = lambda x: min(max(0,x), 511)
    ptprocess = HRNetProcess()
    for name in tqdm(data):
        img = cv2.imread(os.path.join(base, task, name+'_color.jpg'))
        result = inference_detector(model[0], img)[0][0]

        pt = model[1].predict(ptprocess.trans(img[int(result[1]): int(result[3]), int(result[0]): int(result[2])]))[0]
        pt[:, 0] = (pt[:, 0]*(result[3]- result[1])+result[1])/511
        pt[:, 1] = (pt[:, 1]*(result[2]- result[0])+result[0])/511
        bottom = f(int((max(pt[11, 0], pt[12, 0])+0.7*abs(pt[5, 1]-pt[6, 1]))*511))

        half = int(min(((bottom-result[1]+18)//2), bottom//2))
        if half <= 120:
            print(name, half)
        mid = int((result[0]+result[2])//2)
        mid = max(0, mid-half)+half
        mid = min(511, mid+half)-half
        result = [f(mid-half), f(bottom-2 * half), f(mid+half), bottom]
        assert result[2]-result[0] == result[3]-result[1]
        visualize(img, result, pt, os.path.join(base, 'result', task, name+'_color.jpg'))
        boxes.append([name, result])  
    with open(f'preparation/pre_data/box_{task}.txt', 'w') as f:
        for box in boxes:
            f.write('{:s},{:d},{:d},{:d},{:d}\n'.format(box[0], box[1][0], box[1][1], box[1][2], box[1][3]))

def main():
    parser = ArgumentParser()
    parser.add_argument('--config', help='Config file', default='/opt/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_2x_coco.py')
    parser.add_argument('--checkpoint', help='Checkpoint file', default='preparation/pre_data/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()

    # test_img = 'preparation/pre_data/signer35_sample9_color.jpg'

    # build the model from a config file and a checkpoint file
    model = [
        init_detector(args.config, args.checkpoint, device=args.device),
        SimpleHRNet2(48, 17, "preparation/Pose_HRNet/weights/pose_hrnet_w48_384x288.pth",
                     multiperson=False, max_batch_size=300, device=args.device),
    ]

    # get_box('data/AUTSL/first', 'val', model)
    # get_box('data/AUTSL/first', 'train', model)
    get_box('data/AUTSL/first', 'test', model)

if __name__ == '__main__':
    main()
