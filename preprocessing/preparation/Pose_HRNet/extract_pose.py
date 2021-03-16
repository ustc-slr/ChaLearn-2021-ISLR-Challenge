import argparse
import glob
import os
import pickle
import sys

# import imageio
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from tqdm import tqdm

from misc.utils import draw_points, draw_points_and_skeleton, draw_skeleton, joints_dict
from SimpleHRNet import SimpleHRNet2


class VideoDataset(Dataset):
    def __init__(self, clip_info):
        self.list = clip_info
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        frames = []
        paths = glob.glob(self.list[index][1])
        paths.sort()
        for path in paths:
            img = cv2.imread(path)
            img = cv2.resize(img, (288, 384), interpolation=cv2.INTER_CUBIC)
            img = self.transform(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            frames.append(img)
        frames = torch.stack(frames, dim=0)
        return frames


def demo0906(hrnet_c=48, hrnet_j=17, hrnet_weights="./weights/pose_hrnet_w48_384x288.pth",
    hrnet_joints_set="coco", single_person=True,
    max_batch_size=16):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SimpleHRNet2(
        hrnet_c,
        hrnet_j,
        hrnet_weights,
        multiperson=False,
        max_batch_size=300,
        device=device
    )

    base_dir = '/data2/zhouhao/dataset/project/demo0906'
    video_info = []
    for tmp in tqdm(os.listdir(base_dir), leave=False):
        path = os.path.join(base_dir, tmp, '*.jpg')
        num = len(glob.glob(path))
        info = [tmp, path, num]
        video_info.append(info)

    video_info.sort(key=lambda x: x[2], reverse=True)
    print(video_info[0])
    print(len(video_info))
    # video_info = video_info[:5]
    dataset = VideoDataset(video_info)
    loader = DataLoader(dataset, 1, shuffle=False, num_workers=6)
    result = {}
    video_cut = []
    for i, video in enumerate(tqdm(loader)):
        video = video[0]
        assert video.shape[0] == video_info[i][2]
        pt = model.predict(video)
        result[video_info[i][0]] = pt

        info =  video_info[i]
        name = info[0]
        length = info[2]
        init_pos = []
        start = 0
        flag_start = False
        end = length-1
        flag_end = False
        for i in range(length):
            now_pos = [int(pt[i][9][0]*223), int(pt[i][10][0]*223)]
            if i == 0:
                init_pos = [now_pos[0]-20, now_pos[1]-20]
            if (now_pos[0]<init_pos[0] or now_pos[1]<init_pos[1]) and (not flag_start):
                start = max(0, i-6)
                flag_start = True
            if flag_start and flag_end and (now_pos[0]<init_pos[0] or now_pos[1]<init_pos[1]):
                flag_end = False

            if now_pos[0]-5>init_pos[0] and now_pos[1]-5>init_pos[1] and flag_start and (not flag_end):
                end = min(length-1, i+6)
                flag_end = True

        print(name, length, end-start, start, end)
        video_cut.append([name, length, start, end])
    
    pickle.dump(result, open('demo0906.pkl', 'wb'))
    pickle.dump(video_cut, open('demo0906_start.pkl', 'wb'))

    for video in video_info:
        name = video[0]
        print(name)
        imgs = glob.glob(video[1])
        imgs.sort()
        init_pos = []
        for i, img in enumerate(imgs):
            img = cv2.imread(img)
            now_pos = [int(result[name][i][9][0]*223), int(result[name][i][10][0]*223)]
            if i == 0:
                init_pos = [now_pos[0]-20, now_pos[0]-20]
            
            cv2.line(img, (0, init_pos[0]), (223, init_pos[0]), (225, 0, 0))
            cv2.line(img, (0, init_pos[1]), (223, init_pos[1]), (225, 0, 0))
            for j, joint in enumerate(result[name][i]):
                if j==0 or 5<=j<=10:
                    x = int(joint[0]*223)
                    y = int(joint[1]*223)
                    img[x-2:x+2, y-2:y+2, :] = np.array([0, 0, 255])
            cv2.imwrite('result/'+name+str(i)+'.jpg', img)


class VideoDataset_1(Dataset):
    def __init__(self, clip_info):
        self.list = clip_info
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        cap = cv2.VideoCapture(self.list[index][1])
        frames = []
        while(True):
            ret, img = cap.read()
            if ret:
                img = cv2.resize(img, (288, 384), interpolation=cv2.INTER_CUBIC)
                img = self.transform(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                frames.append(img)
            else:
                break
        frames = torch.stack(frames, dim=0)
        return frames


def csl_icme(opts, hrnet_c=48, hrnet_j=17, hrnet_weights="./weights/pose_hrnet_w48_384x288.pth",
    hrnet_joints_set="coco", single_person=True,
    max_batch_size=16,):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SimpleHRNet2(
        hrnet_c,
        hrnet_j,
        hrnet_weights,
        multiperson=False,
        max_batch_size=300,
        device=device
    )

    base_dir = opts.video_dir
    save_dir = opts.save_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    print('video_dir', base_dir)
    print('save_dir', save_dir)

    video_info = []
    for tmp in sorted(glob.glob(os.path.join(base_dir, '*.mp4'))):
        name = os.path.basename(tmp).split('.')[0]
        video_info.append([name, tmp])

    # print(video_info[0])
    print(len(video_info))
    # video_info = video_info[:5]
    dataset = VideoDataset_1(video_info)
    loader = DataLoader(dataset, 1, shuffle=False, num_workers=6)
    result = {}
    video_cut = []
    for i, video in enumerate(tqdm(loader)):
        video = video[0]
        # assert video.shape[0] == video_info[i][2]
        pt = model.predict(video)
        # result[video_info[i][0]] = pt

        # info =  video_info[i]
        name = video_info[i][0]
        np.save(os.path.join(save_dir, name+'.npy'), pt)
        
        # cap = cv2.VideoCapture(video_info[i][1])
        # imgs = []
        # while(True):
        #     ret, img = cap.read()
        #     if ret:
        #         imgs.append(img)
        #     else:
        #         break
        
        # for i, img in enumerate(imgs):
        #     h, w, _ = img.shape
        #     for j, joint in enumerate(pt[i]):
        #         if j==0 or 5<=j<=10:
        #             x = int(joint[0]*w)
        #             y = int(joint[1]*h)
        #             img[x-2:x+2, y-2:y+2, :] = np.array([0, 0, 255])
        #     cv2.imwrite('demo/'+name+str(i)+'.jpg', img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=str, default='')
    parser.add_argument('-vd', '--video_dir', type=str, default='')
    parser.add_argument('-sd', '--save_dir', type=str, default='')
    parser.add_argument('-cd', '--change_dir', type=str, default='')
    opts = parser.parse_args()

    if opts.gpu !=  '':
        os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu

    csl_icme(opts=opts)
