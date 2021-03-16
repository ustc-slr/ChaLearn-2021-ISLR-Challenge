import argparse
import glob
import os
import pickle

import numpy as np
import cv2
from torch.utils.data import DataLoader, Dataset

import lmdb


class Video3D(object):
    def __init__(self, info_dict):
        r"""
            keys of info_dict: name, path
            notes: if indexs of your img is 0,1,2,3, your frame num should be 4.
        """
        self.name = info_dict['name']
        self.path = info_dict['path']
        self.save_path = info_dict['save_path']

    def get_cover_frames(self):
        r"""
            return:
                num_frames * height * width * channel (rgb:3 , flow:2) 
        """
        video = cv2.VideoCapture(self.path)
        count = video.get(cv2.CAP_PROP_FRAME_COUNT)
        frames = []
        ret, frame = video.read()
        assert ret
        cv2.imwrite(self.save_path+'.jpg', frame)
        return frames


class FullVideoDateset(Dataset):

    def __init__(self, video_info):
        self.videos = [Video3D(x) for x in video_info]

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        clip = self.videos[index].get_cover_frames()     
        name = self.videos[index].name
        return clip, name


def collate_fn_fullvideo(batch):
    clip, name = zip(*batch)
    return clip[0], name[0]


def load_info(data_path, target_path):
    infos = []
    # for task in ['train', 'val']:
    for task in ['test']:
        task_info = []
        data = glob.glob(os.path.join(data_path, task, '*color.mp4'))
        for _, row in enumerate(data):
            name = os.path.basename(row).split('.')[0]
            video_info = {
                'name': name,
                'path': row,
                'save_path': os.path.join(target_path, task, name)
            }
            task_info.append(video_info)
        infos.append(task_info)
        print('Count-{:s}: {:d}'.format(task, len(task_info)))
    return infos


def main(opts):
    infos = load_info(opts.source_path, opts.target_path)

    dataset = [FullVideoDateset(x) for x in infos]

    loader = [DataLoader(x, 1, shuffle=False, num_workers=opts.num_workers, collate_fn=collate_fn_fullvideo) for x in dataset]

    # if not os.path.exists(opts.target_path):
    #     os.makedirs(opts.target_path)
    #     print('makedirs: {:s}'.format(opts.target_path))

    for k, task_loader in enumerate(loader):
        for i, (clip, name) in enumerate(task_loader):
            print('{:d}/{:d}, {:d}, {:s}'.format(i+1, len(infos[k]), len(clip), name))


def parse_args():
    p = argparse.ArgumentParser(description='slr')
    p.add_argument('source_path', type=str, default='', help='source path of original video')
    p.add_argument('target_path', type=str, default='', help='target path of first frame') 
    p.add_argument('-nw', '--num_workers', type=int, default=4, help='num_workers of dataloader')
    parameter = p.parse_args()    
    return  parameter


if __name__ == '__main__':
    opts = parse_args()
    main(opts)
