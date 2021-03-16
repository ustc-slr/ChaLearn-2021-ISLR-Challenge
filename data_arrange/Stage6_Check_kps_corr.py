import os
from tqdm import tqdm

kps_path = '/data3/alexhu/Datasets/AUTSL_Upper/keypoints_2d/'
img_path = '/data3/alexhu/Datasets/AUTSL_Upper/jpg_video/'

split_path = os.listdir(img_path)
for split in split_path:
    real_split_path = os.path.join(img_path, split)
    video_path = os.listdir(real_split_path)
    for video in tqdm(video_path):
        if not video.endswith('color'):
            continue
        real_video_path = os.path.join(real_split_path, video)
        real_kps_path = os.path.join(kps_path, split, video)
        if len(os.listdir(real_video_path)) != len(os.listdir(real_kps_path)) + 1:
            print(real_video_path, len(os.listdir(real_video_path)), len(os.listdir(real_kps_path)))