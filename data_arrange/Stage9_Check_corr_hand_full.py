import os
from tqdm import tqdm

root_img_path = '/data3/alexhu/Datasets/AUTSL_Upper/jpg_video/'
root_hand_img_path = '/data3/alexhu/Datasets/AUTSL_Upper/jpg_left_hand/'

split_path = os.listdir(root_img_path)
for split_name in split_path:
    if split_name != 'test':
        continue
    real_split_path = os.path.join(root_img_path, split_name)
    video_list = os.listdir(real_split_path)
    for video in tqdm(video_list):
        if video.endswith('depth'):
            continue
        real_video_name = os.path.join(real_split_path, video)
        n_frames_full = len(os.listdir(real_video_name))
        real_hand_name = os.path.join(root_hand_img_path, split_name, video)
        n_frames_hand = len(os.listdir(real_hand_name))
        if n_frames_full != n_frames_hand + 1:
            print(real_video_name, n_frames_full, n_frames_hand)