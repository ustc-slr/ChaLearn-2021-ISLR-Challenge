import os
from tqdm import tqdm
import pickle

root_pose_path = '/data3/alexhu/Datasets/AUTSL_Upper/Keypoints_2d_mmpose/'
root_img_path = '/data3/alexhu/Datasets/AUTSL_Upper/jpg_video/'

split_path = os.listdir(root_img_path)
for split_index in (range(len(split_path))):
    split_name = split_path[split_index]
    real_split_name = os.path.join(root_img_path, split_name)
    video_list = os.listdir(real_split_name)
    for video_name in tqdm(video_list):
        if video_name.endswith('depth'):
            continue
        real_video_name = os.path.join(real_split_name, video_name)
        N_frame_img = len(os.listdir(real_video_name))
        real_kps_path = os.path.join(root_pose_path, split_name, video_name+'.pkl')
        with open(real_kps_path, 'rb') as f:
            kps_dict = pickle.load(f)
        N_frame_pose = kps_dict['keypoints'].shape[0]
        if N_frame_img != N_frame_pose + 1:
            print(real_video_name)