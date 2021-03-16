import os
import pickle
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

root_img_path = '/data/user/AUTSL/jpg_video/'
root_face_path = '/data/user/AUTSL/jpg_face/'
root_joint_path = '/data/user/AUTSL/Keypoints_2d_mmpose/'


def visual(bbx, save_path, img_path=None, img=None):
    if img is None:
        img = Image.open(img_path)
    plt.figure()
    plt.cla()
    plt.imshow(img)
    rect = plt.Rectangle((bbx[0], bbx[2]), int(bbx[1] - bbx[0]), int(bbx[3] - bbx[2]), fill=False, edgecolor='red')
    plt.gca().add_patch(rect)
    # joints = [(0, 1), (1, 2), (2, 3), (3, 4),
    #           (0, 5), (5, 6), (6, 7), (7, 8),
    #           (0, 9), (9, 10), (10, 11), (11, 12),
    #           (0, 13), (13, 14), (14, 15), (15, 16),
    #           (0, 17), (17, 18), (18, 19), (19, 20)]
    # for j in range(len(joints)):
    #     plt.plot([kp2ds[joints[j][0]][0], kp2ds[joints[j][1]][0]],
    #              [kp2ds[joints[j][0]][1], kp2ds[joints[j][1]][1]],
    #              linewidth=1, color='cyan')
    # plt.show()
    plt.savefig(save_path)

def crop_face(keypoints):
    bbx_ratio = 2.0
    face_kp2d = keypoints[0:5, :]
    u_max = np.max(face_kp2d[:, 0])
    u_min = np.min(face_kp2d[:, 0])
    v_max = np.max(face_kp2d[:, 1])
    v_mid = np.min(face_kp2d[:, 1])  # 认为眼睛所在位置正好处于脸的中部
    center_x = (u_min + u_max) / 2
    center_y = v_mid
    u_len = (u_max - u_min) * bbx_ratio
    v_len = (v_max - v_mid) * 2 * bbx_ratio
    bbx_len = max(u_len, v_len)
    assert bbx_len > 20
    return [center_x - bbx_len/2, center_x + bbx_len/2, center_y - bbx_len/2, center_y + bbx_len/2]

split_list = os.listdir(root_img_path)
for split_name in split_list:
    if split_name != 'test':
        continue
    real_split_path = os.path.join(root_img_path, split_name)
    video_list = os.listdir(real_split_path)
    for video_name in tqdm(video_list):
        if video_name.endswith('depth'):
            continue
        # print(split_name, video_name)

        joint_path = os.path.join(root_joint_path, split_name, video_name+'.pkl')
        with open(joint_path, 'rb') as f:
            all_dict = pickle.load(f)
        video_joints = all_dict['keypoints']
        bbx = crop_face(video_joints[0, :, :])
        real_video_path = os.path.join(real_split_path, video_name)
        img_list = os.listdir(real_video_path)
        for img_name in img_list:
            if not img_name.endswith('jpg'):
                continue
            real_img_path = os.path.join(real_video_path, img_name)
            clr = Image.open(real_img_path)
            save_dir = os.path.join(root_face_path, split_name, video_name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = os.path.join(root_face_path, split_name, video_name, img_name)
            clr = clr.crop([bbx[0], bbx[2], bbx[1], bbx[3]])
            clr.save(save_path)


        # visual(bbx, save_path=save_path, img=clr)

