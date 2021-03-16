import os
import pickle
import numpy as np
import math
from PIL import Image
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm

LEN_KPS = 10
THRE = 0.5
threshold = 0.5
img_size = np.array([[256.0, 256.0]], dtype=np.float32)
HAND_SIDE = 'left'

def crop_hand(frames_list, ori_img_size, hand_side):
    frames_list_new = []
    bbxes = []
    valid_frame_list = []
    for i in range(frames_list.shape[0]):
        frame_data = frames_list[i]
        skeleton = frame_data[:, 0:2]
        confidence = frame_data[:, 2]
        usz, vsz = [ori_img_size[0][0], ori_img_size[0][1]]
        minsz = min(usz, vsz)
        maxsz = max(usz, vsz)
        if hand_side == 'right':
            right_keypoints = skeleton[112:133, :]
            kp_visible = (confidence[112:133] > 0.1)
            uvis = right_keypoints[kp_visible, 0]
            vvis = right_keypoints[kp_visible, 1]
        elif hand_side == 'left':
            left_keypoints = skeleton[91:112, :]
            kp_visible = (confidence[91:112] > 0.1)
            uvis = left_keypoints[kp_visible, 0]
            vvis = left_keypoints[kp_visible, 1]
        else:
            raise ValueError('wrong hand side')
        if len(uvis) < LEN_KPS:
            bbx = elbow_hand(skeleton, confidence, ori_img_size, hand_side)
            if bbx is None:
                continue
            else:
                bbxes.append(bbx)
                frames_list_new.append(frame_data)
                valid_frame_list.append(i)
        else:
            umin = min(uvis)
            vmin = min(vvis)
            umax = max(uvis)
            vmax = max(vvis)

            B = round(2.2 * max([umax - umin, vmax - vmin]))

            us = 0
            ue = usz - 1
            vs = 0
            ve = vsz - 1
            umid = umin + (umax - umin) / 2
            vmid = vmin + (vmax - vmin) / 2

            if (B < minsz - 1):
                us = round(max(0, umid - B / 2))
                ue = us + B
                if (ue > usz - 1):
                    d = ue - (usz - 1)
                    ue = ue - d
                    us = us - d
                vs = round(max(0, vmid - B / 2))
                ve = vs + B
                if (ve > vsz - 1):
                    d = ve - (vsz - 1)
                    ve = ve - d
                    vs = vs - d
            if (B >= minsz - 1):
                B = minsz - 1
                if usz == minsz:
                    vs = round(max(0, vmid - B / 2))
                    ve = vs + B
                    if (ve > vsz - 1):
                        d = ve - (vsz - 1)
                        ve = ve - d
                        vs = vs - d
                if vsz == minsz:
                    us = round(max(0, umid - B / 2))
                    ue = us + B

                    if (ue > usz - 1):
                        d = ue - (usz - 1)
                        ue = ue - d
                        us = us - d
            us = int(us)
            vs = int(vs)
            ue = int(ue)
            ve = int(ve)
            bbx = [us, ue, vs, ve]
            bbxes.append(bbx)
            frames_list_new.append(frame_data)
            valid_frame_list.append(i)

    bbxes = np.array(bbxes, dtype=np.float32)
    average_width = np.average(bbxes[:, 1] - bbxes[:, 0])
    average_height = np.average(bbxes[:, 3] - bbxes[:, 2])
    rescale_bbx = np.array([average_width, average_height], dtype=np.float32)
    return frames_list_new, bbxes, rescale_bbx, valid_frame_list


def elbow_hand(pose_keypoints, confidence, ori_img_size, hand_side):
    right_hand = pose_keypoints[[2, 3, 4]]
    left_hand = pose_keypoints[[5, 6, 7]]
    ratioWristElbow = 0.33
    detect_result = []
    img_width, img_height = [ori_img_size[0][0], ori_img_size[0][1]]
    if hand_side == 'right':
        has_right = np.sum(confidence[[2, 3, 4]] < THRE) == 0
        if not has_right:
            return None
        x1, y1 = right_hand[0][:2]
        x2, y2 = right_hand[1][:2]
        x3, y3 = right_hand[2][:2]

        x = x3 + ratioWristElbow * (x3 - x2)
        y = y3 + ratioWristElbow * (y3 - y2)
        distanceWristElbow = math.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
        distanceElbowShoulder = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        width = 1.1 * max(distanceWristElbow, 0.9 * distanceElbowShoulder)
        x -= width / 2
        y -= width / 2  # width = height

        if x < 0: x = 0
        if y < 0: y = 0
        width1 = width
        width2 = width
        if x + width > img_width: width1 = img_width - x
        if y + width > img_height: width2 = img_height - y
        width = min(width1, width2)
        detect_result.append([int(x), int(y), int(width)])

    elif hand_side == 'left':
        has_left = np.sum(confidence[[5, 6, 7]] < THRE) == 0
        if not has_left:
            return None
        x1, y1 = left_hand[0][:2]
        x2, y2 = left_hand[1][:2]
        x3, y3 = left_hand[2][:2]

        x = x3 + ratioWristElbow * (x3 - x2)
        y = y3 + ratioWristElbow * (y3 - y2)
        distanceWristElbow = math.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
        distanceElbowShoulder = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        width = 1.1 * max(distanceWristElbow, 0.9 * distanceElbowShoulder)
        x -= width / 2
        y -= width / 2  # width = height
        if x < 0: x = 0
        if y < 0: y = 0
        width1 = width
        width2 = width
        if x + width > img_width: width1 = img_width - x
        if y + width > img_height: width2 = img_height - y
        width = min(width1, width2)
        detect_result.append([int(x), int(y), int(width)])

    x, y, width = int(x), int(y), int(width)
    return [x, x + width, y, y + width]


def get_kp2ds(skeleton, conf, threshold, hand_side):
        if hand_side == 'left':
            hand_kp2d = skeleton[91:112, :]
            confidence = conf[91:112]
        elif hand_side == 'right':
            hand_kp2d = skeleton[112:133, :]
            confidence = conf[112:133]
        else:
            raise Exception('wrong hand_side type')
        confidence = np.where(confidence > threshold, confidence, 0.0)
        indexes = np.where(confidence < threshold)[0].tolist()
        for i in range(len(indexes)):
            hand_kp2d[indexes[i]] = np.zeros((1, 2), dtype=np.float32)
        confidence = np.tile(confidence[:, np.newaxis], (1, 2))
        return hand_kp2d, confidence


def visual(kp2ds, save_path, img_path=None, img=None):
        if img is None:
            img = Image.open(img_path)
        plt.figure()
        plt.imshow(img)
        for j in range(0, 1):
            if kp2ds[j][0] < 0.0:
                continue
            plt.plot(kp2ds[j][0], kp2ds[j][1], 'bo')
        for j in range(1, 5):
            if kp2ds[j][0] < 0.0:
                continue
            plt.plot(kp2ds[j][0], kp2ds[j][1], 'ro')
        for j in range(5, 9):
            if kp2ds[j][0] < 0.0:
                continue
            plt.plot(kp2ds[j][0], kp2ds[j][1], 'go')
        for j in range(9, 13):
            if kp2ds[j][0] < 0.0:
                continue
            plt.plot(kp2ds[j][0], kp2ds[j][1], 'yo')
        for j in range(13, 17):
            if kp2ds[j][0] < 0.0:
                continue
            plt.plot(kp2ds[j][0], kp2ds[j][1], 'ko')
        for j in range(17, 21):
            if kp2ds[j][0] < 0.0:
                continue
            plt.plot(kp2ds[j][0], kp2ds[j][1], 'mo')
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




root_img_path = '/data/user/AUTSL/jpg_video/'
root_joint_path = '/data/user/AUTSL/Keypoints_2d_mmpose/'
root_hand_img_path = '/data/user/AUTSL/jpg_left_hand/'
video_joint_pkl = {}

split_list = os.listdir(root_img_path)
for split_name in split_list:
    if split_name != 'test':
        continue
    real_split_path = os.path.join(root_img_path, split_name)
    video_list = os.listdir(real_split_path)
    for video_name in tqdm(video_list):
        if video_name.endswith('depth'):
            continue
        print(split_name, video_name)

        joint_path = os.path.join(root_joint_path, split_name, video_name+'.pkl')
        with open(joint_path, 'rb') as f:
            all_dict = pickle.load(f)
        video_joints = all_dict['keypoints']
        real_video_path = os.path.join(real_split_path, video_name)
        real_img_path = os.path.join(real_split_path, video_name, '000000.jpg')
        ori_img_size = Image.open(real_img_path).size
        ori_img_size = np.array([[ori_img_size[0], ori_img_size[1]]], dtype=np.float32)
        video_joints_update, bbxes, rescale_bbx, valid_frame_list = crop_hand(video_joints, ori_img_size, HAND_SIDE)

        clrs = []
        img_list = sorted(os.listdir(real_video_path))
        img_list.pop(-1)
        if len(img_list) != len(valid_frame_list):
            print(real_video_path)
        frame_dict = {}
        for i in range(len(valid_frame_list)):
            frame_index = '%06d.jpg' %(valid_frame_list[i])
            real_frame_path = os.path.join(real_video_path, frame_index)

            frame = video_joints_update[i]
            skeleton = frame[:, 0:2]
            confidence = frame[:, 2]
            kp2ds, confidence = get_kp2ds(skeleton, confidence, threshold, HAND_SIDE)

            # bbox x1 x2 y1 y2
            center = [(bbxes[i][1] + bbxes[i][0])//2, (bbxes[i][3] + bbxes[i][2])//2]
            B = int(rescale_bbx[0]) // 2
            if center[0] < B:
                B = center[0]
            elif center[0] + B > ori_img_size[0][0] -1:
                B = ori_img_size[0][0] - 1 - center[0]
            if center[1] < B:
                B = center[1]
            elif center[1] + B > ori_img_size[0][1] -1:
                B = ori_img_size[0][1] - 1 - center[1]
            scale = np.array(
                [[2 * B, 2 * B]],
                dtype=np.float32)
            trans = np.array([[center[0]-B, center[1]-B]], dtype=np.float32)
            # trans = np.array([[bbxes[i][0], bbxes[i][2]]], dtype=np.float32)
            # scale = np.array(
            #     [[bbxes[i][1] - bbxes[i][0], bbxes[i][3] - bbxes[i][2]]], # bbox x1 x2 y1 y2
            #     dtype=np.float32)
            assert scale[0, 1] > 0.0 and scale[0, 0] > 0.0
            kp2ds = (kp2ds - trans) / scale * img_size
            kp2ds = np.where(kp2ds > 0.0, kp2ds, 0.0)
            gt = copy.deepcopy(kp2ds)

            clr = Image.open(real_frame_path)
            # clr = clr.crop(
            #     (bbxes[i][0], bbxes[i][2], bbxes[i][1], bbxes[i][3]))
            clr = clr.crop(
                (center[0]-B, center[1]-B, center[0]+B, center[1]+B))
            clr = clr.resize((256, 256))
            clrs.append(clr)


            frame_dict[valid_frame_list[i]] = [gt, confidence, [center[0]-B, center[1]-B, center[0]+B, center[1]+B]]
            hand_video_path = os.path.join(root_hand_img_path, split_name, video_name)
            if not os.path.exists(hand_video_path):
                os.makedirs(hand_video_path)
            hand_img_path = os.path.join(hand_video_path, frame_index)
            # visual(gt, hand_img_path, img=clr)
            clr.save(hand_img_path)
        video_joint_pkl[video_name] = frame_dict
        # exit()

with open('Joint_left_hand_test.pkl', 'wb') as f:
    pickle.dump(video_joint_pkl, f)
