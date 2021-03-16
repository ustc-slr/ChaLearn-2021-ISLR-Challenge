import os
import json
import cv2
from tqdm import tqdm

video_root_path = '/data3/alexhu/Datasets/AUTSL_Upper/jpg_video/'
root_path = '/data3/alexhu/Datasets/AUTSL_Upper/keypoints_2d/'
out_path = '/data3/alexhu/Datasets/AUTSL_Upper/Keypoints_2d_process/'


def json_loader(name, frame_index, shape):
    height, width, _ = shape

    with open(name, 'r') as load_f:
        current_frame = {}
        current_frame['frame_index'] = frame_index
        current_frame['skeleton'] = []
        load_dict = json.load(load_f)
        # print(len(load_dict['people'][0]['pose_keypoints']) / 3)
        # print(len(load_dict['people'][0]['face_keypoints']) / 3)
        # print(len(load_dict['people'][0]['hand_left_keypoints']) / 3)
        # print(len(load_dict['people'][0]['hand_right_keypoints']) / 3)
        try:
            people = load_dict['people'][0]
        except:
            print(name)
            pose_frame = {}
            pose_frame['pose'] = []
            pose_frame['score'] = []
            for i in range(25):
                pose_frame['pose'].append(0.0)
                pose_frame['pose'].append(0.0)
                pose_frame['score'].append(0.0)
            # left_hand
            for i in range(21):
                pose_frame['pose'].append(0.0)
                pose_frame['pose'].append(0.0)
                pose_frame['score'].append(0.0)
            # right_hand
            for i in range(21):
                pose_frame['pose'].append(0.0)
                pose_frame['pose'].append(0.0)
                pose_frame['score'].append(0.0)
            # face
            for i in range(70):
                pose_frame['pose'].append(0.0)
                pose_frame['pose'].append(0.0)
                pose_frame['score'].append(0.0)
            current_frame['skeleton'].append(pose_frame)
            return current_frame

        # body pose
        pose_frame = {}
        pose_frame['pose'] = []
        pose_frame['score'] = []
        for i in range(25):
            pose_frame['pose'].append(load_dict['people'][0]['pose_keypoints_2d'][3 * i] / width)
            pose_frame['pose'].append(load_dict['people'][0]['pose_keypoints_2d'][3 * i + 1] / height)
            pose_frame['score'].append(load_dict['people'][0]['pose_keypoints_2d'][3 * i + 2])
        # left_hand
        for i in range(21):
            if load_dict['people'][0]['hand_left_keypoints_2d'][3 * i + 2] < 0.1:
                pose_frame['pose'].append(0.0)
                pose_frame['pose'].append(0.0)
                pose_frame['score'].append(0.0)
            else:
                pose_frame['pose'].append(load_dict['people'][0]['hand_left_keypoints_2d'][3 * i] / width)
                pose_frame['pose'].append(load_dict['people'][0]['hand_left_keypoints_2d'][3 * i + 1] / height)
                pose_frame['score'].append(load_dict['people'][0]['hand_left_keypoints_2d'][3 * i + 2])
        # right_hand
        for i in range(21):
            if load_dict['people'][0]['hand_right_keypoints_2d'][3 * i + 2] < 0.1:
                pose_frame['pose'].append(0.0)
                pose_frame['pose'].append(0.0)
                pose_frame['score'].append(0.0)
            else:
                pose_frame['pose'].append(load_dict['people'][0]['hand_right_keypoints_2d'][3 * i] / width)
                pose_frame['pose'].append(load_dict['people'][0]['hand_right_keypoints_2d'][3 * i + 1] / height)
                pose_frame['score'].append(load_dict['people'][0]['hand_right_keypoints_2d'][3 * i + 2])
        # face
        for i in range(70):
            if load_dict['people'][0]['face_keypoints_2d'][3 * i + 2] < 0.1:
                pose_frame['pose'].append(0.0)
                pose_frame['pose'].append(0.0)
                pose_frame['score'].append(0.0)
            else:
                pose_frame['pose'].append(load_dict['people'][0]['face_keypoints_2d'][3 * i] / width)
                pose_frame['pose'].append(load_dict['people'][0]['face_keypoints_2d'][3 * i + 1] / height)
                pose_frame['score'].append(load_dict['people'][0]['face_keypoints_2d'][3 * i + 2])

        current_frame['skeleton'].append(pose_frame)
        return current_frame



if __name__ == '__main__':
    split_path = os.listdir(root_path)
    for i in range(len(split_path)):
        real_split_path = os.path.join(root_path, split_path[i])
        video_path = os.listdir(real_split_path)
        for k in tqdm(range(len(video_path))):
            real_video_path = os.path.join(real_split_path, video_path[k])
            frame_path = sorted(os.listdir(real_video_path))
            shape = cv2.imread(os.path.join(video_root_path, split_path[i],
                                            video_path[k], '000000.jpg')).shape

            total = {}
            total['data'] = []
            for l in range(len(frame_path)):
                real_frame_path = os.path.join(real_video_path, frame_path[l])
                current_frame = json_loader(real_frame_path, l+1, shape)
                total['data'].append(current_frame)

            out_json_path = os.path.join(out_path, split_path[i], video_path[k] + '.json')
            out_json_dir = os.path.join(out_path, split_path[i])
            if not os.path.exists(out_json_dir):
                os.makedirs(out_json_dir)
            with open(out_json_path, 'w') as f:
                f.write(json.dumps(total))
