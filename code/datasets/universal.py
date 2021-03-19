import torch
import torch.utils.data as data
from PIL import Image
import os
import pickle
import math
import functools
import json
import copy
import random
from spatial_transforms import Compose
from utils import load_value_file
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def video_loader(video_dir_path, frame_indices, image_loader):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, '{:06d}.jpg'.format(i))
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video

    return video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)




##### Flow loader #######

def pil_loader_flow(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB').convert('L')

def pil_loader_depth(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def get_default_image_loader_flow():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader_flow

def get_default_image_loader_depth():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader_depth


def video_loader_flow(video_dir_path, frame_indices, image_loader):
    video = []
    for i in frame_indices:
        image_path_u = os.path.join(video_dir_path, 'u_{:06d}.jpg'.format(i))
        image_path_v = os.path.join(video_dir_path, 'v_{:06d}.jpg'.format(i))

        if os.path.exists(image_path_u) and os.path.exists(image_path_v):
            video.append([image_loader(image_path_u), image_loader(image_path_v)])
        else:
            return video
    return video


def video_loader_depth(video_dir_path, frame_indices, image_loader):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, '{:06d}.jpg'.format(i))
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video
    return video


def get_default_video_loader_flow():
    image_loader = get_default_image_loader_flow()
    return functools.partial(video_loader_flow, image_loader=image_loader)

def get_default_video_loader_depth():
    image_loader = get_default_image_loader_depth()
    return functools.partial(video_loader_depth, image_loader=image_loader)


def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)


def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def get_video_names_and_annotations(data, subset):
    video_names = []
    annotations = []

    for key, value in data['database'].items():
        this_subset = value['subset']
        if this_subset == subset:
            label = value['annotations']['label']
            video_names.append('{}/{}'.format(label, key))
            annotations.append(value['annotations'])

    return video_names, annotations


def make_dataset(root_path, annotation_path, subset, n_samples_for_each_video,
                 sample_duration):
    data = load_annotation_data(annotation_path)
    video_names, annotations = get_video_names_and_annotations(data, subset)
    class_to_idx = get_class_labels(data)
    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name

    dataset = []
    for i in range(len(video_names)):
        if i % 1000 == 0:
            print('dataset loading [{}/{}]'.format(i, len(video_names)))

        if subset == 'training':
            subset_real = 'train'
        elif subset == 'validation':
            subset_real = 'val'
        else:
            subset_real = 'test'

        real_name = video_names[i].split('/')[1] + '_color'


        video_path = os.path.join(root_path, subset_real, real_name) # for all
        if not os.path.exists(video_path):
            video_path = os.path.join(root_path, 'train', real_name)  # for all
        if not os.path.exists(video_path):
            video_path = os.path.join(root_path, 'val', real_name)  # for all
        # video_path = os.path.join(root_path, 'train', real_name) # for split

        if not os.path.exists(video_path):
            print(video_path)
            continue

        n_frames_file_path = os.path.join(video_path, 'n_frames')
        n_frames = int(load_value_file(n_frames_file_path))
        if n_frames <= 0:
            continue

        begin_t = 1
        end_t = n_frames
        sample = {
            'video': video_path,
            'segment': [begin_t, end_t],
            'n_frames': n_frames,
            'video_id': video_names[i].split('/')[1]
        }
        if len(annotations) != 0:
            sample['label'] = class_to_idx[annotations[i]['label']]
            # sample['label'] = int(annotations[i]['label'])

        else:
            sample['label'] = -1

        if n_samples_for_each_video == 1:
            sample['frame_indices'] = list(range(0, n_frames))
            dataset.append(sample)
        else:
            if n_samples_for_each_video > 1:
                step = max(1,
                           math.ceil((n_frames - 1 - sample_duration) /
                                     (n_samples_for_each_video - 1)))
            else:
                step = sample_duration
            for j in range(1, n_frames, step):
                sample_j = copy.deepcopy(sample)
                sample_j['frame_indices'] = list(
                    range(j, min(n_frames + 1, j + sample_duration)))
                dataset.append(sample_j)

    return dataset, idx_to_class


class Universal(data.Dataset):
    """
    Args:
        root (string): Root directory path.
        spatial_transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self,
                 root_path,
                 annotation_path,
                 subset,
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=16,
                 modality='rgb',
                 get_loader=get_default_video_loader):

        if subset == 'training':
            self.data, self.class_names = make_dataset(
                root_path, annotation_path, subset, n_samples_for_each_video,
                sample_duration)
            # self.val_data, _ = make_dataset(
            #     root_path, annotation_path, 'validation', n_samples_for_each_video,
            #     sample_duration)
            # self.data += self.val_data
        else:
            self.data, self.class_names = make_dataset(
                root_path, annotation_path, 'testing', n_samples_for_each_video,
                sample_duration)

        print('loaded', len(self.data))

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform

        self.subset = subset
        self.modality = modality
        if self.modality == 'flow':
            self.loader = get_default_video_loader_flow()
        elif self.modality == 'depth':
            self.loader = get_default_video_loader_depth()
        else:
            self.loader = get_loader()

        sometimes = lambda aug: iaa.Sometimes(0.3, aug)
        self.aug_seq = iaa.Sequential([
            # iaa.Fliplr(0.5),
            # sometimes(iaa.MotionBlur(k=2)),
            # sometimes(iaa.ChangeColorTemperature((1100, 10000))),
            sometimes(iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30))),
            # sometimes(iaa.Affine(scale={'x': (0.8, 1.2), 'y': (0.8, 1.2)},
            #                      translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            #                      rotate=(-20, 20),
            #                      shear=(-10, 10),
            #                      cval=(0, 255),
            #                      mode=ia.ALL, )),
            # sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.15))),
            # sometimes(iaa.AdditiveGaussianNoise(scale=0.05 * 255)),
            ])
        self.aug_seq.to_deterministic()


        # added by alexhu
        self.root_path = root_path
        if self.modality != 'pose':
            self.to_tensor = Compose(self.spatial_transform.transforms[-2:])
            self.spatial_transform.transforms = self.spatial_transform.transforms[:-2]
        # add

    def random_move(self, data_numpy,
                    angle_candidate=[-10., -5., 0., 5., 10.],
                    scale_candidate=[0.9, 1.0, 1.1],
                    transform_candidate=[-0.2, -0.1, 0.0, 0.1, 0.2],
                    move_time_candidate=[1]):
        # input: C,T,V,M
        C, T, V, M = data_numpy.shape
        move_time = random.choice(move_time_candidate)
        node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
        node = np.append(node, T)
        num_node = len(node)

        A = np.random.choice(angle_candidate, num_node)
        S = np.random.choice(scale_candidate, num_node)
        T_x = np.random.choice(transform_candidate, num_node)
        T_y = np.random.choice(transform_candidate, num_node)

        a = np.zeros(T)
        s = np.zeros(T)
        t_x = np.zeros(T)
        t_y = np.zeros(T)

        # linspace
        for i in range(num_node - 1):
            a[node[i]:node[i + 1]] = np.linspace(
                A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
            s[node[i]:node[i + 1]] = np.linspace(S[i], S[i + 1],
                                                 node[i + 1] - node[i])
            t_x[node[i]:node[i + 1]] = np.linspace(T_x[i], T_x[i + 1],
                                                   node[i + 1] - node[i])
            t_y[node[i]:node[i + 1]] = np.linspace(T_y[i], T_y[i + 1],
                                                   node[i + 1] - node[i])

        theta = np.array([[np.cos(a) * s, -np.sin(a) * s],
                          [np.sin(a) * s, np.cos(a) * s]])

        # perform transformation
        for i_frame in range(T):
            xy = data_numpy[0:2, i_frame, :, :]
            new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))
            new_xy[0] += t_x[i_frame]
            new_xy[1] += t_y[i_frame]
            data_numpy[0:2, i_frame, :, :] = new_xy.reshape(2, V, M)

        return data_numpy


    def pose2d_loader_mmpose(self, path, total_frame):
        def pickle_load(path):
            with open(path, 'rb') as f:
                pose_all = pickle.load(f)
            return pose_all

        split_root_path = self.root_path.split('/')[:-1]
        root_path_new = ''
        for path_tmp in split_root_path:
            root_path_new = root_path_new + path_tmp + '/'
        root_path = os.path.join(root_path_new, 'Keypoints_2d_mmpose')
        root_img_path = os.path.join(root_path_new, 'jpg_video')
        split_path = path.split('/')
        pose_path = os.path.join(root_path, split_path[-2], split_path[-1]+'.pkl')
        img_path = os.path.join(root_img_path, split_path[-2], split_path[-1], '000000.jpg')
        img_shape = Image.open(img_path).size

        if not os.path.exists(pose_path):
            print(pose_path)
        pose_all = pickle_load(pose_path)

        if pose_all['keypoints'].shape[0] != total_frame:
            print(path, total_frame, pose_all['keypoints'].shape)
        pose_ori = torch.from_numpy(pose_all['keypoints'])# T, J, 3
        pose_ori = torch.cat([pose_ori[:, 0:11], pose_ori[:, 40:]], 1)
        pose_2d = pose_ori / torch.tensor([img_shape[0], img_shape[1], 1.])

        return pose_2d



    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        if self.modality == 'rgb':
            path = self.data[index]['video']
            if self.modality == 'flow':
                path = path.replace('jpg_video', 'flow')
                path = path.replace('_color', '_flow')
            frame_indices = self.data[index]['frame_indices']
            if self.temporal_transform is not None:
                frame_indices = self.temporal_transform(frame_indices)
            clip = self.loader(path, frame_indices)
            if self.spatial_transform is not None:
                self.spatial_transform.randomize_parameters()
                clip = [self.spatial_transform(img) for img in clip]

                aug_seq_det = self.aug_seq.to_deterministic()
                if self.subset == 'training':
                    clip = [np.asarray(img) for img in clip]
                    clip = [aug_seq_det.augment_images([img])[0] for img in clip]
                    # clip = self.aug_seq.augment_images(clip)
                    clip = [Image.fromarray(img) for img in clip]

                clip = [self.to_tensor(img) for img in clip]
            clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
            bbox = np.array([[0., 0., 0., 0., 0.]])
        elif self.modality == 'flow':
            path = self.data[index]['video']
            if self.modality == 'flow':
                path = path.replace('jpg_video', 'flow')
                path = path.replace('_color', '_flow')
            frame_indices = self.data[index]['frame_indices']
            if self.temporal_transform is not None:
                frame_indices = self.temporal_transform(frame_indices)
            clip = self.loader(path, frame_indices)
            if self.spatial_transform is not None:
                self.spatial_transform.randomize_parameters()
                clip = [self.spatial_transform(img) for img in clip]
                clip = [self.to_tensor(img) for img in clip]
            clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
            bbox = np.array([[0., 0., 0., 0., 0.]])
        elif self.modality == 'depth':
            path = self.data[index]['video']
            if self.modality == 'depth':
                path = path.replace('_color', '_depth')
            frame_indices = self.data[index]['frame_indices']
            if self.temporal_transform is not None:
                frame_indices = self.temporal_transform(frame_indices)
            clip = self.loader(path, frame_indices)
            if self.spatial_transform is not None:
                self.spatial_transform.randomize_parameters()
                clip = [self.spatial_transform(img) for img in clip]
                clip = [self.to_tensor(img) for img in clip]
            clip = torch.stack(clip, 0).permute(1, 0, 2, 3)[:2]
            bbox = np.array([[0., 0., 0., 0., 0.]])
        elif self.modality in ['part', 'face', 'lhand', 'rhand']:
            path = self.data[index]['video']
            split_path = path.split('/')
            part_path = os.path.join('../data/AUTSL/jpg_right_hand/', split_path[-2], split_path[-1])
            frame_indices = [int(x.split('.')[0]) for x in sorted(os.listdir(part_path))]

            # frame_indices = self.data[index]['frame_indices']
            if self.temporal_transform is not None:
                frame_indices = self.temporal_transform(frame_indices)
            clip = self.loader(part_path, frame_indices)
            if self.spatial_transform is not None:
                self.spatial_transform.randomize_parameters()
                clip = [self.spatial_transform(img) for img in clip]

                aug_seq_det = self.aug_seq.to_deterministic()
                if self.subset == 'training':
                    clip = [np.asarray(img) for img in clip]
                    clip = [aug_seq_det.augment_images([img])[0] for img in clip]
                    # clip = self.aug_seq.augment_images(clip)
                    clip = [Image.fromarray(img) for img in clip]

                clip = [self.to_tensor(img) for img in clip]
            clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
            bbox = np.array([[0., 0., 0., 0., 0.]])
        else:
            path = self.data[index]['video']
            n_frames = self.data[index]['n_frames']
            split_path = path.split('/')

            frame_indices = self.data[index]['frame_indices']
            if self.temporal_transform is not None:
                frame_indices = self.temporal_transform(frame_indices)

            # 2d pose
            pose_2d = self.pose2d_loader_mmpose(path, n_frames)  # T, J, 3
            pose_indices = [i - 1 for i in frame_indices]
            pose = pose_2d[pose_indices]
            pose = pose.unsqueeze(3).permute(2, 0, 1, 3).numpy()  # C,T,V,M
            if self.subset == 'training':
                pose = self.random_move(pose)  # C,T,V,M
            clip = torch.Tensor(pose).squeeze(3).permute(1, 2, 0).float()
            clip = clip.permute(2, 0, 1).unsqueeze(3).float()
            bbox = np.array([[0., 0., 0., 0., 0.]])

        target = self.data[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return clip, target, bbox

    def __len__(self):
        return len(self.data)
