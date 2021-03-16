import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from argparse import ArgumentParser
import pickle
import cv2
from tqdm import tqdm
import time

from mmpose.apis import (inference_top_down_pose_model_multi, init_pose_model,
                         vis_pose_result)

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


def process_mmdet_results(mmdet_results, cat_id=0):
    """Process mmdet results, and return a list of bboxes.

    :param mmdet_results:
    :param cat_id: category id (default: 0 for human)
    :return: a list of detected bounding boxes
    """
    if isinstance(mmdet_results, tuple):
        det_results = mmdet_results[0]
    else:
        det_results = mmdet_results

    bboxes = det_results[cat_id]

    person_results = []
    for bbox in bboxes:
        person = {}
        person['bbox'] = bbox
        person_results.append(person)

    return person_results


def main():
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument('N', type=int, help='THRES')

    # parser.add_argument('--video-path', type=str, help='Video path')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show visualizations.')
    # parser.add_argument(
    #     '--out-video-root',
    #     default='',
    #     help='Root of the output video file. '
    #     'Default not saving the visualization video.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')

    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()

    # assert args.show or (args.out_video_root != '')
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    det_model = init_detector(
        args.det_config, args.det_checkpoint, device=args.device.lower())
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())

    dataset = pose_model.cfg.data['test']['type']

    # optional
    return_heatmap = False
    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None
    N = args.N
    print(N)
    root_path = '/data/alexhu/Datasets/NMFs-CSL/jpg_video_ori/'
    root_path_pose = '/data/alexhu/Datasets/NMFs-CSL/Joint_2D_mmpose/'
    class_path = sorted(os.listdir(root_path))

    for class_index in tqdm(range(len(class_path))):
        if not(class_index < N and class_index >= N - 100):
            continue
        class_name = class_path[class_index]
        real_class_name = os.path.join(root_path, class_name)
        video_list = os.listdir(real_class_name)
        for video_name in tqdm(video_list):
            real_video_name = os.path.join(real_class_name, video_name)
            img_name_list = sorted(os.listdir(real_video_name))
            pose_pkl = {}
            start_time = time.time()
            img_list = []
            img_list_pose = []
            mmdet_results_all = []

            # test images, the resulting box is (x1, y1, x2, y2)
            for img_name in img_name_list:
                if not img_name.endswith('jpg'):
                    continue
                real_img_name = os.path.join(real_video_name, img_name)
                img = cv2.imread(real_img_name)
                img_list_pose.append(img)
                img_list.append(img)
                if len(img_list) > 30:
                    mid_tmp = inference_detector(det_model, img_list)
                    mmdet_results_all += mid_tmp
                    img_list = []
            if len(img_list) != 0:
                mid_tmp = inference_detector(det_model, img_list)
                mmdet_results_all += mid_tmp
            # keep the person class bounding boxes.
            person_results = []
            for mmdet_img in mmdet_results_all:
                tmp = process_mmdet_results(mmdet_img)
                if len(tmp) > 1:
                    # print(video_name)
                    person_results.append([tmp[0]])
                else:
                    person_results.append(tmp)
            det_time = time.time() - start_time
            # test a single image, with a list of bboxes.

            pose_start_time = time.time()
            pose_results, returned_outputs = inference_top_down_pose_model_multi(
                pose_model,
                img_list_pose,
                person_results,
                bbox_thr=args.bbox_thr,
                format='xyxy',
                dataset=dataset,
                return_heatmap=return_heatmap,
                outputs=output_layer_names)
            print('Stage Detection Time:', det_time, 'Stage Pose Time:', time.time() - pose_start_time)
            pose_results['img_list'] = img_name_list
            if len(pose_results['img_list']) != len(pose_results['bbox']) or len(pose_results['img_list']) != pose_results['keypoints'].shape[0]:
                print('Unmatch:', video_name)
            if len(img_name_list) != len(pose_results['img_list']):
                print('Lost frames:', video_name)

            real_pkl_path = os.path.join(root_path_pose, class_name, video_name+'.pkl')
            if not os.path.exists(os.path.join(root_path_pose, class_name)):
                os.makedirs(os.path.join(root_path_pose, class_name))
            # print(real_pkl_path)
            with open(real_pkl_path, 'wb') as f:
                pickle.dump(pose_results, f)


            # # show the results
            # fps = 30
            # size = (img_list_pose[0].shape[0], img_list_pose[0].shape[0])
            # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # videoWriter = cv2.VideoWriter(
            #     'test.mp4', fourcc,
            #     fps, size)
            # i = 0
            # for img in img_list_pose:
            #     pose_results_cur = {'keypoints': pose_results['keypoints'][i],
            #                         'bbox': pose_results['bbox'][i]}
            #     vis_img = vis_pose_result(
            #         pose_model,
            #         img,
            #         [pose_results_cur],
            #         dataset=dataset,
            #         kpt_score_thr=args.kpt_thr,
            #         show=False)
            #     videoWriter.write(vis_img)
            #     i += 1
            # videoWriter.release()




if __name__ == '__main__':

    main()
