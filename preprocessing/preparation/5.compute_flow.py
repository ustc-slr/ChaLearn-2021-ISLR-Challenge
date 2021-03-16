import os
import numpy as np
import cv2
from glob import glob
import multiprocessing as mp
import time
_IMAGE_SIZE = 256


def imread_resize(path):
    image = cv2.imread(path)
    image = cv2.resize(image, (_IMAGE_SIZE, _IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
    return image


def cal_for_frames(device, video_path):
    cv2.cuda.setDevice(device)
    frames = glob(os.path.join(video_path, '*.jpg'))
    frames.sort()

    flow = []
    prev = imread_resize(frames[0])
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    prev = cv2.cuda_GpuMat(prev)
    for i, frame_curr in enumerate(frames):
        curr = imread_resize(frame_curr)
        curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        curr = cv2.cuda_GpuMat(curr)
        tmp_flow = compute_TVL1(prev, curr)
        flow.append(tmp_flow)
        prev = curr

    return flow


def compute_TVL1(prev, curr, bound=20):
    """Compute the TV-L1 optical flow."""
    TVL1 = cv2.cuda.OpticalFlowDual_TVL1_create()
    flow = TVL1.calc(prev, curr, None)
    flow = flow.download()
    assert flow.dtype == np.float32

    flow = (flow + bound) * (255.0 / (2*bound))
    flow = np.round(flow).astype(int)
    flow[flow >= 255] = 255
    flow[flow <= 0] = 0

    return flow


def save_flow(video_flows, flow_path):
    for i, flow in enumerate(video_flows):
        cv2.imwrite(os.path.join(flow_path, "u_{:06d}.jpg".format(i)), flow[:, :, 0])
        cv2.imwrite(os.path.join(flow_path, "v_{:06d}.jpg".format(i)), flow[:, :, 1])


def gen_args(task, num_thread, num_gpu):
    with open(f'preparation/pre_data/file_{task}.txt', 'r') as f:
        info = f.readlines()
        info = [x.strip().split(',')[0] for x in info]

    args = []
    source = f'./data/AUTSL/image/{task}'
    target = f'./data/AUTSL/flow/{task}'

    for i, name in enumerate(info):
        args.append((os.path.join(source, name+'_color'), os.path.join(target, name+'_flow'), i, num_thread, num_gpu))
    return args


def extract_flow(args):
    video_path, flow_path, idx, num_thread, num_gpu = args
    device = mp.current_process()._identity[0] % num_gpu
    start = time.time()
    if os.path.exists(flow_path):
        raw = len(glob(os.path.join(video_path, '*.jpg')))
        now = len(glob(os.path.join(flow_path, '*.jpg')))
        if now == 2*raw:
            return
        else:
            print(10*'!', flow_path)
    flow = cal_for_frames(device, video_path)
    if not os.path.exists(flow_path):
        os.mkdir(flow_path)
    save_flow(flow, flow_path)
    print(device, mp.current_process()._identity[0], flow_path, time.time()-start)
    return


if __name__ =='__main__':
    num_thread = 10
    num_gpu = 2
    pool = mp.Pool(num_thread)   # multi-processing

    # args = gen_args('val', num_thread, num_gpu)
    # pool.map(extract_flow, args)
    # print('over')
    # args = gen_args('train', num_thread, num_gpu)
    # pool.map(extract_flow, args)

    # print(len(args))
    args = gen_args('test', num_thread, num_gpu)
    pool.map(extract_flow, args)