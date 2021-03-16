import os
import sys


def gen_ffmpeg(base, task, save):
    with open(f'preparation/pre_data/box_{task}.txt', 'r') as f:
        boxes = f.readlines()

    lines = []
    folders = []
    def f(x):
        assert 0<=int(x)<=511
        return int(x)
    for i, box in enumerate(boxes):
        name, x1, y1, x2, y2 = box.strip().split(',')
        x1 = f(x1)
        y1 = f(y1)
        x2 = f(x2)
        y2 = f(y2)
        assert x2-x1 == y2-y1
        for modality in ['color', 'depth']:
            raw = os.path.join(base, task, name+f'_{modality}.mp4')
            new = os.path.join(save, task, name+f'_{modality}', r'%06d.jpg')
            x1 = int(x1)
            y1 = int(y1)
            side = int(x2)-int(x1)
            lines.append('ffmpeg -i {:s} -qscale:v 2 -start_number 0 -filter:v \"crop={:d}:{:d}:{:d}:{:d}\" {:s} -loglevel 24 \n'.format(
                raw, side, side, x1, y1, new
                ))
            folders.append(f'mkdir data/AUTSL/image/{task}/{name}_{modality}\n')
    with open(f'preparation/pre_data/ffmpeg_{task}.sh', 'w') as f:
        f.writelines(lines)
    with open(f'preparation/pre_data/ffmpeg_folder_{task}.sh', 'w') as f:
        f.writelines(folders)

if __name__ == "__main__":
    # gen_ffmpeg('data/AUTSL/raw', 'train', 'data/AUTSL/image')
    # gen_ffmpeg('data/AUTSL/raw', 'val', 'data/AUTSL/image')
    gen_ffmpeg('data/AUTSL/raw', 'test', 'data/AUTSL/image')
