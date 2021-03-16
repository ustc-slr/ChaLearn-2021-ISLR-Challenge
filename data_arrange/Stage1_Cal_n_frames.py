import os
from tqdm import tqdm

root_path = '/data3/alexhu/Datasets/AUTSL_Upper/jpg_video/'
split_path = os.listdir(root_path)
frames = []

for i in tqdm(range(len(split_path))):
    if split_path[i] != 'val':
        continue
    real_split_path = os.path.join(root_path, split_path[i])
    video_path = os.listdir(real_split_path)
    for j in range(len(video_path)):
        real_video_path = os.path.join(real_split_path, video_path[j])
        frame_path = sorted(os.listdir(real_video_path))
        frames.append(len(frame_path))
        if len(frame_path) < 10:
            print(split_path[i], video_path[j], len(frame_path))

print('Avg', sum(frames) / float(len(frames)))
print('Max', max(frames), 'Min', min(frames))
