from __future__ import print_function, division
import os
import sys

if __name__ == "__main__":
    source_path = sys.argv[1]
    dst_path = sys.argv[2]
    N = int(sys.argv[3])

    class_path = os.listdir(source_path)
    for i in range(len(class_path)):
        if class_path[i] != 'test':
            continue

        real_class_path = os.path.join(source_path, class_path[i])
        # if os.path.exists(os.path.join(dst_path, class_path[i])):
        #     continue

        video_path = os.listdir(real_class_path)
        for j in range(len(video_path)):
            signder_id = int(video_path[j].split('_')[0][6:])
            if not (signder_id < N and signder_id >= N - 10):
                continue
            print(video_path[j], signder_id)
            if 'color' in video_path[j]:
                real_video_path = os.path.join(real_class_path, video_path[j])
                new_json_path = os.path.join(dst_path, class_path[i], video_path[j])
                if not os.path.exists(new_json_path):
                    os.makedirs(new_json_path)
                    pass

                cmd = './build/examples/openpose/openpose.bin --image_dir \"{}\" --hand --face --write_json \"{}\" --display 0 --render_pose 0'.format(real_video_path,
                                                                                                                                   new_json_path)
                # print(cmd)
                os.system(cmd)
            else:
                continue
