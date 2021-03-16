import pickle
import torch
import torch.nn.functional as F

with open('results/AUTSL_I3D_BSL2_all.pkl', 'rb') as f:
    avg_score_i3d = pickle.load(f)
with open('results/AUTSL_I3D_BSL_all.pkl', 'rb') as f:
    avg_score_i3d2 = pickle.load(f)
with open('results/AUTSL_slowfast2_all.pkl', 'rb') as f:
    avg_score_slowfast = pickle.load(f)
with open('results/AUTSL_slowfast_all.pkl', 'rb') as f:
    avg_score_slowfast2 = pickle.load(f)
with open('results/AUTSL_I3D_flow2_all.pkl', 'rb') as f:
    avg_score_flow2 = pickle.load(f)
with open('results/AUTSL_I3D_flow_all.pkl', 'rb') as f:
    avg_score_flow = pickle.load(f)
with open('results/AUTSL_I3D_BSL2_lhand_all.pkl', 'rb') as f:
    avg_score_lhand = pickle.load(f)
with open('results/AUTSL_I3D_BSL2_rhand_all.pkl', 'rb') as f:
    avg_score_rhand = pickle.load(f)
with open('results/AUTSL_I3D_BSL2_face_all.pkl', 'rb') as f:
    avg_score_face = pickle.load(f)
with open('results/AUTSL_sgn_pose2_all.pkl', 'rb') as f:
    avg_score_pose = pickle.load(f)
with open('results/AUTSL_sgn_pose_all.pkl', 'rb') as f:
    avg_score_pose2 = pickle.load(f)


with open('datasets/template_test.csv', 'r') as f:
    template_sample = {}
    for line in f.readlines():
        name = line.split(',')[0]
        template_sample[name] = -1

true = 0.0
total = 0.0
f = F.softmax

for video_name, video_info in avg_score_i3d.items():
    target, pred_rgb, _ = video_info

    pred = (pred_rgb + avg_score_i3d2[video_name][1] * 1.0) / 2.0 * 2.0 \
           + (avg_score_slowfast[video_name][1] + avg_score_slowfast2[video_name][1]) / 2.0 \
           + (avg_score_lhand[video_name][1] + avg_score_rhand[video_name][1]) * 2.0 \
           + avg_score_face[video_name][1] * 4.0 \
           + (avg_score_flow[video_name][1] + avg_score_flow2[video_name][1]) / 2.0 * 3.0 \
           + (avg_score_pose[video_name][1] + avg_score_pose2[video_name][1]) / 2.0 * 4.0 \

    template_sample[video_name] = torch.argmax(pred).item()
    pred_softmax = F.softmax(pred, 1)

    if target == torch.argmax(pred):
        true += 1
    else:
        pass
    total += 1
print('Top1:', true/total)
with open('predictions_fusion-final-RGB.csv', 'w') as f2:
    for k, v in template_sample.items():
        line = k + ',' + str(v) + '\n'
        f2.writelines(line)
