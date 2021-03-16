#!/bin/bash
cd code

# slowfast-full
CUDA_VISIBLE_DEVICES=0 python main.py --root_path ../data/AUTSL --video_path jpg_video \
    --annotation_path autsl_01_final.json --result_path results_test \
    --dataset autsl --n_finetune_classes 226 --n_classes 226 --model slowfast \
    --batch_size 16 --n_threads 8 --test --test_subset val --sample_size 256 \
    --pretrain_path weights/results_slowfast_split_1/save_102.pth \
    --no_train --no_val --sample_duration 64
mv AUTSL_slowfast_all.pkl results/AUTSL_slowfast_all.pkl

CUDA_VISIBLE_DEVICES=0 python main.py --root_path ../data/AUTSL --video_path jpg_video \
    --annotation_path autsl_01_final.json --result_path results_test \
    --dataset autsl --n_finetune_classes 226 --n_classes 226 --model slowfast \
    --batch_size 16 --n_threads 8 --test --test_subset val --sample_size 256 \
    --pretrain_path weights/results_slowfast_final/save_34.pth \
    --no_train --no_val --sample_duration 64
mv AUTSL_slowfast_all.pkl results/AUTSL_slowfast2_all.pkl

# i3d-full
CUDA_VISIBLE_DEVICES=0 python main.py --root_path ../data/AUTSL --video_path jpg_video \
    --annotation_path autsl_01_final.json --result_path results_test \
    --dataset autsl --n_finetune_classes 226 --n_classes 226 --model I3D_BSL \
    --batch_size 1 --n_threads 4 \
    --test --test_subset test --sample_size 224 \
    --pretrain_path weights/results_i3d_bsl_split_1/save_38.pth \
    --no_train --no_val --sample_duration 64 
mv AUTSL_I3D_BSL_all.pkl results/AUTSL_I3D_BSL_all.pkl

CUDA_VISIBLE_DEVICES=0 python main.py --root_path ../data/AUTSL --video_path jpg_video \
    --annotation_path autsl_01_final.json --result_path results_test \
    --dataset autsl --n_finetune_classes 226 --n_classes 226 --model I3D_BSL \
    --batch_size 1 --n_threads 4 \
    --test --test_subset test --sample_size 224 \
    --pretrain_path weights/results_i3d_bsl_final/save_37.pth \
    --no_train --no_val --sample_duration 64 
mv AUTSL_I3D_BSL_all.pkl results/AUTSL_I3D_BSL2_all.pkl

# i3d-flow
CUDA_VISIBLE_DEVICES=0 python main.py --root_path ../data/AUTSL --video_path jpg_video \
    --annotation_path autsl_01_final.json --result_path results_test \
    --dataset autsl --n_finetune_classes 226 --n_classes 226 --model I3D_flow \
    --batch_size 1 --n_threads 4 \
    --test --test_subset test --sample_size 224 \
    --pretrain_path weights/results_i3d_flow_split_1/save_42.pth \
    --no_train --no_val --sample_duration 64 
mv AUTSL_I3D_flow_all.pkl results/AUTSL_I3D_flow_all.pkl

CUDA_VISIBLE_DEVICES=0 python main.py --root_path ../data/AUTSL --video_path jpg_video \
    --annotation_path autsl_01_final.json --result_path results_test \
    --dataset autsl --n_finetune_classes 226 --n_classes 226 --model I3D_flow \
    --batch_size 1 --n_threads 4 \
    --test --test_subset test --sample_size 224 \
    --pretrain_path weights/results_i3d_flow_final/save_46.pth \
    --no_train --no_val --sample_duration 64 
mv AUTSL_I3D_flow_all.pkl results/AUTSL_I3D_flow2_all.pkl

# sgn-pose
CUDA_VISIBLE_DEVICES=0 python main.py --root_path ../data/AUTSL --video_path jpg_video \
    --annotation_path autsl_01_final.json --result_path results_test \
    --dataset autsl --n_finetune_classes 226 --n_classes 226 --model sgn_pose \
    --batch_size 32 --n_threads 4 \
    --test --test_subset test --sample_size 224 \
    --pretrain_path weights/results_mmpose_sgn_split_1_pre/save_184.pth \
    --no_train --no_val --sample_duration 32 
mv AUTSL_sgn_pose_all.pkl results/AUTSL_sgn_pose_all.pkl

CUDA_VISIBLE_DEVICES=0 python main.py --root_path ../data/AUTSL --video_path jpg_video \
    --annotation_path autsl_01_final.json --result_path results_test \
    --dataset autsl --n_finetune_classes 226 --n_classes 226 --model sgn_pose \
    --batch_size 32 --n_threads 4 \
    --test --test_subset test --sample_size 224 \
    --pretrain_path weights/results_sgn_final/save_94.pth \
    --no_train --no_val --sample_duration 32 
mv AUTSL_sgn_pose_all.pkl results/AUTSL_sgn_pose2_all.pkl


## i3d-face
## Change universal.py Line#447  /data/user/AUTSL/jpg_face/
## Face train I3D_BSL_part-AUTSL_Upper split
cp tmp/universal_face.py datasets/universal.py
CUDA_VISIBLE_DEVICES=0 python main.py --root_path ../data/AUTSL --video_path jpg_video \
    --annotation_path autsl_01_final.json --result_path results_test \
    --dataset autsl --n_finetune_classes 226 --n_classes 226 --model I3D_BSL_part \
    --batch_size 1 --n_threads 4 \
    --test --test_subset test --sample_size 224 \
    --pretrain_path weights/results_i3d_bsl_face_final/save_26.pth \
    --no_train --no_val --sample_duration 64 
mv AUTSL_I3D_BSL_part_all.pkl results/AUTSL_I3D_BSL2_face_all.pkl

## i3d-rhand
## Change universal.py Line#447  /data/user/AUTSL/jpg_right_hand/
## R-HAND train I3D_BSL_part-AUTSL_Upper split
cp tmp/universal_rhand.py datasets/universal.py
CUDA_VISIBLE_DEVICES=0 python main.py --root_path ../data/AUTSL --video_path jpg_video \
    --annotation_path autsl_01_final.json --result_path results_test \
    --dataset autsl --n_finetune_classes 226 --n_classes 226 --model I3D_BSL_part \
    --batch_size 1 --n_threads 4 \
    --test --test_subset test --sample_size 224 \
    --pretrain_path weights/results_i3d_bsl_rhand_final/save_18.pth \
    --no_train --no_val --sample_duration 64
mv AUTSL_I3D_BSL_part_all.pkl results/AUTSL_I3D_BSL2_rhand_all.pkl

## i3d-lhand
## Change universal.py Line#447  /data/user/AUTSL/jpg_left_hand/
## L-HAND train I3D_BSL_part-AUTSL_Upper split
cp tmp/universal_lhand.py datasets/universal.py
CUDA_VISIBLE_DEVICES=0 python main.py --root_path ../data/AUTSL --video_path jpg_video \
    --annotation_path autsl_01_final.json --result_path results_test \
    --dataset autsl --n_finetune_classes 226 --n_classes 226 --model I3D_BSL_part \
    --batch_size 1 --n_threads 4 \
    --test --test_subset test --sample_size 224 \
    --pretrain_path weights/results_i3d_bsl_lhand_final/save_39.pth \
    --no_train --no_val --sample_duration 64
mv AUTSL_I3D_BSL_part_all.pkl results/AUTSL_I3D_BSL2_lhand_all.pkl


### i3d-depth
CUDA_VISIBLE_DEVICES=0 python main.py --root_path ../data/AUTSL --video_path jpg_video \
    --annotation_path autsl_01_final.json --result_path results_test \
    --dataset autsl --n_finetune_classes 226 --n_classes 226 --model I3D_depth \
    --batch_size 1 --n_threads 4 \
    --test --test_subset test --sample_size 224 \
    --pretrain_path weights/results_i3d_depth_split_1/save_46.pth \
    --no_train --no_val --sample_duration 64
mv AUTSL_I3D_depth_all.pkl results/AUTSL_I3D_depth_all.pkl

CUDA_VISIBLE_DEVICES=0 python main.py --root_path ../data/AUTSL --video_path jpg_video \
    --annotation_path autsl_01_final.json --result_path results_test \
    --dataset autsl --n_finetune_classes 226 --n_classes 226 --model I3D_depth \
    --batch_size 1 --n_threads 4 \
    --test --test_subset test --sample_size 224 \
    --pretrain_path weights/results_i3d_depth_final/save_35.pth \
    --no_train --no_val --sample_duration 64
mv AUTSL_I3D_depth_all.pkl results/AUTSL_I3D_depth2_all.pkl