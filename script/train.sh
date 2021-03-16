# train slowfast-AUTSL_Upper split
python main.py --root_path ../data/user/AUTSL --video_path jpg_video \
    --annotation_path autsl_01_split.json --result_path results_test \
    --dataset autsl --n_classes 400 --n_finetune_classes 226 --weight_decay 0.0001 --learning_rate 0.005 \
    --pretrain_path pre_trained/i3d_r50_kinetics.pth --model slowfast --batch_size 16 --n_threads 8 \
    --checkpoint 1 --sample_duration 64 

# train I3D_BSL-AUTSL_Upper split
python main.py --root_path ../data/user/AUTSL --video_path jpg_video \
    --annotation_path autsl_01_split.json \
    --result_path results_test --dataset autsl --n_classes 1064 --n_finetune_classes 226 --weight_decay 0.0001 \
    --learning_rate 0.01 --pretrain_path pre_trained/i3d_bsl_model.pth.tar \
    --ft_begin_index 4 --model I3D_BSL  --batch_size 4 --n_threads 8 \
    --checkpoint 1 --sample_duration 64


# train I3D_flow-AUTSL_Upper split
python main.py --root_path ../data/AUTSL --video_path jpg_video \
    --annotation_path autsl_01_final.json \
    --result_path results_test --dataset autsl --n_classes 400 --n_finetune_classes 226 --weight_decay 0.0001 \
    --learning_rate 0.01 --pretrain_path pre_trained/model_flow.pth \
    --ft_begin_index 4 --model I3D_flow  --batch_size 4 --n_threads 8 \
    --checkpoint 1 --sample_duration 64

# train SGN-AUTSL_Upper split
python main.py --root_path ../data/user/AUTSL --video_path jpg_video \
    --annotation_path autsl_01_final.json \
    --result_path results_test --dataset autsl --n_classes 500 --n_finetune_classes 226 --weight_decay 0.0001 \
    --learning_rate 0.001 --pretrain_path pre_trained/sgn_slr500_save_13.pth \
    --ft_begin_index 4 --model sgn_pose  --batch_size 64 --n_threads 8 \
    --checkpoint 1 --sample_duration 32

# train I3D_depth-AUTSL_Upper split
python main.py --root_path ../data/user/AUTSL --video_path jpg_video \
    --annotation_path autsl_01_split.json \
    --result_path results_test --dataset autsl --n_classes 400 --n_finetune_classes 226 --weight_decay 0.0001 \
    --learning_rate 0.01 --pretrain_path pre_trained/model_flow.pth \
    --ft_begin_index 4 --model I3D_depth  --batch_size 4 --n_threads 8 \
    --checkpoint 1 --sample_duration 64

# train I3D_BSL_part-AUTSL_Upper split
python main.py --root_path ../data/user/AUTSL --video_path jpg_video \
    --annotation_path autsl_01_final.json \
    --result_path results_test --dataset autsl --n_classes 1064 --n_finetune_classes 226 --weight_decay 0.0001 \
    --learning_rate 0.01 --pretrain_path pre_trained/i3d_bsl_model.pth.tar \
    --ft_begin_index 4 --model I3D_BSL_part  --batch_size 4 --n_threads 8 \
    --checkpoint 1 --sample_duration 64