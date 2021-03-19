# Code for CVPR2021 ChaLearn Challenge of ISLR

Our Team Ranked 3rd in [2021 ChaLearn LAP ISLR CVPR Challenge](http://chalearnlap.cvc.uab.es/challenge/43/description/). 

## 1. Run Docker
Run the docker first and then execute all the following steps.
```bash
docker pull rhythmblue/chalearn2021:v1
docker run -it --rm --shm-size=16G -v /xxx/ChaLearn-2021-ISLR-Challenge:/ChaLearn-2021-ISLR-Challenge rhythmblue/chalearn2021:v1
cd /ChaLearn-2021-ISLR-Challenge
```

## 2. Preprocessed data
- Download the preprocessed test data at [GoogleDrive](https://drive.google.com/drive/folders/1_1hqomdM0UsPMFAuBM4EEuUPt6zUZ2-g) and organize them as follows,

```
data/    
└── AUTSL
    ├── autsl_01_split.json
    ├── autsl_final.json 
    ├── flow          
    │   └── flow.zip  
    ├── jpg_face    
    │   └── face.zip  
    ├── jpg_left_hand   
    │   └── lhand.zip  
    ├── jpg_right_hand  
    │   └── rhand.zip 
    ├── jpg_video 
    │   └── full.zip
    └── Keypoints_2d_mmpose
        └── kps.zip    
```
- Unzip
```bash
cd /ChaLearn-2021-ISLR-Challenge
sh 1_unzip.sh
```

- Download the pretrained models and weights [GoogleDrive](https://drive.google.com/drive/folders/1KL0KQyvanNc_WsFU4D1Eg1zhMSt4aWXT), and organize them as follows,
```
code/                                 
├── pre_trained
├── weights
└── ...   
```

## 3. Inference
In this step, we need to run the commands in `2_inference.sh` in order.
```bash
cd /ChaLearn-2021-ISLR-Challenge
sh 2_inference.sh
```

## 4. Ensemble
Run the script to generate the `.csv` file.
```bash
cd /ChaLearn-2021-ISLR-Challenge
sh 3_ensemble.sh
```

## 5. Final Submission
- RGB: `predictions_fusion-final-RGB.csv`
- RGBD: `predictions_fusion-final-RGBD.csv`

## Training (If needed)
The detailed training script are provided in `script/train.sh`

## Precessing the data by yourself (If needed)
- Unzip the original videos in the `raw` folder.
```
preprocessing/data/AUTSL
├── first               
│   ├── test            
│   ├── train           
│   └── val             
├── flow                
│   ├── test            
│   ├── train           
│   └── val             
├── image               
│   ├── test            
│   ├── train           
│   └── val             
├── label               
│   ├── test_random.csv 
│   ├── train_labels.csv
│   └── val_random.csv  
└── raw                 
    ├── test            
    ├── train           
    └── val             
```

- Run docker
```bash 
docker pull rhythmblue/openpose:cuda11.1-cudnn8-v1
docker run -it docker run -it --rm --shm-size=16G -v /xxx/ChaLearn-2021-ISLR-Challenge:/ChaLearn-2021-ISLR-Challenge rhythmblue/openpose:cuda11.1-cudnn8
cd /ChaLearn-2021-ISLR-Challenge
```

- extract the first frame of each video to help localize signers. 
```bash
cd preprocessing
python preparation/1.gen_list.py
python preparation/2.extract_first.py data/AUTSL/raw data/AUTSL/first
```

- localize signer and crop the video. Pre-trained weights `preprocessing/preparation/weights/pose_hrnet_w48_384x288.pth` and `preprocessing/preparation/pre_data/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth` can be downloaded at [GoogleDrive](https://drive.google.com/drive/folders/1gS7kULCYqAd8F0z8ce456YywHhX1ILah?usp=sharing)
```bash
cd preprocessing
python preparation/3.localize_signer.py
python preparation/4.gen_ffmpeg_list.py
sh ffmpeg_folder_train.sh
sh ffmpeg_folder_val.sh
sh ffmpeg_folder_test.sh
sh ffmpeg_train.sh
sh ffmpeg_val.sh
sh ffmpeg_test.sh
```

- extract optical flow
```
cd preprocessing
python preparation/5.compute_flow.py
```

- extract keypoints with [DarkPose-Wholebody](https://github.com/open-mmlab/mmpose/tree/master/configs/wholebody/darkpose) in MMPose. Pre-trained weights of `mmpose/models/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth` and `mmpose/models/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth` can be downloaded at [GoogleDrive](https://drive.google.com/drive/folders/1gS7kULCYqAd8F0z8ce456YywHhX1ILah?usp=sharing)
```
python Stage1_top_down_SL_pose_video_AUTSL.py \
    demo/mmdetection_cfg/faster_rcnn_r50_fpn_1x_coco.py models/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    configs/top_down/darkpose/coco-wholebody/hrnet_w48_coco_wholebody_384x288_dark.py \
    models/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth 100
```

- Arrange data
```
python Stage2_All_Prepare_txt_file_w_val.py
python Stage3_n_frames_ucf101_hmdb51.py /data/user/AUTSL/jpg_video
python Stage4_AUTSL_json.py All_new
python Stage8_Crop_Hand.py
python Stage9_Crop_Face.py
```

## 
Hao Zhou, zhouh156(AT)mail.ustc.edu.cn