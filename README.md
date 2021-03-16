# Code for CVPR2021 ChaLearn Challenge of ISLR


## 1. Run Docker
Run the docker first and then execute all the following steps.
```bash
docker pull rhythmblue/chalearn2021:v1
docker run -it docker run -it --rm --shm-size=16G -v /xxx/ChaLearn-2021-ISLR-Challenge:/ChaLearn-2021-ISLR-Challenge rhythmblue/chalearn2021:v1
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

- Download the pretrained models and weights [GoogleDrive](), and organize them as follows,
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


## Precessing the data by yourself (If needed)

