## A Structured Feature Learning Model for Clothing Keypoints Localization

### Requirements
- Python 3
- Pytorch >= 0.4.1
- torchvision

### Quick Start

#### 1. Download the datasets
* Deepfashion [[download](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/AttributePrediction.html)]
* FLD [[download](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/LandmarkDetection.html)]

#### 2. Create info.csv
```
python create_info.py
```
#### 3. Train
```
# 'root' and 'dataset' options are necessary.
python train_s.py --root [root_directory] --dataset [dataset_option]
```

#### 4. Evaluate
```
# You can run the file only for evaluation
python train_s.py --root [root_directory] --dataset [dataset_option] --evaluate True
```

--------------


