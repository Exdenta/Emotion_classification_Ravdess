# Emotion_classification_Ravdess
Real-time face detection and emotion classification using Ravdess dataset. Tensorflow, Keras, OpenCV.

## Setup virtual environment (Optional)
```powershell
pip install virtualenv
py -m venv env
.\env\Scripts\activate
```

## Prerequisites

- Install [CMake](https://cmake.org/download/) (to build dlib), make sure it's in the system PATH.
- [Install](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html) CUDA `v9.0` and cuDNN `v7.1.4` for CUDA `v9.0` (to run tensorflow model on gpu).
- Install requirements:
```powershell
pip install -r requirements.txt 
```

[Here](https://www.tensorflow.org/install/source_windows#gpu) you can find a compatible version of tensorflow-gpu if you have a different version of CUDA and cuDNN.

# Run pretrained emotion classificator

Runs emotion detector with default parameters (webcam, base model, etc.)
```powershell
python .\emotion_detector.py
```

You can also specify --input that can be camera, image or video. 

1. In camera mode you can specify its number (--camera_number), output directory, where results will be saved (--output_dir), emotion classification model path (--model), threshold for face detector (--conf_threshold). Example:
```powershell
python .\emotion_detector.py --camera_number 1 --output_dir 'D:\\Results' --model 'path_to_model' --conf_threshold 0.9
``` 

2. In image and video mode you can also specify the path to the image or video (--source). Example:
```powershell
python .\emotion_detector.py --input image --source 'path_to_image'
``` 

# Prepare data to train emotion classificator

## 1. Load data
* Load all archives `Video_Speech_Actor_xxx.zip` from [Ravdess](https://zenodo.org/record/1188976#.XQf5d4gvOUm) dataset (24 actors). 
* Unzip and put all videos that contain no sounds (name starts with '02') into 2 folders: 
    * 23 actors - training (`dataset/TrainingVideos`)
    * 1 actor - test (`dataset/TestVideos`)

## 2. Convert into frames
Run script `preprocessing/video2frames.py` that will convert all videos from `dataset/TrainingVideos` and `dataset/TestVideos` into frames and will save them in the folders `dataset/TrainingFrames` and `dataset/TestFrames` accordingly.

```powershell
python preprocessing/video2frames.py
```

For each video will be created a folder that contains all frames and annotation file with emotion number. 

## 3. Convert frames into dataset (pickle file)
Run `preprocessing/frames2pickle.py` to form training and test dataset with default parameters or open `preprocessing/Frames2Pickle.ipynb` in Jupyter Notebook to follow data preparation step by step.

```powershell
python preprocessing/frames2pickle.py
```

# Train emotion classificator

```powershell
python train.py
```

# Test your emotion classificator

Run jupyter notebook file `test.ipynb` to evaluate your model
