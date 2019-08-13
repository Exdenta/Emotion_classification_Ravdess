# Emotion_classification_Ravdess
Real-time face detection and emotion classification using Ravdess dataset. Tensorflow, Keras, OpenCV.

## Setup virtual environment (Optional)
```
pip install virtualenv
py -m venv env
.\venv\Scripts\activate
```

## Install 3rd-party libraries
```
pip install -r requirements.txt
```

# Run emotion detector


# Train emotion classificator

## 1. Load data
- Load all archives `'Video_Speech_Actor_xxx.zip'` from [Ravdess](https://zenodo.org/record/1188976#.XQf5d4gvOUm "Title") dataset (24 actors). 
- Unzip and put all videos that contain no sounds (name starts with '02') into 2 folders: one for training - `'../Emotion_classification_Ravdess/dataset/TrainingVideos'` with 23 actors and second for the test - `'../Emotion_classification_Ravdess/dataset/TestVideos'` with 1 actor to validate the results of training.

## 2. Convert into frames
Run script `'preprocessing/video2frames.py'` that will convert all videos from `'../TrainingVideos'` and `'../TestVideos'` to frames and will save them into folder `'../TrainingFrames'` and `'../TestFrames'` accordingly.

```
python preprocessing/video2frames.py
```

For each video will be created a folder that contains all frames and annotation file with emotion number. 

## 3. Convert frames into dataset (pickle file)
Run `'preprocessing/frames2pickle.py'` to form training and test dataset with default parameters or open `'preprocessing/Frames2Pickle.ipynb'` in Jupyter Notebook to follow data preparation step by step.

```
python preprocessing/frames2pickle.py
```

