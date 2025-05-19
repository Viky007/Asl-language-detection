README.txt for ASL & Hand-Gesture Recognition Project
=====================================================

Project Overview
----------------
This repository contains two key Python scripts:

1. **main 1.py**  
   - A data-preprocessing and model-training script that builds a convolutional neural network (CNN) to classify American Sign Language (ASL) alphabet images.  
   - Outputs:
     - Trained model saved as `asl_cnn_model2.keras`
     - Label encoder saved as `label_map.pkl`

2. **app.py**  
   - A real-time application that combines:
     - MediaPipe-based simple hand-gesture detection (thumbs up, open palm, fist, unknown)
     - ASL-alphabet recognition using the pretrained CNN model
     - Text output to `asl_output.txt` and optional speech via `pyttsx3`
   - Launches a window showing camera feed with overlays and speaks recognized letters/gestures.

Directory Structure
-------------------

Requirements
------------
- Python 3.7+
- Libraries:
  - `numpy`
  - `pandas`
  - `opencv-python`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`
  - `tensorflow` (>=2.x)
  - `mediapipe`
  - `pyttsx3`

Install dependencies with pip:
```bash
pip install numpy pandas opencv-python matplotlib seaborn scikit-learn tensorflow mediapipe pyttsx3
