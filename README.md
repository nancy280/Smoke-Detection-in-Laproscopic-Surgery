# Smoke-Detection-in-Laproscopic-Surgery

## Overview
This repository contains the implementation of a Convolutional Neural Network (CNN)-based model for detecting smoke in laparoscopic images. The model integrates multi-modal feature fusion by combining spatial image data with extracted numerical features such as Normalized-RGB, Wavelet energy, GLCM texture, HSV color space, Optical flow, and Fog Area. The proposed method aims to enhance real-time surgical visibility by accurately detecting smoke obstructions in laparoscopic video frames.

## Features
Real-time smoke detection in laparoscopic surgical videos
Multi-modal feature extraction to improve accuracy
Optimized CNN architecture for efficient processing
Preprocessing techniques such as normalization to handle lighting variations
Comparative analysis with other machine learning models like Random Forest, XGBoost, and SVM
Dataset
The dataset consists of frames extracted from 10 robot-assisted laparoscopic hysterectomy procedure videos obtained from the EPSRC Centre for Interventional and Surgical Sciences. Each frame is manually annotated for smoke regions, and features are extracted to aid in classification.

## Methodology

### Preprocessing:
Frames are extracted at 1 FPS from laparoscopic videos.
Image normalization is applied to standardize brightness and contrast.

### Feature Extraction:
Normalized-RGB (captures chromatic information)
Wavelet Energy (detects sharpness reduction due to smoke)
GLCM Texture (analyzes texture variations in smoke regions)
HSV Color Space (identifies desaturated smoke pixels)
Optical Flow (tracks smoke motion patterns)
Fog Area (detects accumulated smoke covering the scene)

### Model Architecture:
Convolutional layers extract spatial patterns from laparoscopic images.
Fully connected layers integrate numerical features with CNN-extracted features.
Sigmoid activation is used for binary classification (Smoke / No Smoke).

### Training and Evaluation:
The model is trained using Adam optimizer with binary cross-entropy loss.
Performance metrics include Accuracy, Precision, Recall, and F1-score.
Comparison with Random Forest, XGBoost, SVM, and alternative CNN variants.

###Installation

1. Clone the Repository
   
git clone https://github.com/nancy280/Smoke-Detection-in-Laproscopic-Surgery.git
cd Laparoscopic-Smoke-Detection

2. Install Dependencies
   
pip install -r requirements.txt

3. Run the Model

To train the model:
python train.py

To evaluate on test data:
python evaluate.py

4. Inference on New Data

To perform smoke detection on a new laparoscopic video:
python predict.py --input path/to/video.mp4


###Results
The proposed CNN achieves 92.6% accuracy, outperforming existing models:

Model	      Precision	Recall	F1-Score	Accuracy
Proposed CNN	0.93	  0.92	  0.91	    0.92
CNN with ReLU	0.74	  0.74	  0.74      0.74
CNN with Tanh	0.68	  0.68	  0.68	    0.68
Random Forest	0.80	  0.80	  0.80	    0.80
XGBoost	      0.80	  0.80	  0.80	    0.80
SVM	          0.75	  0.75	  0.75	    0.75

###Visualization
Feature Distributions
Correlation Matrix shows relationships between extracted features.
Histograms & Pair Plots highlight class separability.
Confusion Matrix provides a breakdown of classification results.






