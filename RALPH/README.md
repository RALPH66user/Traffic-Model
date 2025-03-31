# Traffic Sign Recognition System

## Overview
This project implements a deep learning model for traffic sign recognition using the German Traffic Sign Recognition Benchmark (GTSRB) dataset. The system consists of two main components:
1. A training script (`traffic.py`) that builds and trains the neural network model
2. A graphical user interface (`predict_sign.py`) for uploading images and getting predictions

## Dataset
The German Traffic Sign Recognition Benchmark (GTSRB) dataset contains over 50,000 images of traffic signs across 43 different classes. The dataset includes various challenges such as different lighting conditions, occlusions, and varying perspectives.

## Project Structure
```
traffic-sign-recognition/
├── traffic.py            # Training script
├── predict_sign.py       # GUI for predictions
├── best_model.h5         # Saved trained model
├── requirements.txt      # Dependencies
└── README.md             # This file
```

## Installation
1. Clone the repository:
```bash
git clone https://github.com/RALPH66user/Traffic-Model.git
cd Traffic-Model
```

## Usage

### Training the Model
To train the model from scratch:
```bash
python traffic.py /path/to/dataset
```

This will:
- Load and preprocess the GTSRB dataset
- Train the neural network model
- Save the best performing model as `best_model.h5`

### Making Predictions
To predict traffic signs using the trained model:
```bash
python predict_sign.py
```

This opens a GUI where you can:
1. Upload an image of a traffic sign
2. View the predicted sign class
3. See the confidence score of the prediction

## Model Architecture
The current model architecture includes:
- Convolutional layers for feature extraction
- Max pooling layers for dimensionality reduction
- Dropout layers to prevent overfitting
- Dense layers for classification

## Dependencies
- TensorFlow/Keras
- OpenCV
- NumPy
- tkinter (for GUI)
- sklearn

## Future Improvements
Areas for potential improvement:
- Data augmentation to increase dataset diversity
- Transfer learning with pre-trained models
- Hyperparameter tuning
- Ensemble methods for better accuracy
