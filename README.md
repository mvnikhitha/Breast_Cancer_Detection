# Breast Cancer Detection using CNN

## Overview
This project focuses on detecting breast cancer using Convolutional Neural Networks (CNNs). The dataset used is the **CBIS-DDSM Breast Cancer Image Dataset**, and the model classifies images as **benign or malignant**.

## Features
- **Preprocessing**: Image resizing, grayscale conversion, and normalization.
- **Model Architecture**: CNN with multiple convolutional and pooling layers.
- **Training & Evaluation**: Uses binary cross-entropy loss with Adam optimizer.
- **Prediction**: Classifies breast cancer images as either benign or malignant.

## Dataset
The dataset is automatically downloaded from Kaggle using `kagglehub`. It contains mammogram images categorized into benign and malignant classes.

## Installation & Usage
### Prerequisites
Ensure you have the following installed:
- Python 3.x
- TensorFlow
- OpenCV
- NumPy
- scikit-learn

### Clone the Repository
```bash
git clone https://github.com/yourusername/breast-cancer-detection.git
cd breast-cancer-detection
```

### Install Dependencies
```bash
pip install tensorflow opencv-python numpy scikit-learn kagglehub
```

### Run the Model
```bash
python breast_cancer_detection.py
```

### Predict on New Images
Modify the script to test with custom images:
```python
print(predict_image("path_to_test_image.jpg"))
```

## Model Performance
The model achieves **high accuracy** on the test set. Performance can be improved with hyperparameter tuning and data augmentation.

## Contributing
Feel free to contribute by:
- Improving the model architecture.
- Enhancing preprocessing steps.
- Adding support for more datasets.

## License
This project is licensed under the MIT License.

## Acknowledgments
- [CBIS-DDSM Dataset](https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM)
- TensorFlow & Keras
- OpenCV for image processing

