# MobileNetV2 Fine-Tuning with Data Augmentation on Shortened COVID-19 Dataset

## Purpose
This repository demonstrates fine-tuning a pre-trained **MobileNetV2** deep learning model for 4-class classification of chest X-ray images, using data augmentation techniques on a shortened version of the COVID-19 Radiography dataset.

## Dataset
- **Source**: COVID-19 / Chest X-ray Shortened Dataset (Kaggle).
- **Total Images**: 3,707 images.
- **Classes & Distribution**:
  - `COVID-19`: 576 images
  - `Normal`: 1,066 images
  - `pneumonia`: 1,013 images
  - `Lung Opacity`: 1,052 images

## Model Architecture
- **Base Model**: Pre-trained **MobileNetV2** (ImageNet weights).
- **Fine-Tuning Strategy**: Transfer learning and fine-tuning with custom classification head.
- **Custom Layers**: `GlobalAveragePooling2D`, `Dropout`, `BatchNormalization`, and `Dense` layers with Softmax activation for 4-class prediction.
- **Optimizer & Loss**: Adam / SGD optimizers with Categorical Cross-Entropy loss.
- **Callbacks**: `ModelCheckpoint`, `ReduceLROnPlateau`, and `EarlyStopping`.

## Data Augmentation Techniques
Data augmentation is applied via Keras `ImageDataGenerator` to improve model generalization and prevent overfitting:
- Rescaling (`1./255`)
- Random Rotation
- Width & Height Shifts
- Shear Range & Zoom Range
- Horizontal Flip

## Results
- The fine-tuned MobileNetV2 model effectively learns multi-class features for COVID-19, Normal, Pneumonia, and Lung Opacity detection on chest X-ray images.
- Data augmentation and learning rate scheduling help achieve stable convergence and high classification performance.
