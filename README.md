# Fashion MNIST Clothing Classifier

Deep Learning based Fashion MNIST image classifier built using Convolutional Neural Networks (CNN) and PyTorch.

## Project Overview

This project uses a Convolutional Neural Network (CNN) to classify images from the Fashion-MNIST dataset into one of 10 clothing categories.

The model is trained using PyTorch and achieves high classification accuracy on unseen test images.

## Features

- CNN-based image classification
- Fashion-MNIST dataset support
- PyTorch implementation
- GPU support (if CUDA is available)
- Training and testing pipeline

## Dataset

Fashion-MNIST contains:

- 60,000 training images
- 10,000 testing images
- 10 clothing categories

Classes:

1. T-shirt/Top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle Boot

## Tech Stack

- Python
- PyTorch
- TorchVision
- Deep Learning
- CNN

## Model Architecture

Input Image (28×28)

↓ Conv2D (32 Filters)

↓ ReLU

↓ Max Pooling

↓ Conv2D (64 Filters)

↓ ReLU

↓ Max Pooling

↓ Fully Connected Layer (128)

↓ Output Layer (10 Classes)

## Installation

```bash
git clone https://github.com/uday-gautam04/fashion-mnist-clothing-classifier.git

cd fashion-mnist-clothing-classifier

pip install -r requirements.txt
```

## Run Project

```bash
python fashion_mnist_classifier.py
```

## Expected Results

- Training Loss decreases with each epoch
- Test Accuracy around 88–91%

## Future Improvements

- Hyperparameter tuning
- More epochs
- Data augmentation
- Web deployment using Flask or Streamlit

## Author

Uday Gautam

LinkedIn:
www.linkedin.com/in/uday-gautam04
