# GAN Image Generator

A Generative Adversarial Network (GAN) implementation for generating images using TensorFlow and Keras.

## Overview

This project contains the implementation of a GAN for generating images based on the MNIST dataset. The GAN consists of a generator and a discriminator, both implemented as Keras models.

## Features

- Build a generator to produce images from random noise.
- Build a discriminator to classify images as real or fake.
- Train the GAN on the MNIST dataset.
- Generate and save images using the trained generator.

## Requirements

- Python 3.x
- TensorFlow
- NumPy
- Matplotlib

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/gan-image-generator.git
   cd gan-image-generator

   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
3. Run the GAN:

    ```bash
    python gan.py

    ```

## Usage
- Modify the GAN architecture by adjusting parameters in the gan.py file.
- Experiment with different hyperparameters for training.

## Directory Structure
```bash
├── gan.py              # Main GAN implementation
├── README.md           # Project README
├── requirements.txt    # Project dependencies
├── images/             # Folder to store generated images
│   └── ...             # Generated images will be saved here
```
