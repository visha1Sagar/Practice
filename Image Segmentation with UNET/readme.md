# Image Segmentation with U-Net
## Project Description:

This project implements image segmentation using a U-Net architecture. It utilizes the DUTS-TE dataset for training and aims to segment foreground objects from background images.

## Dependencies:

- TensorFlow
- Keras
- NumPy
- Matplotlib


## Data:

The project uses the `DUTS-TE` dataset, downloaded and unzipped within the notebook.

## Hyperparameters:

- Image size: 256x256 pixels
- Batch size: 8
- Output classes: 1 (foreground or background)
- Train/validation split: 90%/10%

## Model:

The model utilizes a U-Net architecture with encoder-decoder blocks for downsampling and upsampling features.
Encoder blocks use `Conv2D` layers with ReLU activation and dropout for regularization.
Decoder blocks use `Conv2DTranspose` layers for upsampling and concatenate features from corresponding encoder blocks.
The final layer uses a Conv2D layer with sigmoid activation for binary classification.
Training:

The model is trained using the Adam optimizer with a learning rate of `1e-2` for `30 epochs`.
Mean squared error (MSE) is used as the loss function, and accuracy is monitored as a metric.
Other loss function were also tried but MSE was found good.


## Evaluation:

Loss curves for training and validation sets are visualized to track model performance.
Predictions on validation images are displayed to qualitatively evaluate segmentation results.
