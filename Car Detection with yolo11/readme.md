# Car Object Detection Using YOLO - README

This project demonstrates object detection on car images using the YOLO (You Only Look Once) model. It involves data preprocessing, model training, and object detection with YOLO, creating a gif with predicted bounding boxes.

## Prerequisites

Make sure the following libraries are installed:
- `os`, `random`
- `pandas`, `opencv-python`, `torch`, `tqdm`, `shutil`
- `IPython`, `matplotlib`, `ultralytics`, `imageio`

Use the following command to install missing libraries:
```bash
pip install -r requirements.txt
```

**Note**: `ultralytics` should be installed separately.
```bash
pip install ultralytics --quiet
```

## Dataset

Download the dataset from Kaggle:
```bash
kaggle datasets download -d sshikamaru/car-object-detection
```

## Data Preprocessing

- Normalize bounding boxes and transform data to YOLO-compatible format.
- Split data into training and validation sets.

## Training

1. Set up directories:
   - `datasets/training/` for training images and labels
   - `datasets/validation/` for validation images and labels

2. Write the `ds.yaml` configuration file:
   ```yaml
   train: training/images
   val: validation/images

   nc: 1
   names: ['car']
   ```

3. Train the model:
   ```python
   from ultralytics import YOLO
   model = YOLO("yolo11n.pt")
   train_results = model.train(data="ds.yaml", epochs=10, imgsz=640)
   ```

## Prediction

1. Load the trained model:
   ```python
   model = YOLO('runs/detect/train/weights/best.pt')
   ```

2. Perform predictions on validation images and create a GIF:
   ```python
   import imageio
   val_images_folder = 'datasets/validation/images'
   predicted_images = []

   for filename in os.listdir(val_images_folder):
       if filename.endswith(('.jpg', '.jpeg', '.png')):
           image_path = os.path.join(val_images_folder, filename)
           results = model(image_path)
           results[0].save('runs/detect/predict_gif.jpg')
           predicted_images.append(imageio.imread("runs/detect/predict_gif.jpg"))

   imageio.mimsave('predicted_objects.gif', predicted_images, fps=2)
   ```

## Results

The trained model can detect cars in images and draw bounding boxes around them. You can visualize the results by viewing `predicted_objects.gif`.

## Clean Up

Optionally, remove temporary files:
```bash
rm -rf data
```

Feel free to modify the training parameters, dataset paths, or models as needed.
