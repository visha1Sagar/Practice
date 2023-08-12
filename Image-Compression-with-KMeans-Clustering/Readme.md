# Image-Compression-with-KMeans-Clustering

This Python script uses the K-Means clustering algorithm from the scikit-learn library to compress an input image.

## Installation
This script requires the following Python packages:

- scikit-learn
- Pillow
- NumPy

You can install these packages using pip:

 ``` pip install scikit-learn Pillow numpy ```

## Usage
To use this script, follow these steps:

1. Open the script in your preferred Python editor or IDE.
2. Change the path in the ```img = Image.open("cow.jpg")``` line to the path of your input image.
3. Change the number of clusters in the ```n_clusters = 3``` line to the desired number of colors in the compressed image.
4. Run the script.
The script will apply K-Means clustering to the pixel values in the input image, quantize the pixel values by replacing each pixel with the centroid of its corresponding cluster, and save the compressed image as a new JPEG file.

The filename of the compressed image can be changed by modifying the ```quantized_img.save("compressed_image.jpg", quality=100)``` line.
