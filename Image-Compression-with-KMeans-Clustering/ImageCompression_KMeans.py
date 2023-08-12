from sklearn.cluster import KMeans # import KMeans from scikit-learn's cluster module
from PIL import Image # import Image module from Python's PIL library
import numpy as np # import numpy for array manipulation

# Load the image and convert it to a numpy array
img = Image.open("cow.jpg") # change "tiger.png" to the path of your image
img_arr = np.array(img)

# Reshape the image so that each pixel is a row in a 2D array
rows, cols, channels = img_arr.shape
pixels = img_arr.reshape(rows * cols, channels)

# Apply K-means clustering to the pixel values
n_clusters = 3 # choose the number of clusters (i.e. the number of colors in the compressed image)
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(pixels)

# Quantize the pixel values by replacing each pixel with the centroid of its corresponding cluster
quantized_pixels = kmeans.cluster_centers_[kmeans.labels_]

# Reshape the quantized pixel array to match the original image shape
quantized_img_arr = quantized_pixels.reshape(rows, cols, channels)

# Save the compressed image as a new JPEG file
quantized_img = Image.fromarray(quantized_img_arr.astype('uint8'), mode='RGB')
quantized_img.save("compressed_image.jpg", quality=100) # change "compressed_image.jpg" to the filename you want to use