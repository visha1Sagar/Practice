import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2

def ctc_loss(y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)

    return loss


def load_compile_model(model_path= "saved_model.keras"):
    model = tf.keras.models.load_model(model_path, compile=False)

    # Optimizer.
    opt = keras.optimizers.Adam(lr =1e-4)

    # Compile the model
    model.compile(optimizer=opt, loss=ctc_loss)


    return model


img_size = (128, 32)

def distortion_free_resize(image, img_size=img_size):
    w, h = img_size
    image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)

    # Check tha amount of padding needed to be done.
    pad_height = h - tf.shape(image)[0]
    pad_width = w - tf.shape(image)[1]

    # Only necessary if you want to do same amount of padding on both sides.
    if pad_height % 2 != 0:
        height = pad_height // 2
        pad_height_top = height + 1
        pad_height_bottom = height
    else:
        pad_height_top = pad_height_bottom = pad_height // 2

    if pad_width % 2 != 0:
        width = pad_width // 2
        pad_width_left = width + 1
        pad_width_right = width
    else:
        pad_width_left = pad_width_right = pad_width // 2

    image = tf.pad(
        image,
        paddings=[
            [pad_height_top, pad_height_bottom],
            [pad_width_left, pad_width_right],
            [0, 0],
        ],
    )

    image = tf.transpose(image, perm=[1, 0, 2])
    image = tf.image.flip_left_right(image)
    return image



# A utility function to decode the output of the network.
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search.
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :max_len
    ]
    # Iterate over the results and get back the text.
    output_text = []
    for res in results:
        res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text


def preprocess_user_sentence(image1):
    def remove_padded_255(arr):
        """
        Removes all padded 255 numbers in a 2d array and reshapes to its new shape.
        """
        # Find non-zero rows
        if len(arr.shape) >=3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        arr = arr.copy() # Make copy so that it should affect actual image

        non_zero_rows = np.any(arr < 247, axis=1)
       
        
        # Filter the array based on non-zero rows
        filtered_arr = arr[non_zero_rows]
        
        # Reshape the filtered array
        return filtered_arr.reshape(-1, *filtered_arr.shape[1:])

    filtered_arr = remove_padded_255(image1)
    

    def find_w_streaks(array):
        """ This function finds streaks of "w" columns (all 255 values) in a 2D array. """
        streaks = [(0,3)]
        start_index = None
        for col in range(array.shape[1]):
            if np.all(array[:, col] > 240):
                if start_index is None:
                    start_index = col
            else:
                if start_index is not None :
                    
                    if start_index!= 0  and ((col - 1 -start_index) > 3 ):
                        streaks.append((start_index, col - 1))
                    
                    start_index = None

        streaks.append((array.shape[1] -3, array.shape[1]))
        return streaks

    w_streaks = find_w_streaks(filtered_arr)


    images = []
    for t1, t2 in zip(w_streaks, w_streaks[1:]):
        image = filtered_arr[:, t1[1]-3:t2[0]+3]
        image = np.expand_dims(image, axis=-1)
        image = distortion_free_resize(image)
        image = image.numpy()
        
        image = (image/255.0).clip(0, 1).astype(np.float16)
        images.append(image)
    images = np.array(images)

    return images


def predict(model, images):


  pred = model.predict(images)
  pred_texts = decode_batch_predictions(pred)
  return " ".join(pred_texts)

