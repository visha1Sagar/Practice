import streamlit as st
from PIL import Image
import numpy as np
from model_load import load_compile_model, decode_batch_predictions, preprocess_user_sentence, predict
st.set_page_config(page_title="Hand Written Text Recognition", layout="wide")



st.title("Image Processing App")

# st.image("sentence.png", use_column_width=True)

st.write("Loading Model..")
model = load_compile_model("saved_model.keras")
st.write("Model Loaded Successfully")


st.header("Upload Image")
image = st.file_uploader('File uploader')

if image:
    st.image(image, use_column_width=True)
    st.write('Image Uploaded Successfully')
    st.write('Processing Image...')

    image = Image.open(image)
    img_array = np.array(image)
    st.write(img_array.shape)
    images = preprocess_user_sentence(img_array)
    st.write('Image Processed Successfully')

    st.write('Predicting the text...')
    prediction = predict(model, images)
    st.write('Prediction :', prediction)
    st.write('Prediction Complete')




def select_random_image():
    st.header("Select Random Image")
    image = st.file_uploader('File uploader')



# st.button('Upload Image', on_click=upload_image)
st.button('Select Random Image', on_click=select_random_image)


