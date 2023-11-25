import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_v3 import preprocess_input

# Load the pre-trained model
model = load_model('best_model_chair.h5', compile=False)

target_img_shape = (128, 128)
camera_index = 0  # Use 0 for the default camera

cap = cv2.VideoCapture(camera_index)

st.title("Chair Classification App")

# Function to classify chair and display result
def classify_chair(img):
    x = 0
    y = 0
    w = 500
    h = 500

    imgcut = img.copy()[y:y + h, x:x + w]

    img_input = cv2.resize(imgcut, target_img_shape)

    img_input = img_to_array(img_input)
    img_input = preprocess_input(img_input)

    img_input = np.expand_dims(img_input, axis=0)  # (1, 128, 128, 3)

    result = model.predict(img_input)

    class_answer = np.argmax(result, axis=1)
    if class_answer == 0:
        predict = 'broken'
    elif class_answer == 1:
        predict = 'good'

    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 5)

    conf = int(np.max(result) * 100)
    if conf >= 75:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, predict + ' ' + str(conf) + '%', (160, 80), font, 2, (0, 200, 200), 5)

    return img

# Streamlit app
while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)

    st.image(classify_chair(img), channels="BGR", use_column_width=True)
