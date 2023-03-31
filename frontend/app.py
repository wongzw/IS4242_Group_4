# import streamlit as st
# from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import joblib
from tensorflow import keras
from PIL import Image


# #load model, set cache to prevent reloading
# @st.cache(allow_output_mutation=True)
# def load_model():
#     loaded_model = joblib.load('model.joblib')
#     # model=tf.keras.models.load_model('models/basic_model.h5')
#     return loaded_model


# with st.spinner("Loading Model...."):
#     model=load_model()

# modelF= keras.models.load_model('rec_0.h5')

import streamlit as st
import cv2
import numpy as np
import os
import mediapipe as mp

mphands = mp.solutions.hands
hand = mphands.Hands(static_image_mode =True, max_num_hands =2, min_detection_confidence=0.75)
mpdraw = mp.solutions.drawing_utils
script_dir = os.path.dirname(__file__)
# image = Image.open('./assets/legend.jpeg')
image = Image.open(os.path.join(script_dir, 'assets/legend.jpeg'))
header = Image.open(os.path.join(script_dir, 'assets/header.png'))
 
st.set_page_config(layout="centered")
st.image(header)
# st.title("  ✌️Real Time Sign Language Detection 🤟")
st.title(" Real Time Sign Language Detection ")
st.subheader(":orange[ ☑ Tick the checkbox 'Run' below to start detecting and ☐ uncheck to stop.]")

models = ["CNN", "SVM", "KNN"]
choice = st.sidebar.selectbox("Select model to use.", models)
st.sidebar.title("Guide")
st.sidebar.image(image, caption='Singapore Sign Language Guide', use_column_width=True)

# all_classes = os.listdir("C:/Users/harsh/Downloads/ASL")
all_alphabets = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y']

# Initialize mediapipe hand

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
# knn_model = joblib.load('./saved_models/knn.joblib')
# svm_model = joblib.load('./saved_models/svm.joblib')
# cnn_model = keras.models.load_model('./saved_models/cnn.h5')
knn_model = joblib.load(os.path.join(script_dir, 'saved_models/knn.joblib'))
svm_model = joblib.load(os.path.join(script_dir, 'saved_models/svm.joblib'))
cnn_model = keras.models.load_model(os.path.join(script_dir, 'saved_models/cnn.h5'))

# scaler = joblib.load('./saved_models/standard_scaler.pkl')
scaler = joblib.load(os.path.join(script_dir, 'saved_models/standard_scaler.pkl'))

run = st.checkbox('Run')
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)

def load_model(model_name):
  if(model_name == 'KNN'):
    return knn_model
  elif(model_name == 'SVM'):
    return svm_model
  else:
    return cnn_model
  
def process_landmarks(model_name, data):
  if(model_name == 'KNN'):
    coords = list(np.array([[landmark.x, landmark.y] for landmark in data]).flatten())
    coords = scaler.transform([coords])
    return coords
  elif(model_name == 'SVM'):
    coords = [list(np.array([[landmark.x, landmark.y] for landmark in data]).flatten())]
    return coords
  else:
    coords = list(np.array([[landmark.x, landmark.y] for landmark in data]).flatten())
    coords = scaler.transform([coords])
    return coords

def process_output(model_name, output):
  if(model_name == 'KNN'):
    return (output[0].upper())
  elif(model_name == 'SVM'):
    return (all_alphabets[int(output[0])]).upper()
  else:
    return (all_alphabets[np.argmax(output[0], axis=0)]).upper()

if run is False:
    camera.release()

while run:
    x_points = []
    y_points = []
    _, frame = camera.read()
    # Flip the image horizontally for a later selfie-view display, and convert the BGR image to RGB.
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hand.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
          for id, ld in enumerate(landmarks.landmark):
            h, w, channels = frame.shape
            x_points.append(int(ld.x * w))
            y_points.append(int(ld.y * h))

          a1 = (int(max(y_points) + 30), int(min(y_points) - 30))
          a2 = (int(max(x_points) + 30), int(min(x_points) - 30))
        cv2.rectangle(frame, (a2[1], a1[1]), (a2[0], a1[0]), (0, 255, 0), 3)

        if len(x_points) == 21 or len(x_points) == 42:
            target = frame[a1[1]:a1[0], a2[1]:a2[0]]

            if len(target) > 0:
              # print(landmarks.landmark)

              #here switch to different models
              # coords = [list(np.array([[landmark.x, landmark.y] for landmark in landmarks.landmark]).flatten())]
              coords = process_landmarks(choice, landmarks.landmark)
              # coords = scaler.transform([coords])
              chosen_model = load_model(choice)
              predicted = chosen_model.predict(coords)
              print(predicted)
              
              # Draw the hand annotations on the image.
              mpdraw.draw_landmarks(frame, landmarks, mphands.HAND_CONNECTIONS)

              #Show text predicted
              # cv2.putText(frame, (all_alphabets[int(predicted[0])]).upper() + " ", (80, 80), cv2.FONT_ITALIC, 2, (255, 100, 100), 2)
              cv2.putText(frame, process_output(choice, predicted), (80, 80), cv2.FONT_ITALIC, 2, (255, 100, 100), 2)


    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame_rgb)


else:
    st.write('Stopped')
    camera.release()
