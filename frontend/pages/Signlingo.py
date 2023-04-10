import joblib
from tensorflow import keras
from PIL import Image

import streamlit as st
import cv2
import numpy as np
import os
import mediapipe as mp

mphands = mp.solutions.hands
hand = mphands.Hands(static_image_mode=True, max_num_hands=2,
                     min_detection_confidence=0.75)
mpdraw = mp.solutions.drawing_utils
script_dir = os.path.dirname(__file__)
# image = Image.open('./assets/legend.jpeg')
image = Image.open(os.path.join(script_dir, '../assets/legend.jpeg'))
header = Image.open(os.path.join(script_dir, '../assets/signlingo-header.png'))
logo = Image.open(os.path.join(script_dir, '../assets/signlingo.png'))
resized_logo = logo.resize((150, 150))

st.set_page_config(layout="centered")
st.image(header)
# st.title("  ✌️Real Time Sign Language Detection 🤟")
# st.image(resized_logo)
st.title(" :orange[ Sign Language Alphabets Game] ")
# models = ["CNN", "SVM", "KNN"]
# choice = st.selectbox("Select model to use.", models)
choice = "CNN"
st.subheader("Instructions")
st.text("This game is to help you learn the Singapore Sign Language alphabets.")
st.text("Try gesturing the alphabet displayed and check if you are right or wrong.")
st.text("You can shuffle if the letter is too hard for you.")
st.text("Tick ☑ the checkbox 'Play' below to start playing the game and uncheck ☐ to stop.")

st.sidebar.title("Guide")
st.sidebar.image(image, caption='Singapore Sign Language Guide',
                 use_column_width=True)

# all_classes = os.listdir("C:/Users/harsh/Downloads/ASL")
all_alphabets = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k',
                 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y']
quiz_alphabets = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
                  'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 'u', 'v', 'w', 'x', 'y']


def random_alphabet():
    return quiz_alphabets[np.random.randint(0, len(quiz_alphabets))]


if ('current_alphabet' not in st.session_state):
    st.session_state['current_alphabet'] = random_alphabet()

# Initialize mediapipe hand
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
knn_model = joblib.load(os.path.join(script_dir, '../saved_models/knn.joblib'))
svm_model = joblib.load(os.path.join(script_dir, '../saved_models/svm.joblib'))
cnn_model = keras.models.load_model(
    os.path.join(script_dir, '../saved_models/cnn.h5'))

# scaler = joblib.load('./saved_models/standard_scaler.pkl')
scaler = joblib.load(os.path.join(
    script_dir, '../saved_models/standard_scaler.pkl'))

run = st.checkbox(' 👈 Play the game!')
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)


def load_model(model_name):
    # if(model_name == 'KNN'):
    #   return knn_model
    # elif(model_name == 'SVM'):
    #   return svm_model
    # else:
    return cnn_model


def process_landmarks(model_name, data):
    # if(model_name == 'KNN'):
    #   coords = list(np.array([[landmark.x, landmark.y] for landmark in data]).flatten())
    #   coords = scaler.transform([coords])
    #   return coords
    # elif(model_name == 'SVM'):
    #   coords = [list(np.array([[landmark.x, landmark.y] for landmark in data]).flatten())]
    #   return coords
    # else:
    coords = list(np.array([[landmark.x, landmark.y]
                  for landmark in data]).flatten())
    coords = scaler.transform([coords])
    return coords


def process_output(model_name, output):
    # if(model_name == 'KNN'):
    #   return (output[0].upper())
    # elif(model_name == 'SVM'):
    #   return (all_alphabets[int(output[0])]).upper()
    # else:
    return (all_alphabets[np.argmax(output[0], axis=0)]).upper()


def checkans(frame, current_alphabet):
    x_points = []
    y_points = []
    # _, frame = camera.read()
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

        if len(x_points) == 21 or len(x_points) == 42:
            target = frame[a1[1]:a1[0], a2[1]:a2[0]]

            if len(target) > 0:
                coords = process_landmarks(choice, landmarks.landmark)
                chosen_model = load_model(choice)
                predicted = chosen_model.predict(coords)
                print(predicted)

                ans = process_output(choice, predicted)

                if (ans == current_alphabet.upper()):
                    st.balloons()
                    st.success("Correct!")
                    st.session_state['current_alphabet'] = random_alphabet()
                else:
                    st.error("Wrong!")
                    st.write("The letter you gestured is: " + ans +
                             "but the correct letter is: " + current_alphabet)

# Define a function to capture an image from the camera


def capture_image():
    # camera.release()
    run = False
    # Initialize the camera
    cap = cv2.VideoCapture(0)
    # Capture a frame from the camera
    ret, frame = cap.read()
    # Release the camera
    cap.release()

    checkans(frame, st.session_state['current_alphabet'])
    # Return the captured frame
    return frame

# Define a callback function to handle mouse clicks


def on_click():
    # Capture an image from the camera
    image = capture_image()
    # Display the captured image using st.image
    st.image(image, channels="BGR")


if run is False:
    camera.release()

if run is True:
    captureButton = st.button('Capture Image')
    shuffleButton = st.button('Shuffle')

# Create a button to capture an image from the camera
if run is True and captureButton:
    # Capture an image from the camera
    image = capture_image()
    # Display the captured image using st.image
    st.image(image, channels="BGR")

if run is True and shuffleButton:
    st.session_state['current_alphabet'] = random_alphabet()


if run is True:
    st.write("🚩 Current Alphabet: ",
             st.session_state['current_alphabet'].upper())


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

        if len(x_points) == 21 or len(x_points) == 42:
            target = frame[a1[1]:a1[0], a2[1]:a2[0]]

            if len(target) > 0:
                coords = process_landmarks(choice, landmarks.landmark)
                chosen_model = load_model(choice)
                predicted = chosen_model.predict(coords)
                print(predicted)

                ans = process_output(choice, predicted)
                cv2.putText(frame, ans, (80, 80),
                            cv2.FONT_ITALIC, 2, (255, 191, 0), 4)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame_rgb)


else:
    st.write('🛑 Game is stopped!')
    camera.release()
