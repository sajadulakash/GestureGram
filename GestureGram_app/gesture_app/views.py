from django.shortcuts import render

import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Create your views here.

def gesture_detection_view(request):
    # Load model and labels
    model = load_model("model.h5")
    labels = np.load("labels.npy")

    # Initialize MediaPipe holistic model
    holistic = mp.solutions.holistic.Holistic()

    # Access webcam
    cap = cv2.VideoCapture(0)

    # Process frames in a loop
    while True:
        ret, frame = cap.read()

        # Process frame with MediaPipe
        results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Perform gesture detection logic (similar to your provided code)
        ...

        # Convert frame to JPEG format for display in the browser
        frame = cv2.imencode('.jpg', frame)[1].tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # Release resources
    cap.release()
