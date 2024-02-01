from flask import Flask, render_template, Response
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load your model and labels (adjust paths as needed)
model = load_model("model.h5")
labels = np.load("labels.npy")

# Initialize MediaPipe holistic model
holistic = mp.solutions.holistic.Holistic()

# Define a function for gesture detection and frame processing
def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Process frame with MediaPipe
        results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # ... (rest of your gesture detection logic) ...

        # Encode frame as JPEG for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')  # Replace with your HTML template

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)  # Enable debug mode for easier development
