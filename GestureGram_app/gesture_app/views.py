from django.http import StreamingHttpResponse
from django.shortcuts import render
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

def gesture_detection_view(request):
    def generate():
        model = load_model("model.h5")
        labels = np.load("labels.npy")
        holistic = mp.solutions.holistic.Holistic()
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if results.face_landmarks:
                lst = []
                for i in results.face_landmarks.landmark:
                    lst.append(i.x - results.face_landmarks.landmark[1].x)
                    lst.append(i.y - results.face_landmarks.landmark[1].y)
                for hand_results in [results.left_hand_landmarks, results.right_hand_landmarks]:
                    if hand_results:
                        for i in hand_results.landmark:
                            lst.append(i.x - hand_results.landmark[8].x)
                            lst.append(i.y - hand_results.landmark[8].y)
                    else:
                        lst.extend([0.0] * 42)
                lst = np.array(lst).reshape(1, -1)
                pred = labels[np.argmax(model.predict(lst))]
                cv2.putText(frame, pred, (50, 50), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)
                mp.solutions.drawing_utils.draw_landmarks(frame, results.face_landmarks, holistic.FACEMESH_CONTOURS)
                mp.solutions.drawing_utils.draw_landmarks(frame, results.left_hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                mp.solutions.drawing_utils.draw_landmarks(frame, results.right_hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

            frame = cv2.imencode('.jpg', frame)[1].tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        cap.release()

    return StreamingHttpResponse(generate())
