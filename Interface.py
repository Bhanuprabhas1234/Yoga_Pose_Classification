# Import necessary libraries

from flask import Flask, render_template, Response, redirect, url_for
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

import threading
import time

app = Flask(__name__)
model = load_model("model/CNN-LSTM9.h5")

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
pose_classes = ['catpose','downdog','goddess','natrajasana','navasana', 'plank', 'tree','trikonasana','unknown','warrior2']

cap = None

# Initialize pyttsx3 engine once
# engine = pyttsx3.init()

def extract_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if results.pose_landmarks:
        landmarks = []
        for lm in results.pose_landmarks.landmark:
            landmarks.append([lm.x, lm.y, lm.z, lm.visibility])
        return np.array(landmarks), results
    return None, None

def are_landmarks_reliable(visibilities, visibility_threshold=0.5):
    left=True
    for i in range(11,33,2):
        if visibilities[i] < visibility_threshold:
            left=False
            break
    right=True
    for i in range(12,33,2):
        if visibilities[i] < visibility_threshold:
            right=False
            break
    return right or left

confidence_threshold = 0.90
        
def generate_frames():
    global cap
    cap = cv2.VideoCapture(0)
    last_pose_name = ""
    last_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        landmarks=[]
        visibilities=[]
        results = pose.process(image_rgb)


        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])
                visibilities.append(lm.visibility)
            if are_landmarks_reliable(visibilities):
                landmarks = np.array(landmarks)
                landmarks = landmarks.reshape(-1,1,33,3)
                prediction = model.predict(landmarks)
                class_idx = np.argmax(prediction)
                confidence = np.max(prediction)
                if confidence > confidence_threshold:
                    pose_name = pose_classes[class_idx]
                    if pose_name != "unknown":
                        cv2.putText(frame, f'Pose: {pose_name}, Confidence: {confidence:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    else:
                        cv2.putText(frame, f'Unknown pose detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(frame, f'No confident pose detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, f'All Landmarks not visible', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, f'No pose detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        # Display the output frame
    

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/capture')
def capture():
    return render_template('capture.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_video')
def stop_video():
    global cap
    if cap:
        cap.release()
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=False,threaded=True)