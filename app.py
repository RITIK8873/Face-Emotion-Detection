from flask import Flask, render_template, Response
import cv2
from deepface import DeepFace
import os
import numpy as np

app = Flask(__name__)

# Load camera
camera = cv2.VideoCapture(0)

# Face database path
FACE_DB_PATH = "face_db"

# Emoji mapping
emotion_emoji = {
    'angry': '😡',
    'disgust': '🤢',
    'fear': '😨',
    'happy': '😊',
    'sad': '😢',
    'surprise': '😲',
    'neutral': '😐'
}

# Frame skipping for speed optimization
frame_skip_rate = 5
frame_count = 0
last_faces = []

def gen_frames():
    global frame_count, last_faces

    while True:
        success, frame = camera.read()
        if not success:
            break

        display_frame = frame.copy()
        frame_count += 1

        if frame_count % frame_skip_rate == 0:
            try:
                small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                result = DeepFace.analyze(small_frame, actions=['emotion'], enforce_detection=False)

                if isinstance(result, dict):
                    result = [result]

                faces = []

                for face in result:
                    # Scale coordinates back up
                    x = int(face['region']['x'] * 2)
                    y = int(face['region']['y'] * 2)
                    w = int(face['region']['w'] * 2)
                    h = int(face['region']['h'] * 2)

                    emotion = face['dominant_emotion']
                    emoji = emotion_emoji.get(emotion, '😶')

                    # Crop for recognition
                    face_crop = frame[y:y+h, x:x+w]

                    try:
                        recog = DeepFace.find(face_crop, db_path=FACE_DB_PATH, enforce_detection=False)
                        person_name = "Unknown"
                        if len(recog) > 0 and not recog[0].empty:
                            person_name = os.path.basename(os.path.dirname(recog[0].iloc[0]['identity']))
                    except:
                        person_name = "Unknown"

                    faces.append((x, y, w, h, person_name, emotion, emoji))

                last_faces = faces

            except Exception as e:
                print(f"Error in DeepFace analysis: {e}")

        # Draw last known face data
        for (x, y, w, h, name, emotion, emoji) in last_faces:
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"{name} | {emotion} {emoji}"
            cv2.putText(display_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', display_frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/end', methods=['POST'])
def end():
    camera.release()
    cv2.destroyAllWindows()
    return "Session Ended!"


if __name__ == '__main__':
    app.run(debug=False)
