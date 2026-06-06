import base64
import os
import re

import cv2
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FRONTEND_DIR = os.path.join(BASE_DIR, 'face-detector-frontend')

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path='')
# Enable CORS for API routes. Set the `CORS_ORIGINS` env var to restrict origins if needed.
CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*')
CORS(app, resources={r"/api/*": {"origins": CORS_ORIGINS}})

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
SMILE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_smile.xml'
)
EYE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml'
)
IMAGE_DATA_URI_PATTERN = re.compile(r'^data:image/\w+;base64,')


def estimate_emotion(face_gray, width, height):
    smiles = SMILE_CASCADE.detectMultiScale(
        face_gray,
        scaleFactor=1.7,
        minNeighbors=20,
        minSize=(24, 24),
    )
    if len(smiles) > 0:
        score = min(0.99, 0.72 + 0.08 * len(smiles))
        return 'happy', score

    eyes = EYE_CASCADE.detectMultiScale(
        face_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(20, 20),
    )

    if len(eyes) >= 2:
        avg_eye_height = sum(h for (_, _, _, h) in eyes) / len(eyes)
        avg_eye_width = sum(w for (_, _, w, _) in eyes) / len(eyes)
        eye_ratio = avg_eye_height / (avg_eye_width + 1e-6)
        face_ratio = height / (width + 1e-6)
        if eye_ratio > 0.42 or face_ratio > 1.1:
            return 'surprised', 0.82
        return 'neutral', 0.70

    if len(eyes) == 1:
        if height > width:
            return 'sad', 0.65
        return 'angry', 0.65

    return 'neutral', 0.60


def parse_image_from_data_url(image_data_url):
    image_data = IMAGE_DATA_URI_PATTERN.sub('', image_data_url)
    image_bytes = base64.b64decode(image_data)
    np_array = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(np_array, cv2.IMREAD_COLOR)


def analyze_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(64, 64),
    )

    if len(faces) == 0:
        return {'emotion': 'no_face', 'confidence': 0.0, 'box': None}

    x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
    face_gray = gray[y:y + h, x:x + w]
    if face_gray.size == 0:
        return {'emotion': 'no_face', 'confidence': 0.0, 'box': None}

    emotion, confidence = estimate_emotion(face_gray, w, h)
    return {
        'emotion': emotion,
        'confidence': round(confidence, 2),
        'box': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)},
    }


@app.route('/', methods=['GET'])
def serve_frontend():
    return app.send_static_file('index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze():
    payload = request.get_json(silent=True)
    if not payload or 'image' not in payload:
        return jsonify({'error': 'Missing image payload'}), 400

    try:
        frame = parse_image_from_data_url(payload['image'])
        if frame is None:
            raise ValueError('Invalid image data')
    except Exception as exc:
        return jsonify({'error': str(exc)}), 400

    result = analyze_frame(frame)
    return jsonify(result)


def draw_label(frame, label, x, y):
    cv2.putText(
        frame,
        label,
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


if __name__ == '__main__':
    print('Starting Face Detector backend on http://127.0.0.1:5000')
    app.run(host='127.0.0.1', port=5000)
