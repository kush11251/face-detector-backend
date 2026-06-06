# Face Detector

A simple Python project that detects faces from the webcam and estimates basic emotions using OpenCV Haar cascades.

## Features

- Detects faces in real time using OpenCV Haar cascade
- Adds a bounding box around each face
- Estimates basic emotions from facial features:
  - `happy`
  - `surprised`
  - `neutral`
  - `sad`
  - `angry`
- Displays emotion label and confidence on the camera feed

## Requirements

- Python 3.11+ or compatible
- OpenCV
- NumPy

## Setup

From the project root:

```bash
cd /Users/kussagrapathak/Documents/personal/projects/python/face-detector/backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run

Activate the environment and run:

```bash
source .venv/bin/activate
python main.py
```

Then open the browser at:

```bash
http://127.0.0.1:5000
```

The frontend captures webcam frames and sends them to the backend at `/api/analyze`.

## Deploying the backend (Render)

Recommended: deploy the backend as a web service on Render using Gunicorn.

1. Create a new Web Service on Render.
2. Connect your GitHub repo and select the `backend` folder as the root path.
3. Set the build command to:

```bash
pip install -r requirements.txt
```

4. Set the start command to:

```bash
gunicorn main:app --bind 0.0.0.0:$PORT
```

5. Add any environment variables you need in Render's dashboard. After deploy, note the service URL (e.g. `https://your-backend.onrender.com`).

When deploying the frontend separately, update the frontend `config.js` to point `API_BASE` at the backend URL.

## Notes about production
- Use HTTPS endpoints for the frontend to call the backend.
- For production, consider adding CORS handling and rate-limiting.

## Cleanup and Git

The project ignores local Python environments and generated caches via `.gitignore`.

## Notes

- The emotion detection is heuristic-based, using smile and eye detections.
- For more accurate emotion recognition, a model-based approach can be added later.
