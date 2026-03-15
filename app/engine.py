import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import cv2
import urllib.request
import os

# Pose Landmarker Model configuration
MODEL_PATH = "app/models/pose_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"

def download_model():
    """Downloads the MediaPipe pose landmarker model if missing."""
    if not os.path.exists(MODEL_PATH):
        print("Downloading MediaPipe pose landmarker model...")
        os.makedirs("app/models", exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded successfully.")

# Initialize model download
download_model()

# MediaPipe landmark indices mapping
LANDMARKS = {
    "nose":             0,
    "left_shoulder":    11,
    "right_shoulder":   12,
    "left_hip":         23,
    "right_hip":        24,
    "left_ankle":       27,
    "right_ankle":      28,
}

def get_pixel_distance(point_a, point_b):
    """Calculates straight-line Euclidean distance between two pixel points."""
    return np.sqrt((point_a[0] - point_b[0])**2 + (point_a[1] - point_b[1])**2)

def extract_landmarks_from_image(image_rgb, image_width, image_height):
    """Detects pose landmarks and returns pixel coordinates."""
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE
    )

    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=image_rgb
        )
        results = landmarker.detect(mp_image)

    if not results.pose_landmarks or len(results.pose_landmarks) == 0:
        return None

    landmarks_raw = results.pose_landmarks[0]
    landmarks = {}
    for name, index in LANDMARKS.items():
        landmark = landmarks_raw[index]
        x_pixel = int(landmark.x * image_width)
        y_pixel = int(landmark.y * image_height)
        landmarks[name] = (x_pixel, y_pixel)

    return landmarks

def estimate_measurements(image_rgb, image_width, image_height, real_height_cm=170):
    """Converts detected landmarks into estimated physical measurements."""
    landmarks = extract_landmarks_from_image(image_rgb, image_width, image_height)

    if landmarks is None:
        return {"error": "No person detected in the image. Please use a clear full-body photo."}

    # Calculate scale factor based on pixel height
    ankle_y = (landmarks["left_ankle"][1] + landmarks["right_ankle"][1]) / 2
    top_y = landmarks["nose"][1]
    pixel_height = abs(ankle_y - top_y)

    if pixel_height == 0:
        return {"error": "Could not calculate height from image. Ensure full body is visible."}

    cm_per_pixel = real_height_cm / pixel_height

    # Calculate pixel-based measurements with approximation factors
    shoulder_px = get_pixel_distance(landmarks["left_shoulder"], landmarks["right_shoulder"])
    chest_px = shoulder_px * 1.05
    waist_px = get_pixel_distance(landmarks["left_hip"], landmarks["right_hip"]) * 1.1
    hip_px = get_pixel_distance(landmarks["left_hip"], landmarks["right_hip"])

    # Final conversion to centimeters
    return {
        "height_cm":          round(real_height_cm, 1),
        "shoulder_width_cm":  round(shoulder_px * cm_per_pixel, 1),
        "chest_cm":           round(chest_px * cm_per_pixel, 1),
        "waist_cm":           round(waist_px * cm_per_pixel, 1),
        "hip_cm":             round(hip_px * cm_per_pixel, 1),
    }