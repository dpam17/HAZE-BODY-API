# engine.py
# This file is the core AI logic of our API.
# NOTE: We are using MediaPipe's newer Tasks API instead of mp.solutions
# because mp.solutions is not supported on Python 3.13+
# The logic is the same — we detect body landmarks and calculate measurements

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import cv2
import urllib.request
import os

# --- Download the Pose Landmarker Model ---
# MediaPipe's new API requires a .task model file downloaded separately
# We download it once and save it locally so it doesn't re-download every time
MODEL_PATH = "app/models/pose_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"

def download_model():
    """
    Downloads the MediaPipe pose landmarker model file if it doesn't exist yet.
    This only runs once — after that it uses the locally saved file.
    """
    if not os.path.exists(MODEL_PATH):
        print("Downloading MediaPipe pose landmarker model...")
        os.makedirs("app/models", exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded successfully.")

# Download model when engine.py is first imported
download_model()

# --- Landmark Index Reference ---
# MediaPipe gives us 33 landmarks, each with an index number
# We only need specific ones for our measurements
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
    """
    Calculates the straight-line distance between two (x, y) pixel points.
    Uses the Pythagorean theorem — same as before.
    """
    return np.sqrt((point_a[0] - point_b[0])**2 + (point_a[1] - point_b[1])**2)


def extract_landmarks_from_image(image_rgb, image_width, image_height):
    """
    Feeds an image into the MediaPipe pose landmarker.
    Returns a dictionary of landmark name → (x_pixel, y_pixel) positions.
    """

    # --- Create the pose landmarker ---
    # BaseOptions tells MediaPipe where the model file is
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)

    # PoseLandmarkerOptions configures how the detector runs
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE  # IMAGE mode = single static images
    )

    # Create the landmarker and process the image
    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:

        # Convert our numpy RGB image into a MediaPipe Image object
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=image_rgb
        )

        # Run detection
        results = landmarker.detect(mp_image)

    # If no person was detected return None
    if not results.pose_landmarks or len(results.pose_landmarks) == 0:
        return None

    # Use the first detected person's landmarks
    landmarks_raw = results.pose_landmarks[0]

    landmarks = {}
    for name, index in LANDMARKS.items():
        landmark = landmarks_raw[index]

        # Convert normalized (0-1) coordinates to actual pixel positions
        x_pixel = int(landmark.x * image_width)
        y_pixel = int(landmark.y * image_height)
        landmarks[name] = (x_pixel, y_pixel)

    return landmarks


def estimate_measurements(image_rgb, image_width, image_height, real_height_cm=170):
    """
    Main function — detects landmarks and converts pixel distances to centimeters.
    Logic is identical to before — only the landmark detection method changed.
    """

    # Step 1: Get landmark positions
    landmarks = extract_landmarks_from_image(image_rgb, image_width, image_height)

    if landmarks is None:
        return {"error": "No person detected in the image. Please use a clear full-body photo."}

    # Step 2: Calculate pixel height (nose to ankles)
    ankle_y = (landmarks["left_ankle"][1] + landmarks["right_ankle"][1]) / 2
    top_y = landmarks["nose"][1]
    pixel_height = abs(ankle_y - top_y)

    if pixel_height == 0:
        return {"error": "Could not calculate height from image. Ensure full body is visible."}

    # Step 3: Build scale factor (cm per pixel)
    cm_per_pixel = real_height_cm / pixel_height

    # Step 4: Calculate measurements in pixels
    shoulder_px = get_pixel_distance(landmarks["left_shoulder"], landmarks["right_shoulder"])
    chest_px = shoulder_px * 1.05
    waist_px = get_pixel_distance(landmarks["left_hip"], landmarks["right_hip"]) * 1.1
    hip_px = get_pixel_distance(landmarks["left_hip"], landmarks["right_hip"])

    # Step 5: Convert pixels to centimeters
    return {
        "height_cm":          round(real_height_cm, 1),
        "shoulder_width_cm":  round(shoulder_px * cm_per_pixel, 1),
        "chest_cm":           round(chest_px * cm_per_pixel, 1),
        "waist_cm":           round(waist_px * cm_per_pixel, 1),
        "hip_cm":             round(hip_px * cm_per_pixel, 1),
    }