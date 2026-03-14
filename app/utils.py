# utils.py
# This file contains helper functions that support the main engine.
# Specifically it handles:
# 1. Loading and preparing images for processing
# 2. Processing multiple images and averaging their results
# These are kept separate from engine.py to keep the code clean and organized
# Rule of thumb: engine.py THINKS, utils.py PREPARES

import cv2
import numpy as np
from app.engine import estimate_measurements


def load_and_prepare_image(file_bytes):
    """
    Takes raw image bytes (the image as it comes in from the API upload)
    and converts it into a format MediaPipe can work with.

    Why bytes? When someone uploads an image through the API,
    it doesn't arrive as a neat file — it arrives as a stream of raw bytes.
    We need to decode those bytes back into an actual image.

    Steps:
    1. Convert bytes → numpy array (a grid of numbers)
    2. Decode that array → actual image OpenCV can read (BGR format)
    3. Convert BGR → RGB (what MediaPipe expects)
    4. Return the RGB image + its dimensions
    """

    # Step 1: Convert the raw bytes into a numpy array
    # np.frombuffer reads the raw bytes and turns them into a 1D array of numbers
    np_array = np.frombuffer(file_bytes, np.uint8)

    # Step 2: Decode the numpy array into an actual image
    # cv2.IMREAD_COLOR means: load it as a full color image (not grayscale)
    image_bgr = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    # If decoding failed (corrupt file, wrong format, etc.) return None
    if image_bgr is None:
        return None, None, None

    # Step 3: Get the image dimensions
    # image.shape returns (height, width, channels)
    # channels = 3 because every pixel has 3 color values (B, G, R)
    image_height, image_width, _ = image_bgr.shape
    # The underscore _ means "I know there's a third value but I don't need it"

    # Step 4: Convert BGR → RGB for MediaPipe
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    return image_rgb, image_width, image_height


async def process_multiple_images(image_files, real_height_cm=170):
    """
    Handles the case where the user uploads 2-4 images instead of just one.
    Now async because reading uploaded files requires await on the server.
    """

    all_results = []

    for image_file in image_files:
        # await is required here — reading uploaded files is an async operation
        file_bytes = await image_file.read()

        image_rgb, image_width, image_height = load_and_prepare_image(file_bytes)

        if image_rgb is None:
            continue

        result = estimate_measurements(image_rgb, image_width, image_height, real_height_cm)

        if "error" in result:
            continue

        all_results.append(result)

    if not all_results:
        return {"error": "No valid measurements could be extracted from the uploaded images."}

    keys = ["height_cm", "shoulder_width_cm", "chest_cm", "waist_cm", "hip_cm"]

    averaged = {}
    for key in keys:
        values = [result[key] for result in all_results]
        averaged[key] = round(sum(values) / len(values), 1)

    return averaged