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


def process_multiple_images(image_files, real_height_cm=170):
    """
    Handles the case where the user uploads 2-4 images instead of just one.
    
    Why average multiple images?
    A single photo might have the person slightly turned, arms in the way,
    or a landmark partially hidden. Multiple photos give us multiple readings
    and averaging them produces a more reliable final measurement.

    Think of it like getting measured by a tailor 3 times and averaging the result
    — more reliable than a single measurement.

    Steps:
    1. Loop through each uploaded image
    2. Load and prepare each one
    3. Run estimate_measurements on each
    4. Collect all valid results
    5. Average them into one final result
    """

    # This list will collect the measurement results from each image
    all_results = []

    for image_file in image_files:
        # Read the raw bytes from the uploaded file
        file_bytes = image_file.read()

        # Prepare the image for processing
        image_rgb, image_width, image_height = load_and_prepare_image(file_bytes)

        # If image failed to load, skip it and move to the next one
        if image_rgb is None:
            continue

        # Run the AI measurement engine on this image
        result = estimate_measurements(image_rgb, image_width, image_height, real_height_cm)

        # If the engine returned an error (no person detected etc.) skip this image
        if "error" in result:
            continue

        # If we got valid measurements, add them to our collection
        all_results.append(result)

    # If NO images produced valid results, return an error
    if not all_results:
        return {"error": "No valid measurements could be extracted from the uploaded images."}

    # --- Averaging the results ---
    # We now have a list of measurement dictionaries, one per image
    # We want to average each measurement across all images

    # Start with the measurement keys we care about
    keys = ["height_cm", "shoulder_width_cm", "chest_cm", "waist_cm", "hip_cm"]

    averaged = {}
    for key in keys:
        # For each measurement, collect its value from every result
        values = [result[key] for result in all_results]

        # Calculate the average: sum all values, divide by how many there are
        averaged[key] = round(sum(values) / len(values), 1)

    return averaged