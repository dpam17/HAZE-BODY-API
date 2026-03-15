import cv2
import numpy as np
from app.engine import estimate_measurements

def load_and_prepare_image(file_bytes):
    """Decodes raw bytes into an RGB image for MediaPipe."""
    np_array = np.frombuffer(file_bytes, np.uint8)
    image_bgr = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if image_bgr is None:
        return None, None, None

    image_height, image_width, _ = image_bgr.shape
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    return image_rgb, image_width, image_height

async def process_multiple_images(image_files, real_height_cm=170):
    """Processes 2-4 images and returns the averaged measurements."""
    all_results = []

    for image_file in image_files:
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