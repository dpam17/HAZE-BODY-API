# main.py
# This is the entry point of our API — the front door of the whole system.
# It does three things:
# 1. Creates the FastAPI application
# 2. Defines the API endpoints (the URLs people can send requests to)
# 3. Receives uploaded files and passes them to utils.py and engine.py for processing
# Think of it like a receptionist — it receives requests, 
# directs them to the right place, and sends back the response

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from app.utils import process_multiple_images, load_and_prepare_image
from app.engine import estimate_measurements

# --- Create the FastAPI app ---
# This one line creates our entire API application
# Everything else we write attaches to this object
app = FastAPI(
    title="Haze Couture Body Measurement API",
    description="Upload 1-4 photos to get estimated body measurements",
    version="1.0.0"
)

# --- CORS Middleware ---
# CORS = Cross Origin Resource Sharing
# This allows the API to be accessed from browsers and external tools
# Without this, browsers would block requests to our API for security reasons
# allow_origins=["*"] means: accept requests from anywhere
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Health Check Endpoint ---
# This is a simple endpoint that just confirms the API is alive and running
# URL: GET /
# Think of it like knocking on the door to see if anyone's home
@app.get("/")
def root():
    return {"status": "Haze Couture Body Measurement API is running"}


# --- Main Measurement Endpoint ---
# This is the core of the API — where the actual work happens
# URL: POST /measure
# It accepts 1 to 4 uploaded image files
# It returns the estimated body measurements as JSON
@app.post("/measure")
async def measure_body(
    # File(...) means: this field is required — the API will reject requests without it
    # List[UploadFile] means: accept a list of uploaded files (1 to 4 images)
    images: List[UploadFile] = File(...),

    # This is an optional parameter the user can provide
    # If they know their real height, measurements will be more accurate
    # If not provided, we default to 170cm as discussed
    real_height_cm: Optional[float] = Form(170.0)
):
    # --- Validation: Check number of images ---
    # We only accept between 1 and 4 images
    # Less than 1 = nothing to process
    # More than 4 = we set a limit to keep processing fast
    if len(images) < 1 or len(images) > 4:
        # HTTPException sends back a proper error response with a status code
        # 400 = "Bad Request" — the user sent something wrong
        raise HTTPException(
            status_code=400,
            detail="Please upload between 1 and 4 images"
        )

    # --- Validation: Check file types ---
    # We only accept image files — not PDFs, videos, or anything else
    # We check by looking at the file's content_type
    # e.g. "image/jpeg", "image/png" are valid
    allowed_types = ["image/jpeg", "image/png", "image/jpg", "image/webp"]
    for image in images:
        if image.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"{image.filename} is not a supported image type. Use JPEG, PNG, or WEBP."
            )

    # --- Single Image Path ---
    # If only one image is uploaded, we process it directly
    # No averaging needed — just load, prepare, and measure
    if len(images) == 1:
        # Read the raw bytes from the uploaded file
        file_bytes = await images[0].read()

        # Prepare the image (convert bytes → RGB image + get dimensions)
        image_rgb, image_width, image_height = load_and_prepare_image(file_bytes)

        # If image failed to load properly
        if image_rgb is None:
            raise HTTPException(
                status_code=400,
                detail="Could not read the uploaded image. Please try a different file."
            )

        # Run the measurement engine
        result = estimate_measurements(image_rgb, image_width, image_height, real_height_cm)

        # If engine returned an error (no person detected etc.)
        if "error" in result:
            raise HTTPException(status_code=422, detail=result["error"])
            # 422 = "Unprocessable Entity" — file was received but couldn't be processed

        return result

    # --- Multiple Images Path ---
    # If 2-4 images are uploaded, we use process_multiple_images from utils.py
    # This processes each image and averages the results
    else:
        result = process_multiple_images(images, real_height_cm)

        # If something went wrong across all images
        if "error" in result:
            raise HTTPException(status_code=422, detail=result["error"])

        return result