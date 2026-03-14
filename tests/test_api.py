# test_api.py
# This file tests our API to make sure it behaves correctly
# We test three things:
# 1. The API is running and responding
# 2. It correctly rejects bad requests
# 3. It correctly processes a valid image
# 
# We use two libraries for testing:
# - pytest: the testing framework (runs our tests and reports pass/fail)
# - httpx: a lightweight HTTP client that sends requests to our API during tests
# 
# Think of this file like a checklist —
# each function is one item on the checklist that must pass

from fastapi.testclient import TestClient
from app.main import app
import os

# --- Test Client Setup ---
# TestClient is a built-in FastAPI tool that simulates HTTP requests
# without needing the server to actually be running
# Think of it like a robot that pretends to be a user sending requests
client = TestClient(app)


# --- Test 1: Health Check ---
# This test simply checks that the API is alive and returning the right response
# It sends a GET request to "/" and checks:
# 1. The status code is 200 (meaning success)
# 2. The response contains our expected message
def test_root_endpoint():
    # Send a GET request to the root URL
    response = client.get("/")

    # Assert means: "this must be true — if not, the test fails"
    assert response.status_code == 200

    # Check that the response body contains our status message
    assert "status" in response.json()


# --- Test 2: Reject Empty Request ---
# This test checks that the API correctly rejects a request
# that has no images attached
# Expected behavior: return a 422 error (Unprocessable Entity)
def test_measure_no_images():
    # Send a POST request to /measure with no files attached
    response = client.post("/measure")

    # FastAPI automatically returns 422 when required fields are missing
    assert response.status_code == 422


# --- Test 3: Reject Wrong File Type ---
# This test checks that the API rejects non-image files
# We send a .txt file and expect a 400 error (Bad Request)
def test_measure_wrong_file_type():
    # Create a fake text file in memory to upload
    # b"this is not an image" = the file contents as bytes
    fake_file = ("fake.txt", b"this is not an image", "text/plain")

    response = client.post(
        "/measure",
        files=[("images", fake_file)]
    )

    # We expect a 400 Bad Request since it's not an image
    assert response.status_code == 400


# --- Test 4: Reject Too Many Images ---
# This test checks that the API rejects more than 4 images
# We send 5 images and expect a 400 error
def test_measure_too_many_images():
    # Create 5 fake image-like files
    # We mark them as image/jpeg so they pass the file type check
    # but send 5 of them to trigger the count validation
    fake_files = [
        ("images", (f"photo{i}.jpg", b"fake image data", "image/jpeg"))
        for i in range(5)  # creates 5 fake files
    ]

    response = client.post("/measure", files=fake_files)

    # We expect a 400 Bad Request since we sent 5 images
    assert response.status_code == 400


# --- Test 5: Real Image Test ---
# This test checks the full pipeline with a real image
# It looks for a sample image in our data/samples folder
# If no sample image exists it skips the test gracefully
def test_measure_with_real_image():
    # Look for any jpg file in our samples folder
    samples_dir = "data/samples"

    # Get a list of all jpg files in the samples folder
    sample_images = [
        f for f in os.listdir(samples_dir)
        if f.endswith(".jpg") or f.endswith(".png")
    ]

    # If no sample images exist, skip this test
    # This prevents the test from failing just because
    # the samples folder is empty
    if not sample_images:
        print("No sample images found — skipping real image test")
        return

    # Use the first sample image found
    image_path = os.path.join(samples_dir, sample_images[0])

    # Open the image file and send it to the API
    with open(image_path, "rb") as f:
        # "rb" means "read as bytes" — required for binary files like images
        response = client.post(
            "/measure",
            files=[("images", (sample_images[0], f, "image/jpeg"))]
        )

    # We expect either:
    # 200 — measurements extracted successfully
    # 422 — image was received but no person was detected
    # Both are acceptable — the important thing is the API handled it gracefully
    assert response.status_code in [200, 422]

    # If successful, check that all 5 measurements are in the response
    if response.status_code == 200:
        data = response.json()
        assert "height_cm" in data
        assert "shoulder_width_cm" in data
        assert "chest_cm" in data
        assert "waist_cm" in data
        assert "hip_cm" in data