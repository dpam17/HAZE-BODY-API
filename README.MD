# Haze Couture Body Measurement API

A REST API that estimates key body measurements from 1–4 uploaded photos.
Built with FastAPI and MediaPipe for Haze Couture's SIWES Intern Challenge.

---

## What It Does

Upload 1–4 full-body photos of a person and receive estimated measurements:
```json
{
    "height_cm": 170.0,
    "shoulder_width_cm": 38.2,
    "chest_cm": 40.1,
    "waist_cm": 32.5,
    "hip_cm": 36.0
}
```

---

## Project Structure
```
haze-body-api/
├── app/
│   ├── __init__.py         
│   ├── main.py             # FastAPI entry point — API routes live here
│   ├── engine.py           # AI logic — MediaPipe pose estimation
│   ├── utils.py            # Image preparation and multi-image averaging
│   └── models/             # Placeholder for model weights
├── data/
│   └── samples/            # Sample images for testing
├── docs/
│   └── write_up.pdf        # Technical write-up (Deliverable 5)
├── tests/
│   └── test_api.py         # Basic API tests
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/haze-body-api.git
cd haze-body-api
```

### 2. Create and Activate a Virtual Environment
```bash
# Create the virtual environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# Activate it (Mac/Linux)
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the API Locally
```bash
uvicorn app.main:app --reload
```

The API will be live at:
```
http://127.0.0.1:8000
```

---

## How to Use the API

### Option 1 — Browser (Swagger UI)
FastAPI comes with a built-in testing interface. Once running, visit:
```
http://127.0.0.1:8000/docs
```
You can upload images and test the API directly from your browser — no extra tools needed.

### Option 2 — curl (Terminal)
```bash
# Single image
curl -X POST "http://127.0.0.1:8000/measure" \
  -F "images=@/path/to/your/photo.jpg"

# Multiple images
curl -X POST "http://127.0.0.1:8000/measure" \
  -F "images=@photo1.jpg" \
  -F "images=@photo2.jpg"
```

### Option 3 — Live Deployed Demo
Test the API without installing anything:
```
[YOUR DEPLOYED URL HERE]
```

---

## API Endpoints

| Method | URL | Description |
|--------|-----|-------------|
| GET | `/` | Health check — confirms API is running |
| POST | `/measure` | Upload 1–4 images, returns body measurements |

### Parameters for `/measure`

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `images` | File(s) | Yes | — | 1–4 full body photos (JPEG, PNG, WEBP) |
| `real_height_cm` | Float | No | 170.0 | Person's actual height for better accuracy |

---

## Tips for Best Results

- Stand **straight** and **face the camera directly**
- Ensure your **full body is visible** — head to toe
- Use a **plain background** where possible
- Good **lighting** improves landmark detection significantly
- Uploading **2–4 photos** produces more reliable averaged results

---

## Known Limitations

- Chest and waist measurements are approximations —
  MediaPipe does not provide direct landmarks for these points
- Accuracy depends on how close the person's height is to the assumed 170cm default
- Side profile or angled photos will reduce accuracy
- Works best with front-facing full body images

---

## AI Tools Used

| Tool | Purpose |
|------|---------|
| Claude (Anthropic) | Code generation, architecture guidance, explanations |
| MediaPipe (Google) | Pose landmark detection |

---

## What I'd Improve With More Time

- Allow users to input their real height for a more accurate scale factor
- Add side-profile image support for better chest and waist estimation
- Implement actual video frame extraction and processing
- Add a simple frontend UI for easier testing
- Improve waist estimation using the midpoint between shoulders and hips

---

## Running Tests
```bash
pytest tests/test_api.py
```


Built by PAM DAVID WENG — SIWES Intern Challenge Submission