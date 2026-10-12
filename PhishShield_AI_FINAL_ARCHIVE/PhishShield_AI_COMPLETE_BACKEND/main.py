
import os
import io
import json
import pickle
import numpy as np

from PIL import Image
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ============================================================
# PHISHSHIELD AI - CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_CONFIG = os.path.join(
    BASE_DIR, "phishshield_project_config.json"
)

FUSION_CONFIG = os.path.join(
    BASE_DIR, "phishshield_fusion_config.json"
)

TOKENIZER_FILE = os.path.join(
    BASE_DIR, "phishshield_url_tokenizer.pkl"
)

URL_CNN_FILE = os.path.join(
    BASE_DIR, "phishshield_url_cnn.keras"
)

URL_BILSTM_FILE = os.path.join(
    BASE_DIR, "phishshield_url_bilstm.keras"
)

SCREENSHOT_MODEL_FILE = os.path.join(
    BASE_DIR, "phishshield_screenshot_cnn_improved.keras"
)


# ============================================================
# LOAD CONFIG
# ============================================================

with open(PROJECT_CONFIG, "r") as f:
    project_config = json.load(f)

with open(FUSION_CONFIG, "r") as f:
    fusion_config = json.load(f)

with open(TOKENIZER_FILE, "rb") as f:
    tokenizer = pickle.load(f)


# ============================================================
# MODEL SETTINGS
# ============================================================

URL_LENGTH = int(project_config.get("url_length", 74))

URL_WEIGHT = float(
    fusion_config.get("url_weight", 0.60)
)

VISUAL_WEIGHT = float(
    fusion_config.get("visual_weight", 0.40)
)

FUSION_THRESHOLD = float(
    fusion_config.get("selected_threshold", 0.30)
)


# ============================================================
# LOAD MODELS
# ============================================================

print("Loading PhishShield AI models...")

url_cnn = load_model(URL_CNN_FILE)
url_bilstm = load_model(URL_BILSTM_FILE)
screenshot_cnn = load_model(SCREENSHOT_MODEL_FILE)

print("✓ URL CNN loaded")
print("✓ URL BiLSTM loaded")
print("✓ Screenshot CNN loaded")


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="PhishShield AI",
    description="Multimodal Phishing Detection System",
    version="1.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# URL PREDICTION
# ============================================================

def predict_url(url: str):
    sequence = tokenizer.texts_to_sequences([url])

    padded = pad_sequences(
        sequence,
        maxlen=URL_LENGTH,
        padding="post",
        truncating="post"
    )

    # Saved model output:
    # 1 = legitimate
    # 0 = phishing

    cnn_raw = float(
        url_cnn.predict(padded, verbose=0)[0][0]
    )

    bilstm_raw = float(
        url_bilstm.predict(padded, verbose=0)[0][0]
    )

    # Convert model output to PHISHING probability
    cnn_probability = 1.0 - cnn_raw
    bilstm_probability = 1.0 - bilstm_raw

    combined_probability = (
        cnn_probability + bilstm_probability
    ) / 2.0

    label = (
        "Phishing"
        if combined_probability >= 0.5
        else "Legitimate"
    )

    return {
        "url": url,
        "cnn_probability": cnn_probability,
        "bilstm_probability": bilstm_probability,
        "probability": combined_probability,
        "label": label
    }

# ============================================================
# SCREENSHOT PREDICTION
# ============================================================

def predict_screenshot(image_bytes: bytes):

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    image = image.resize((224, 224))

    image_array = np.array(
        image,
        dtype=np.float32
    ) / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    probability = float(
        screenshot_cnn.predict(
            image_array,
            verbose=0
        )[0][0]
    )

    if probability >= 0.5:
        label = "Phishing"
    else:
        label = "Legitimate"

    return {
        "visual_probability": probability,
        "visual_label": label
    }


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "project": "PhishShield AI",
        "description": "Multimodal Phishing Detection System",
        "status": "running",
        "version": "1.0"
    }


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "url_cnn": True,
        "url_bilstm": True,
        "screenshot_cnn": True
    }


# ============================================================
# URL ENDPOINT
# ============================================================

@app.post("/predict/url")
def predict_url_endpoint(
    url: str = Form(...)
):

    result = predict_url(url)

    return result


# ============================================================
# MULTIMODAL ENDPOINT
# ============================================================

@app.post("/predict/multimodal")
async def predict_multimodal(
    url: str = Form(...),
    screenshot: UploadFile = File(...)
):

    # URL branch
    url_result = predict_url(url)

    # Screenshot branch
    image_bytes = await screenshot.read()

    visual_result = predict_screenshot(
        image_bytes
    )

    # Fusion
    url_probability = url_result["probability"]

    visual_probability = visual_result[
        "visual_probability"
    ]

    fusion_probability = (
        URL_WEIGHT * url_probability
        + VISUAL_WEIGHT * visual_probability
    )

    if fusion_probability >= FUSION_THRESHOLD:
        final_label = "Phishing"
    else:
        final_label = "Legitimate"

    return {
        "url": url,
        "url_probability": url_probability,
        "visual_probability": visual_probability,
        "url_weight": URL_WEIGHT,
        "visual_weight": VISUAL_WEIGHT,
        "fusion_probability": fusion_probability,
        "threshold": FUSION_THRESHOLD,
        "final_prediction": final_label
    }
