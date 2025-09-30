import utils.transport
import utils.session
import torch
import torch.nn as nn
from torchvision import models, transforms
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import os
import json
import re
import google.generativeai as genai
from dotenv import load_dotenv

# -----------------------------
# ✅ Load environment variables
# -----------------------------
load_dotenv()

# Configure Gemini with API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# -----------------------------
# ✅ Choose Gemini Model
# -----------------------------
PREFERRED_MODELS = [
    "models/gemini-2.5-flash",     # fast, cheaper
    "models/gemini-2.5-pro",       # smarter, more accurate
    "models/gemini-flash-latest",  # always latest flash
    "models/gemini-pro-latest"     # always latest pro
]

gemini_model = None
for m in PREFERRED_MODELS:
    try:
        gemini_model = genai.GenerativeModel(m)
        print(f"✅ Using Gemini model: {m}")
        break
    except Exception as e:
        print(f"⚠️ Could not load {m}: {e}")

if not gemini_model:
    raise RuntimeError("❌ No valid Gemini model available. Please check your API key or project access.")

# -----------------------------
# ✅ Helper: Extract disease name
# -----------------------------
def extract_disease_name(class_name: str) -> str:
    """
    Extract only the disease name from PlantVillage class.
    Example:
      'Tomato___Late_blight' → 'Late Blight'
      'Cherry_(including_sour)___healthy' → 'Healthy'
    """
    if "___" in class_name:
        _, disease = class_name.split("___")
        return disease.replace("_", " ").strip().capitalize()
    else:
        return class_name.replace("_", " ").strip().capitalize()

# -----------------------------
# ✅ Load class names from file
# -----------------------------
with open("classes.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]

num_classes = len(classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# ✅ Load trained PlantVillage model
# -----------------------------
model = models.mobilenet_v2(weights=None)
state_dict = torch.load("crop_disease_best.pth", map_location=device)

if "classifier.1.weight" in state_dict:
    saved_num_classes = state_dict["classifier.1.weight"].shape[0]
else:
    saved_num_classes = num_classes

model.classifier[1] = nn.Linear(model.last_channel, saved_num_classes)
model.load_state_dict(state_dict, strict=False)

if saved_num_classes < num_classes:
    classes = classes[:saved_num_classes]

model = model.to(device)
model.eval()

# -----------------------------
# ✅ Preprocessing for images
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# -----------------------------
# ✅ FastAPI app
# -----------------------------
app = FastAPI(title="🌿 Crop Disease Analyzer (PlantVillage)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# ✅ Routes
# -----------------------------
@app.get("/")
def home():
    return {"message": "Crop Disease Analyzer API is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Load uploaded image
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # Preprocess
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        conf, pred = torch.max(probs, 0)

    raw_class = classes[pred.item()]
    disease_name = extract_disease_name(raw_class)  # ✅ Only disease
    confidence = float(conf.item()) * 100

    # ✅ Case 1: Plant is healthy → no Gemini call
    if "healthy" in disease_name.lower():
        return {
            "predicted_disease": disease_name,
            "confidence": confidence,
            "solution_en": "The plant is healthy. No treatment is needed.",
            "solution_hi": "पौधा स्वस्थ है। किसी उपचार की आवश्यकता नहीं है।"
        }

    # ✅ Case 2: Plant is diseased → ask Gemini
    prompt = f"""
    The crop disease is: {disease_name}.
    Respond ONLY in raw JSON (no code blocks) with this format:
    {{
      "english": "short farmer-friendly solution in English",
      "hindi": "same solution in Hindi"
    }}
    """

    try:
        gemini_response = gemini_model.generate_content(prompt)
        response_text = gemini_response.text or "{}"

        # Clean ```json ... ``` fences if Gemini added them
        response_text = re.sub(r"```json\s*|\s*```", "", response_text).strip()

        data = json.loads(response_text)
        solution_en = data.get("english", "No English solution")
        solution_hi = data.get("hindi", "No Hindi solution")
    except Exception as e:
        print("⚠️ Gemini API error:", e)
        solution_en = "Could not fetch solution."
        solution_hi = str(e)

    return {
        "predicted_disease": disease_name,
        "confidence": confidence,
        "solution_en": solution_en,
        "solution_hi": solution_hi
    }

# 🔹 List available models
@app.get("/models")
def list_models():
    models = [m.name for m in genai.list_models() if "generateContent" in m.supported_generation_methods]
    return {"available_models": models}
    