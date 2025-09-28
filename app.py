import torch
import torch.nn as nn
from torchvision import models, transforms
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

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

# load checkpoint
state_dict = torch.load("crop_disease_best.pth", map_location=device)

# check how many output classes in checkpoint
if "classifier.1.weight" in state_dict:
    saved_num_classes = state_dict["classifier.1.weight"].shape[0]
else:
    saved_num_classes = num_classes  # fallback

# adjust model classifier to match checkpoint
model.classifier[1] = nn.Linear(model.last_channel, saved_num_classes)

# load weights
model.load_state_dict(state_dict, strict=False)

# if saved model has fewer classes than classes.txt → trim list
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

# 🔹 Enable CORS so frontend (index.html) can call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all, restrict later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

    return {
        "predicted_class": classes[pred.item()],
        "confidence": float(conf.item()) * 100
    }
