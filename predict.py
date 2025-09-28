import torch
from torchvision import models, transforms, datasets
from PIL import Image
import os

# ✅ Dataset path (same as training)
data_dir = r"F:/Download/cropdisease/datasets/plantvillage/plantVillage"

# ✅ Load class names
dataset = datasets.ImageFolder(data_dir, transform=transforms.ToTensor())
class_names = dataset.classes

# ✅ Load trained model
num_classes = len(class_names)
model = models.mobilenet_v2()
model.classifier[1] = torch.nn.Linear(model.last_channel, num_classes)
model.load_state_dict(torch.load("crop_disease_best.pth", map_location="cpu"))
model.eval()

# ✅ Image transform (same as training)
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ✅ Path to test image
img_path = r"/leaf_sample.jpg"
  # change to your test leaf image
img = Image.open(img_path).convert("RGB")
img = transform(img).unsqueeze(0)

# ✅ Predict
with torch.no_grad():
    outputs = model(img)
    probs = torch.nn.functional.softmax(outputs, dim=1)
    conf, predicted = torch.max(probs, 1)

print(f"Prediction: {class_names[predicted.item()]} ({conf.item()*100:.2f}% confidence)")
