from transformers import AutoModelForImageClassification, AutoFeatureExtractor
import requests
from PIL import Image
import torch

print("Loading model...")
model_id = 'google/vit-base-patch16-224'
model = AutoModelForImageClassification.from_pretrained(model_id)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
print("Model loaded successfully.")
# Replace the URL with the actual URL of the diseased crop image
url = 'https://genotypingcenter.com/wp-content/uploads/2022/07/shutterstock_2137022623-scaled.jpg'
image = Image.open(requests.get(url, stream=True).raw)
print("Displaying input image...")
display(image)
print("Image displayed successfully.")
print("Preprocessing the image...")
inputs = feature_extractor(images=image, return_tensors="pt")
print("Making predictions...")
with torch.no_grad():
    logits = model(**inputs).logits

# Convert logits to probabilities and get the predicted label
predicted_label = logits.softmax(dim=1).argmax(dim=1).item()
print("Predicted
