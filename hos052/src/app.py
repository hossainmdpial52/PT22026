"""
AgroGuard AI - Version 1
Plant leaf disease detection using a pre-trained MobileNetV2 model.

How to run:
    python src/app.py
"""

import gradio as gr
import torch
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Pre-trained MobileNetV2 fine-tuned on the PlantVillage dataset (38 classes)
MODEL_NAME = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
TOP_K = 5  # Number of top predictions to show


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------
def build_preprocessor():
    """
    Return a function that converts a PIL image into a model-ready tensor.

    The model has an old preprocessor config that newer versions of
    `transformers` may not recognize, so we try three options in order:
      1. AutoImageProcessor (automatic)
      2. MobileNetV2ImageProcessor (explicit class)
      3. Manual torchvision pipeline (same steps as MobileNetV2 training)
    """
    # Option 1: automatic processor
    try:
        processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
        print("Using AutoImageProcessor.")
        return lambda img: processor(images=img, return_tensors="pt")["pixel_values"]
    except Exception as error:
        print(f"AutoImageProcessor failed ({type(error).__name__}), trying next option...")

    # Option 2: explicit MobileNetV2 processor class
    try:
        from transformers import MobileNetV2ImageProcessor

        processor = MobileNetV2ImageProcessor.from_pretrained(MODEL_NAME)
        print("Using MobileNetV2ImageProcessor.")
        return lambda img: processor(images=img, return_tensors="pt")["pixel_values"]
    except Exception as error:
        print(f"MobileNetV2ImageProcessor failed ({type(error).__name__}), using manual pipeline...")

    # Option 3: manual pipeline (resize, center crop, normalize to [-1, 1])
    print("Using manual torchvision preprocessing.")
    manual = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return lambda img: manual(img).unsqueeze(0)


# ---------------------------------------------------------------------------
# Model loading (done once when the app starts)
# ---------------------------------------------------------------------------
print("Loading model, please wait...")
preprocess = build_preprocessor()
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
model.eval()  # Switch to inference mode (no training)
print("Model loaded successfully.")


def clean_label(raw_label: str) -> str:
    """Convert a raw label like 'Tomato___Late_blight' to 'Tomato - Late blight'."""
    parts = raw_label.split("___")
    if len(parts) == 2:
        plant = parts[0].replace("_", " ")
        disease = parts[1].replace("_", " ")
        return f"{plant} - {disease}"
    return raw_label.replace("_", " ")


def predict(image):
    """Take a PIL image and return the top-K predictions as {label: probability}."""
    # Error handling: the user pressed Submit without uploading an image
    if image is None:
        raise gr.Error("Please upload a leaf image first.")

    try:
        # Resize and normalize the image the way the model expects
        pixel_values = preprocess(image.convert("RGB"))

        # Run the model without calculating gradients (faster, uses less memory)
        with torch.no_grad():
            logits = model(pixel_values=pixel_values).logits

        # Convert raw scores into probabilities that sum to 1
        probabilities = torch.softmax(logits, dim=-1)[0]

        # Pick the most likely classes
        top_probs, top_ids = torch.topk(probabilities, k=TOP_K)

        results = {}
        for prob, idx in zip(top_probs, top_ids):
            raw_label = model.config.id2label[idx.item()]
            results[clean_label(raw_label)] = float(prob)
        return results

    except Exception as error:
        raise gr.Error(f"Could not process the image: {error}")


# ---------------------------------------------------------------------------
# User interface
# ---------------------------------------------------------------------------
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload a plant leaf image"),
    outputs=gr.Label(num_top_classes=TOP_K, label="Prediction"),
    title="AgroGuard AI - Plant Disease Detection",
    description=(
        "Upload a photo of a plant leaf and the AI model will predict "
        "the plant type and its disease (or healthy status)."
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
