import gradio as gr
import torch
from PIL import Image

from model import FakeDetectorCNN, FakeDetectorDualBranch
from transforms import eval_transform, IMG_SIZE

CKPT_PATHS = {
    "CNN + FFT (dual-branch)": "best_dual.pt",
    "Baseline CNN": "best_baseline.pt",
}

_cache = {}


def load(name):
    if name in _cache:
        return _cache[name]
    ckpt = torch.load(CKPT_PATHS[name], map_location="cpu")
    cls = FakeDetectorCNN if ckpt["model_name"] == "baseline" else FakeDetectorDualBranch
    model = cls()
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    _cache[name] = (model, ckpt["class_names"])
    return _cache[name]


def predict(img: Image.Image, model_name: str):
    if img is None:
        return None
    model, class_names = load(model_name)
    # Model was trained on 32x32 images: resize the same way for any upload
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
    x = eval_transform(img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    return {class_names[i]: float(probs[i]) for i in range(len(class_names))}


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil", label="Upload an image"),
        gr.Radio(list(CKPT_PATHS), value="CNN + FFT (dual-branch)", label="Model"),
    ],
    outputs=gr.Label(num_top_classes=2, label="Prediction"),
    title="RealSeeker: Real vs AI-generated photo detector",
    description=(
        "PyTorch CNN (+ FFT branch) trained on 32x32 images. "
        "Uploaded images are downscaled to 32x32 first, so results on large, "
        "real-world photos are not guaranteed (in-distribution only)."
    ),
)

if __name__ == "__main__":
    demo.launch()
