"""Tiny Flask demo: upload a photo of a digit, get a prediction."""
import io
import torch
from flask import Flask, request, jsonify, render_template_string
from PIL import Image, ImageOps
from torchvision import transforms
from model import Net

app = Flask(__name__)
model = Net()
model.load_state_dict(torch.load("model.pt", map_location="cpu"))
model.eval()
tf = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

PAGE = """<h2>Digit classifier</h2>
<form method=post action=/predict enctype=multipart/form-data>
<input type=file name=image accept=image/* required> <button>Predict</button></form>
<p>Use a dark digit on a white background or the reverse; the app auto-inverts.</p>"""


@app.get("/")
def home():
    return render_template_string(PAGE)


@app.post("/predict")
def predict():
    img = Image.open(io.BytesIO(request.files["image"].read())).convert("L")
    if sum(img.getdata()) / (img.width * img.height) > 127:  # white background -> invert to match MNIST
        img = ImageOps.invert(img)
    with torch.no_grad():
        probs = torch.softmax(model(tf(img).unsqueeze(0)), 1)[0]
    return jsonify(digit=int(probs.argmax()), confidence=round(float(probs.max()), 4))


if __name__ == "__main__":
    app.run(debug=True)
