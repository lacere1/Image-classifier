from flask import Flask, request, jsonify
from PIL import Image, ImageOps
from model import ModelWrapper
import io

app = Flask(__name__)

try:
    # Not strictly necessary if front end proxies /api -> api:8000 via nginx,
    # but useful when running the API alone on localhost.
    from flask_cors import CORS
    CORS(app)
except Exception:
    pass

# Load model once on startup
model = ModelWrapper()

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.post("/predict")
def predict():
    if "image" not in request.files:
        return jsonify({"error": "missing file field 'image'"}), 400

    file = request.files["image"]
    if not file or file.filename == "":
        return jsonify({"error": "empty filename"}), 400

    try:
        image = Image.open(file.stream).convert("RGB")
        # Correct orientation from EXIF if present
        image = ImageOps.exif_transpose(image)
    except Exception as e:
        return jsonify({"error": f"invalid image: {e}"}), 400

    try:
        topk = model.predict_topk(image, k=5)
        return jsonify({
            "predictions": [{"label": lbl, "score": float(score)} for lbl, score in topk]
        })
    except Exception as e:
        return jsonify({"error": f"inference failed: {e}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
