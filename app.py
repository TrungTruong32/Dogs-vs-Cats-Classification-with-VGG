from flask import Flask, request, jsonify
import torch
from VGG import VGG, get_vgg_layers, vgg11_config, vgg13_config, vgg16_config, vgg19_config, predict_image
from flask_cors import CORS
import os

app = Flask(__name__)  
CORS(app, resources={r"/*": {"origins": "*"}})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIM = 2

# Map model_name -> config + local file path
MODEL_CONFIGS = {
    "vgg11": (vgg11_config, "VGG11-model.pt"),
    "vgg13": (vgg13_config, "VGG13-model.pt"),
    "vgg16": (vgg16_config, "VGG16-model.pt"),
    "vgg19": (vgg19_config, "VGG19-model.pt"),
}

# Cache models sau khi load
models = {}

# Hàm load model từ local file
def load_model(config, weight_path):
    layers = get_vgg_layers(config, batch_norm=True)
    model = VGG(layers, OUTPUT_DIM).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    return model

##Dùng khi backend deploy trên Hugging Face Space
# @app.route("/predict/<model_name>", methods=["POST"])
# def predict(model_name):
#     if model_name not in MODEL_CONFIGS:
#         return jsonify({"error": "Invalid model name"}), 400

#     # Lazy loading
#     if model_name not in models:
#         config, filename = MODEL_CONFIGS[model_name]
#         local_path = os.path.join(os.path.dirname(__file__), filename)
#         if not os.path.exists(local_path):
#             return jsonify({"error": f"Model file {filename} not found"}), 500
#         models[model_name] = load_model(config, local_path)

#     model = models[model_name]

#     # Nhận file ảnh từ request
#     file = request.files["file"]
#     tmp_path = os.path.join("/tmp", "upload.jpg")   
#     if not os.path.exists(tmp_path):
#         os.makedirs(tmp_path)
#     file.save(tmp_path)

#     label, confidence, _ = predict_image(tmp_path, model)

#     return jsonify({
#         "model": model_name,
#         "prediction": label,
#         "confidence": round(confidence, 4)
#     })

##Dùng chạy local
@app.route("/predict/<model_name>", methods=["POST"])
def predict(model_name):
    if model_name not in MODEL_CONFIGS:
        return jsonify({"error": "Invalid model name"}), 400

    # Lazy loading: chỉ load 1 lần
    if model_name not in models:
        config, filename = MODEL_CONFIGS[model_name]
        local_path = os.path.join(os.path.dirname(__file__), filename) 
        if not os.path.exists(local_path):
            return jsonify({"error": f"Model file {filename} not found"}), 500
        models[model_name] = load_model(config, local_path)

    model = models[model_name]

    # Nhận file ảnh từ request
    file = request.files["file"]
    file_path = "temp.jpg"
    file.save(file_path)

    label, confidence, _ = predict_image(file_path, model)

    return jsonify({
        "model": model_name,
        "prediction": label,
        "confidence": round(confidence, 4)
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)
