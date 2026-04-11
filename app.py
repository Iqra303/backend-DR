import os
import uuid
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
from huggingface_hub import hf_hub_download
from gradcam import generate_gradcam
from lesion_map import generate_lesion_map

# -------- Firebase Imports --------
import firebase_admin
from firebase_admin import credentials, firestore

# ---------------- APP SETUP ----------------
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

model_path = hf_hub_download(
    repo_id="iqrakhawar/eye-disease-model",
    filename="eb7_best.keras"
)

print("Loading model...")
model = load_model(model_path, compile=False)

print("Model loaded!")

# # ---------------- LOAD MODEL ----------------
# print("Loading model...")
# model = load_model("model/eb7_best.keras", compile=False)
# print("Model loaded!")

# ---------------- CLASS & RISK DEFINITIONS ----------------
class_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]

risk_levels = {
    "No DR": "No Risk",
    "Mild": "Very Low Risk",
    "Moderate": "Medium Risk",
    "Severe": "High Risk",
    "Proliferative": "Critical High Risk"
}

IMG_SIZE = (384, 384)

# ---------------- IMAGE PREPROCESS ----------------
def preprocess(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize(IMG_SIZE)
    img = img_to_array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

# ---------------- FIREBASE INIT ----------------
cred = credentials.Certificate("DR.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# ---------------- HOME ----------------
@app.route("/")
def home():
    return "RetinaSight Flask backend running"

# ---------------- PREDICT ----------------
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error":"No file uploaded"}), 400

    file = request.files["file"]

    # Generate unique filename
    filename = f"{uuid.uuid4().hex}_{file.filename}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # -------- Model Prediction --------
    img = preprocess(filepath)
    preds = model.predict(img)[0]

    class_index = int(np.argmax(preds))
    predicted_class = class_names[class_index]
    confidence = float(np.max(preds))
    probs = {class_names[i]: float(preds[i]) for i in range(len(class_names))}

    # -------- Risk Level --------
    risk_level = risk_levels.get(predicted_class, "Unknown")

    # -------- GradCAM --------
    gradcam_path = generate_gradcam(model, filepath)

    # -------- Lesion Map --------
    lesion_path = generate_lesion_map(filepath)

    # -------- Response --------
    response = {
        "filename": filename,
        "prediction": predicted_class,
        "risk_level": risk_level,
        "confidence": round(confidence*100, 2),
        "class_probabilities": {k: round(v*100, 2) for k, v in probs.items()},
        "gradcam_url": f"http://localhost:5000/output/{os.path.basename(gradcam_path)}",
        "lesion_mask_url": f"http://localhost:5000/output/{os.path.basename(lesion_path)}",
        "metrics": {
            "accuracy": 93.15,
            "precision": 93.30,
            "recall": 93.15,
            "f1_score": 93.19,
            "auc_roc": 93,
            "cohen_kappa": 96.91
        }
    }

    # -------- SAVE TO FIRESTORE --------
    try:
        doc_ref = db.collection("predictions").document(filename)
        doc_ref.set(response)
        print(f"Saved prediction to Firestore: {filename}")
    except Exception as e:
        print("Error saving to Firestore:", e)

    return jsonify(response)

# ---------------- SERVE OUTPUT FILES ----------------
@app.route("/output/<filename>")
def output_file(filename):
    return send_from_directory(
        os.path.join(os.path.dirname(__file__), "outputs"),
        filename
    )

# ---------------- RUN SERVER ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render port
    app.run(host="0.0.0.0", port=port)
# import os
# import uuid
# import numpy as np
# from flask import Flask, request, jsonify, send_from_directory
# from flask_cors import CORS
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import img_to_array
# from tensorflow.keras.applications.efficientnet import preprocess_input
# from PIL import Image

# from gradcam import generate_gradcam
# from lesion_map import generate_lesion_map

# # ---------------- APP SETUP ----------------
# app = Flask(__name__)
# CORS(app)

# UPLOAD_FOLDER = "uploads"
# OUTPUT_FOLDER = "outputs"

# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# # ---------------- LOAD MODEL ----------------
# print("Loading model...")
# model = load_model("model/eb7_best.keras", compile=False)
# print("Model loaded!")

# # ---------------- CLASS & RISK DEFINITIONS ----------------
# class_names = ["No DR","Mild","Moderate","Severe","Proliferative"]

# risk_levels = {
#     "No DR": "No Risk",
#     "Mild": "Very Low Risk",
#     "Moderate": "Medium Risk",
#     "Severe": "High Risk",
#     "Proliferative": "Critical High Risk"
# }

# IMG_SIZE = (384, 384)

# # ---------------- IMAGE PREPROCESS ----------------
# def preprocess(img_path):
#     img = Image.open(img_path).convert("RGB")
#     img = img.resize(IMG_SIZE)
#     img = img_to_array(img)
#     img = preprocess_input(img)
#     img = np.expand_dims(img, axis=0)
#     return img

# # ---------------- HOME ----------------
# @app.route("/")
# def home():
#     return "RetinaSight Flask backend running"

# # ---------------- PREDICT ----------------
# @app.route("/predict", methods=["POST"])
# def predict():

#     if "file" not in request.files:
#         return jsonify({"error":"No file uploaded"}), 400

#     file = request.files["file"]

#     filename = f"{uuid.uuid4().hex}_{file.filename}"
#     filepath = os.path.join(UPLOAD_FOLDER, filename)
#     file.save(filepath)

#     # -------- Model Prediction --------
#     img = preprocess(filepath)
#     preds = model.predict(img)[0]

#     class_index = int(np.argmax(preds))
#     predicted_class = class_names[class_index]
#     confidence = float(np.max(preds))
#     probs = {class_names[i]: float(preds[i]) for i in range(len(class_names))}

#     # -------- Risk Level --------
#     risk_level = risk_levels.get(predicted_class, "Unknown")

#     # -------- GradCAM --------
#     gradcam_path = generate_gradcam(model, filepath)

#     # -------- Lesion Map --------
#     lesion_path = generate_lesion_map(filepath)

#     # -------- Response --------
#     return jsonify({
#         "prediction": predicted_class,
#         "risk_level": risk_level,
#         "confidence": round(confidence*100, 2),
#         "class_probabilities": {k: round(v*100, 2) for k, v in probs.items()},
#         "gradcam_url": f"http://localhost:5000/output/{os.path.basename(gradcam_path)}",
#         "lesion_mask_url": f"http://localhost:5000/output/{os.path.basename(lesion_path)}",
#         "metrics": {
#             "accuracy": 93.15,
#             "precision": 93.30,
#             "recall": 93.15,
#             "f1_score": 93.19,
#             "auc_roc": 93,
#             "cohen_kappa": 96.91   # <-- Added Cohen Kappa
#         }
#     })

# # ---------------- SERVE OUTPUT FILES ----------------
# @app.route("/output/<filename>")
# def output_file(filename):
#     return send_from_directory(
#         os.path.join(os.path.dirname(__file__), "outputs"),
#         filename
#     )

# # ---------------- RUN SERVER ----------------
# if __name__ == "__main__":
#     app.run(debug=True)
# # import os
# # import uuid
# # import numpy as np
# # from flask import Flask, request, jsonify, send_from_directory
# # from flask_cors import CORS
# # from tensorflow.keras.models import load_model
# # from tensorflow.keras.preprocessing.image import img_to_array
# # from tensorflow.keras.applications.efficientnet import preprocess_input
# # from PIL import Image

# # from gradcam import generate_gradcam
# # from lesion_map import generate_lesion_map

# # app = Flask(__name__)
# # CORS(app)

# # UPLOAD_FOLDER = "uploads"
# # OUTPUT_FOLDER = "outputs"

# # os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# # os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# # # ---------------- LOAD MODEL ----------------
# # print("Loading model...")
# # model = load_model("model/eb7_best.keras", compile=False)
# # print("Model loaded!")

# # # APTOS class labels
# # class_names = ["No DR","Mild","Moderate","Severe","Proliferative"]

# # IMG_SIZE = (384,384)

# # # ---------------- PREPROCESS ----------------
# # def preprocess(img_path):

# #     img = Image.open(img_path).convert("RGB")
# #     img = img.resize(IMG_SIZE)

# #     img = img_to_array(img)
# #     img = preprocess_input(img)

# #     img = np.expand_dims(img, axis=0)

# #     return img


# # # ---------------- HOME ----------------
# # @app.route("/")
# # def home():
# #     return "RetinaSight Flask backend running"


# # # ---------------- PREDICT ----------------
# # @app.route("/predict", methods=["POST"])
# # def predict():

# #     if "file" not in request.files:
# #         return jsonify({"error":"No file uploaded"}),400

# #     file = request.files["file"]

# #     filename = f"{uuid.uuid4().hex}_{file.filename}"
# #     filepath = os.path.join(UPLOAD_FOLDER, filename)

# #     file.save(filepath)

# #     # -------- Model Prediction --------
# #     img = preprocess(filepath)

# #     preds = model.predict(img)[0]

# #     class_index = int(np.argmax(preds))
# #     predicted_class = class_names[class_index]

# #     confidence = float(np.max(preds))

# #     probs = {
# #         class_names[i]: float(preds[i])
# #         for i in range(len(class_names))
# #     }

# #     print("Prediction:", predicted_class)
# #     print("Confidence:", confidence)
# #     print("Probabilities:", probs)

# #     # -------- GradCAM --------
# #     gradcam_path = generate_gradcam(model, filepath)

# #     # -------- Lesion Map --------
# #     lesion_path = generate_lesion_map(filepath)

# #     return jsonify({

# #         "prediction": predicted_class,
# #         "confidence": round(confidence*100,2),

# #         "class_probabilities": {
# #             k: round(v*100,2) for k,v in probs.items()
# #         },

# #         "gradcam_url":
# #         f"http://localhost:5000/output/{os.path.basename(gradcam_path)}",

# #         "lesion_mask_url":
# #         f"http://localhost:5000/output/{os.path.basename(lesion_path)}",

# #         "metrics": {
# #             "accuracy": 93.15,
# #             "precision": 93.30,
# #             "recall": 93.15,
# #             "f1_score": 93.19,
# #             "auc_roc": 93
# #         }

# #     })


# # # ---------------- OUTPUT FILES ----------------
# # @app.route("/output/<filename>")
# # def output_file(filename):

# #     return send_from_directory(
# #         os.path.join(os.path.dirname(__file__), "outputs"),
# #         filename
# #     )


# # # ---------------- RUN SERVER ----------------
# # if __name__ == "__main__":
# #     app.run(debug=True)
# # # # import os
# # # # import uuid
# # # # import numpy as np
# # # # from flask import Flask, request, jsonify, send_from_directory
# # # # from flask_cors import CORS
# # # # from tensorflow.keras.models import load_model
# # # # from tensorflow.keras.preprocessing.image import img_to_array
# # # # from PIL import Image

# # # # from gradcam import generate_gradcam
# # # # from lesion_map import generate_lesion_map

# # # # app = Flask(__name__)
# # # # CORS(app)

# # # # UPLOAD_FOLDER = "uploads"
# # # # OUTPUT_FOLDER = "outputs"

# # # # os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# # # # os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# # # # print("Loading model...")
# # # # model = load_model("model/eb7_best.keras", compile=False)
# # # # print("Model loaded!")

# # # # class_names = ["No DR","Mild","Moderate","Severe","Proliferative"]

# # # # # Preprocess image
# # # # def preprocess(img_path):
# # # #     img = Image.open(img_path).resize((384,384))  # match model input
# # # #     img = img_to_array(img)/255.0
# # # #     img = np.expand_dims(img,0)
# # # #     return img

# # # # # Home route
# # # # @app.route("/")
# # # # def home():
# # # #     return "Flask backend running"

# # # # # Predict route
# # # # @app.route("/predict", methods=["POST"])
# # # # def predict():

# # # #     if "file" not in request.files:
# # # #         return jsonify({"error":"No file"}), 400

# # # #     file = request.files["file"]
# # # #     filename = f"{uuid.uuid4().hex}_{file.filename}"
# # # #     filepath = os.path.join(UPLOAD_FOLDER, filename)
# # # #     file.save(filepath)

# # # #     img = preprocess(filepath)
# # # #     preds = model.predict(img)[0]

# # # #     predicted_class = class_names[np.argmax(preds)]
# # # #     confidence = float(np.max(preds))
# # # #     probs = {class_names[i]: float(preds[i]) for i in range(len(class_names))}

# # # #     # Generate GradCAM & Lesion maps
# # # #     gradcam_path = generate_gradcam(model, filepath)
# # # #     lesion_path = generate_lesion_map(filepath)

# # # #     # Send response
# # # #     return jsonify({
# # # #         "prediction": predicted_class,
# # # #         "confidence": confidence,
# # # #         "class_probabilities": probs,
# # # #         "gradcam_url": f"http://localhost:5000/output/{os.path.basename(gradcam_path)}",
# # # #         "lesion_mask_url": f"http://localhost:5000/output/{os.path.basename(lesion_path)}",
# # # #         "metrics": {
# # # #             "accuracy": 93.15,
# # # #             "precision": 1.00,
# # # #             "recall": 0.90,
# # # #             "f1_score": 0.905,
# # # #             "auc_roc": 0.97
# # # #         }
# # # #     })

# # # # # Serve outputs
# # # # # Serve outputs
# # # # @app.route("/output/<filename>")
# # # # def output_file(filename):
# # # #     return send_from_directory(
# # # #         os.path.join(os.path.dirname(__file__), "outputs"),
# # # #         filename
# # # #     )

# # # # if __name__ == "__main__":
# # # #     app.run(debug=True)
# # # import os
# # # import uuid
# # # import numpy as np
# # # from flask import Flask, request, jsonify, send_from_directory
# # # from flask_cors import CORS
# # # from tensorflow.keras.models import load_model
# # # from tensorflow.keras.preprocessing.image import img_to_array
# # # from PIL import Image

# # # from gradcam import generate_gradcam
# # # from lesion_map import generate_lesion_map

# # # app = Flask(__name__)
# # # CORS(app)

# # # UPLOAD_FOLDER = "uploads"
# # # OUTPUT_FOLDER = "outputs"
# # # os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# # # os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# # # print("Loading model...")
# # # model = load_model("model/eb7_best.keras", compile=False)
# # # print("Model loaded!")

# # # class_names = ["No DR","Mild","Moderate","Severe","Proliferative"]

# # # # Preprocess image
# # # def preprocess(img_path):
# # #     img = Image.open(img_path).resize((600,600))
# # #     img = img_to_array(img)/255.0
# # #     img = np.expand_dims(img,0)
# # #     return img

# # # @app.route("/")
# # # def home():
# # #     return "Flask backend running"

# # # @app.route("/predict", methods=["POST"])
# # # def predict():
# # #     if "file" not in request.files:
# # #         return jsonify({"error":"No file"}), 400

# # #     file = request.files["file"]
# # #     filename = f"{uuid.uuid4().hex}_{file.filename}"
# # #     filepath = os.path.join(UPLOAD_FOLDER, filename)
# # #     file.save(filepath)

# # #     # Prediction
# # #     img = preprocess(filepath)
# # #     preds = model.predict(img)[0]
# # #     predicted_class = class_names[np.argmax(preds)]
# # #     confidence = float(np.max(preds))
# # #     probs = {class_names[i]: float(preds[i]) for i in range(len(class_names))}

# # #     # GradCAM & Lesion maps
# # #     gradcam_path = generate_gradcam(model, filepath)
# # #     lesion_path = generate_lesion_map(filepath)

# # #     return jsonify({
# # #         "prediction": predicted_class,
# # #         "confidence": confidence,
# # #         "class_probabilities": probs,
# # #         "gradcam_url": f"http://localhost:5000/output/{os.path.basename(gradcam_path)}",
# # #         "lesion_mask_url": f"http://localhost:5000/output/{os.path.basename(lesion_path)}",
# # #         "metrics": {
# # #             "accuracy": 93.15,
# # #             "precision": 1.00,
# # #             "recall": 0.90,
# # #             "f1_score": 0.905,
# # #             "auc_roc": 0.97
# # #         }
# # #     })

# # # @app.route("/output/<filename>")
# # # def output_file(filename):
# # #     return send_from_directory(
# # #         os.path.join(os.path.dirname(__file__), "outputs"),
# # #         filename
# # #     )

# # # if __name__ == "__main__":
# # #     app.run(debug=True)
