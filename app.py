from flask import Flask, request, jsonify
from utils import extract_dob_and_face
from deepface import DeepFace
import cv2
import numpy as np

app = Flask(__name__)

@app.route("/verify", methods=["POST"])
def verify():
    id_img = request.files.get("id")
    selfie_img = request.files.get("selfie")

    if not id_img or not selfie_img:
        return jsonify({"success": False, "error": "Missing files"}), 400

    text, dob, age, _, id_face_path = extract_dob_and_face(id_img.read())

    selfie_array = np.frombuffer(selfie_img.read(), np.uint8)
    selfie_cv = cv2.imdecode(selfie_array, cv2.IMREAD_COLOR)
    selfie_path = "selfie.jpg"
    cv2.imwrite(selfie_path, selfie_cv)

    if id_face_path:
        result = DeepFace.verify(id_face_path, selfie_path, model_name="SFace", enforce_detection=False)
        return jsonify({
            "success": True,
            "dob": dob.strftime("%d-%m-%Y") if dob else None,
            "age": age,
            "eligible": age >= 18 if age is not None else None,
            "match": result["verified"],
            "confidence": round((1 - result["distance"]) * 100, 2)
        })
    else:
        return jsonify({"success": False, "error": "No face detected in Aadhaar image."}), 400

if __name__ == '__main__':
    app.run()
