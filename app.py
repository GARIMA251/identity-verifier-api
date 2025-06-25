from flask import Flask, request, jsonify
import cv2
import pytesseract
import numpy as np
import re
from datetime import datetime
from dateutil import parser as dateparser
from deepface import DeepFace

app = Flask(__name__)

def extract_dob_and_crop_face(image_bytes, save_path="id_face.jpg"):
    img_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    text = pytesseract.image_to_string(gray)
    matches = re.findall(r'\d{2}[-/]\d{2}[-/]\d{4}|\d{4}[-/]\d{2}[-/]\d{2}', text)

    dob, age = None, None
    for date_str in matches:
        try:
            dob = dateparser.parse(date_str, dayfirst=True)
            age = (datetime.now() - dob).days // 365
            break
        except:
            continue

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.2, 4)

    if len(faces) > 0:
        x, y, w, h = faces[0]
        face_img = img[y:y+h, x:x+w]
        cv2.imwrite(save_path, face_img)
        return text, dob, age, True
    return text, dob, age, False

@app.route("/verify", methods=["POST"])
def verify_identity():
    if "id_image" not in request.files or "selfie" not in request.files:
        return jsonify({"error": "Missing files"}), 400

    id_bytes = request.files["id_image"].read()
    selfie_bytes = request.files["selfie"].read()

    text, dob, age, face_found = extract_dob_and_crop_face(id_bytes)
    if not face_found:
        return jsonify({"error": "No face found in ID image"}), 400

    with open("selfie.jpg", "wb") as f:
        f.write(selfie_bytes)

    result = DeepFace.verify("id_face.jpg", "selfie.jpg", model_name="SFace", enforce_detection=False)

    return jsonify({
        "dob": dob.strftime("%d-%m-%Y") if dob else None,
        "age": age,
        "eligible": age >= 18 if age else False,
        "face_match": result["verified"],
        "confidence": round((1 - result["distance"]) * 100, 2),
        "ocr_text": text
    })

if __name__ == "__main__":
    app.run()
