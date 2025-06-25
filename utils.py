import cv2
import pytesseract
import numpy as np
import re
from datetime import datetime
from dateutil import parser as dateparser

def extract_dob_and_face(image_bytes, face_save_path="aadhar_face.jpg"):
    img_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (600, 600))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    text = pytesseract.image_to_string(gray)
    matches = re.findall(r'\d{2}[-/]\d{2}[-/]\d{4}|\d{4}[-/]\d{2}[-/]\d{2}', text)

    dob = None
    age = None
    for match in matches:
        try:
            dob = dateparser.parse(match, dayfirst=True)
            age = (datetime.now() - dob).days // 365
            break
        except:
            continue

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4)

    if len(faces) > 0:
        x, y, w, h = faces[0]
        face_img = img[y:y+h, x:x+w]
        cv2.imwrite(face_save_path, face_img)
        return text, dob, age, face_img, face_save_path
    else:
        return text, dob, age, None, None
