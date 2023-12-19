import os
import numpy as np
from flask import Flask, jsonify, request
# from PIL import Image
from keras.models import load_model
from werkzeug.utils import secure_filename
import cv2
import imutils

app = Flask(__name__)
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['MODEL_FILE'] = 'letrec_model.h5'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

model = load_model(app.config['MODEL_FILE'], compile=False)


def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
    key=lambda b:b[1][i], reverse=reverse))
    return (cnts, boundingBoxes)

letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
labels = [l for l in letters ]

def get_letters(img):
    letters = []
    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 30, 150)
    
    countours, _ = cv2.findContours(edged.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(edged.shape[:2],dtype=np.uint8)
    for c in countours:
        if cv2.contourArea(c):
            x,y,w,h  = cv2.boundingRect(c)
            cv2.rectangle(mask,(x,y),(x+w,y+h),(255),-1)

    cnts= cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    
    cnts = imutils.grab_contours(cnts)
    cnts = sort_contours(cnts, method="left-to-right")[0]
    print(len(cnts))
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        print(x, y, w, h)
        if (w >= 5) and (h >= 15):
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi = gray[y:y + h, x:x + w]
            thresh = cv2.threshold(roi, 
                                   0, 255,
                                   cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            thresh = cv2.resize(thresh,
                                (28, 28),
                                interpolation = cv2.INTER_CUBIC)
            thresh = thresh.astype("float32") / 255.0
            thresh = np.expand_dims(thresh, axis=-1)
            thresh = thresh.reshape(1,28,28,1)
            ypred = model.predict(thresh)
            x = labels[ypred.argmax()-1]
            letters.append(x)
    return letters, image

def get_word(letter):
    word = "".join(letter)
    return word


def predict_letter(image_path):
    letters, images = get_letters(image_path)
    prediction = get_word(letters) 
    
    output = ""

    for index, letter in enumerate(prediction):
        output += letter
        if index != len(prediction)-1:
            output += " "
    
    print(output)
    return output

@app.route("/")
def index():
    return "Hello World!"

@app.route("/predict", methods=["POST"])
def predict_letter_route():
    if request.method == "POST":
        image = request.files["image"]
        if image and allowed_file(image.filename):
            filename = secure_filename(image.filename)
            image.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

            predicted_letter = predict_letter(image_path)

            return jsonify({
                "status": {
                    "code": 200,
                    "message": "Success predicting"
                },
                "data": {
                    "predicted_letter": predicted_letter,
                }
            }), 200
        else:
            return jsonify({
                "status": {
                    "code": 400,
                    "message": "Invalid file format. Please upload a JPG, JPEG, or PNG image."
                },
                "data": None,
            }), 400
    else:
        return jsonify({
            "status": {
                "code": 405,
                "message": "Method not allowed"
            },
            "data": None,
        }), 405

if __name__ == "_main_":
    app.run(debug=True,
            host="0.0.0.0",
            port=int(os.environ.get("PORT", 8080)))
