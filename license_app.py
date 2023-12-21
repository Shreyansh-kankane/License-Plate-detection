from flask import Flask
from flask import request
from flask_cors import CORS, cross_origin
from flask import jsonify
import base64
from io import BytesIO
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from local_utils import detect_lp
from os.path import splitext,basename
from tensorflow.keras.models  import model_from_json
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.mobilenet_v2 import preprocess_input
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
import glob
from PIL import Image
import pytesseract
import easyocr

reader = easyocr.Reader(['en'])

def load_model(path):
    try:
        path = splitext(path)[0]
        with open('%s.json' % path, 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json, custom_objects={})
        model.load_weights('%s.h5' % path)
        # print("Model Loaded successfully...")
        # print("Detecting License Plate ... ")
        return model
    except Exception as e:
        print(e)
wpod_net_path = "models/wpod-net.json"
wpod_net = load_model(wpod_net_path)


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

def preprocess_image(image_uploaded,resize=False):

    # Read the image from the file
    image = Image.open(image_uploaded)

    # Convert the image to a NumPy array
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    if resize:
        img = cv2.resize(img, (224,224))
    return img

def get_plate(image_uploaded, Dmax=608, Dmin = 608):
    vehicle = preprocess_image(image_uploaded)
    ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    _ , LpImg, _, cor = detect_lp(wpod_net, vehicle, bound_dim, lp_threshold=0.5)
    return vehicle, LpImg, cor

def convert_base64_to_image(base64_string):
    # Remove header if present
    if 'data:image' in base64_string:
        base64_string = base64_string.split(',')[-1]

        # Decode Base64 to binary
        image_data = base64.b64decode(base64_string)

        # Create a file-like object from the binary data
        image_file = BytesIO(image_data)
        return image_file
    else:
        return None


@app.route('/license/', methods=['POST'])
@cross_origin()
def prediction():
    print("request method entered")
    # if(request.method == "POST"):

    #     return "POST request"
    # else:
    #     return request.method

    # print(request.get_json())
    jobj = request.get_json()
    if 'image' not in jobj.keys():
        return None
    
    print("base 64 passed")
    base_64_image = jobj['image']
    uploaded_file = convert_base64_to_image(base_64_image)
    if uploaded_file is None:
        return None
    print("base 64 converted")
    vehicle, LpImg, cor = get_plate(uploaded_file)

    # fig = plt.figure(figsize=(12,6))
    # grid = gridspec.GridSpec(ncols=2,nrows=1,figure=fig)
    # fig.add_subplot(grid[0])
    #plt.axis(False)
    #plt.imshow(vehicle)
    # grid = gridspec.GridSpec(ncols=2,nrows=1,figure=fig)
    # fig.add_subplot(grid[1])
    #plt.axis(False)
    #plt.imshow(LpImg[0])

    converted_image = (LpImg[0] * 255).astype(np.uint8)
    image=Image.fromarray(converted_image)
    #reader = easyocr.Reader(['en'])
    result = reader.readtext(converted_image)
    return jsonify(result[0][1])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)