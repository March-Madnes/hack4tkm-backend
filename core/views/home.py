from flask import Blueprint, render_template, jsonify, request
from tensorflow.keras.models import load_model
from PIL import Image
import io
import numpy as np

home =Blueprint('home',__name__)

def predict(image, model):
    '''
    Input the image and model, this function outputs the prediction as:
        1. The class with the highest probability
        2. A dictionary containing each class with their corresponding probability
    '''
    
    LABELS = ["Red soil", "Black Soil", "Clay soil", "Alluvial soil"]
    label_encoder = {label: idx for idx, label in enumerate(LABELS)}
    label_decoder = {idx: label for idx, label in enumerate(LABELS)}
    label_encoder, label_decoder
    IMAGE_SIZE = 128
    image = Image.open(io.BytesIO(image))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    resized_img = image.resize((IMAGE_SIZE, IMAGE_SIZE))

    resized_img_array = np.array(resized_img)
    resized_img_array = resized_img_array.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3)
    probabilities = model.predict(resized_img_array).reshape(-1)
    pred = LABELS[np.argmax(probabilities)]
    return {'pred': pred, 'probabilities': {label: float(prob) for label, prob in zip(LABELS, probabilities)}}

@home.route('/')
def home_html():
    return render_template('home.html')

@home.route('/api')
def home_latest():
    return jsonify({'message' : 'Hello Server'})

@home.route('/api/predict', methods=['POST'])
def predict_moisture():
    model = load_model('core/trained_models/soil_classification.h5')
    image = request.files.get('image')
    if image:
        img_bytes = image.read()
        pred = predict(img_bytes, model)

        return jsonify({'message' : pred})
    
    return 'no image found'