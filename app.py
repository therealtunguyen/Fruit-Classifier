import sys
import os

current_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.dirname(current_path)
sys.path.append(parent_path)

from flask import Flask, request, jsonify, redirect, render_template, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from utils import get_labels, remove_number, get_image_from_url, delete_in_folder

PORT = 8000
IMG_SIZE = 299
UPLOAD_FOLDER = 'static/'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('index.html', has_image=False, request_url=get_request_address())

@app.route('/api', methods=['POST'])
def api():

    # Get the image from the request
    img = request.files['image']
    if img.filename == '':
        return jsonify({'error': 'No image selected'})
    img = Image.open(img)

    predicted_label = get_prediction(img)

    return jsonify({'prediction': predicted_label})

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    This will render a page with the image and the prediction
    """
    is_from_url = False
    if request.method == 'GET':
        return redirect('/')
    
    img = request.files['image']
    if img.filename == '':
        url = request.form.get('url')
        if url == '':
            return redirect('/')
        img = get_image_from_url(url)
        if img is None:
            return redirect('/')
        is_from_url = True
    else:
        # Delete the image
        delete_in_folder(app.config['UPLOAD_FOLDER'])

        # Save the image
        filename = secure_filename(img.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        img.save(filepath)
        img = Image.open(img)

    predicted_label = get_prediction(img)

    if is_from_url:
        return render_template(
            'index.html',
            request_url=get_request_address(),
            prediction=predicted_label,
            from_url=True,
            has_image=True,
            url=url
        )
    return render_template(
        'index.html',
        request_url=get_request_address(),
        prediction=predicted_label,
        from_url=False,
        has_image=True,
        filename=filename
    )

def get_prediction(image: Image.Image) -> str:
    model = load_model('main_model.h5')

    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image)
    image = image.reshape(1, IMG_SIZE, IMG_SIZE, 3)
    image = image / 255.0

    prediction = model.predict(image)
    prediction = np.argmax(prediction)

    labels = get_labels()
    predicted_label = remove_number(labels[int(prediction)])

    return predicted_label

def get_request_address():
    return f"https://{request.host}/predict"


if __name__ == '__main__':
    app.run(port=PORT)