from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
from utils import convert_image, board_to_features
import pickle
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_label = None   # <-- Define early so it exists for GET requests
    image_path = None

    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)

            matrix_8x8 = convert_image(image_path)
            features_26 = board_to_features(matrix_8x8)

            matrix_8x8 = np.array(matrix_8x8).reshape(1, 8, 8)
            features_26 = np.array(features_26).reshape(1, 26)

            probabilities = model.predict([matrix_8x8, features_26])[0]
            predicted_class = int(np.argmax(probabilities))

            label_map = {0: "Equal", 1: "White", 2: "Black"}
            predicted_label = label_map[predicted_class]

    return render_template('index.html', predicted_label=predicted_label, image_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)
