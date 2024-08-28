from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the saved model
model = load_model('plant_disease_classifier_model.h5')

# Define the class labels
class_lab = ['Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___healthy',
             'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
             'Tomato___Spider_mites', 'Two-spotted_spider_mite', 'Tomato___Target_Spot',
             "Tomato___Tomato_mosaic_virus", "Tomato___Tomato_Yellow_Leaf_Curl_Virus"]

# Directory to save uploaded images
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define the corresponding medicine for each class label
medicine = {
    'Tomato___Bacterial_spot': 'Remove and destroy infected plants, avoid overhead watering, use copper-based fungicides.',
    'Tomato___Early_blight': 'Remove affected leaves, apply copper fungicides, mulch to prevent soil splash.',
    'Tomato___healthy': 'No disease detected. No treatment required.',
    'Tomato___Late_blight': 'Remove infected leaves, increase air circulation, avoid overhead watering.',
    'Tomato___Leaf_Mold': 'Remove infected leaves, avoid overhead watering, apply copper fungicides.',
    'Tomato___Septoria_leaf_spot': 'Spray affected plants with water, use insecticidal soap or neem oil, introduce predatory mites.',
    'Tomato___Spider_mites': 'Remove infected leaves, avoid overhead watering, apply copper fungicides.',
    'Two-spotted_spider_mite': 'Remove and destroy infected plants, control aphids, plant disease-resistant varieties.',
    'Tomato___Target_Spot': 'Remove and destroy infected plants, control whiteflies with insecticidal soap or neem oil.',
    "Tomato___Tomato_mosaic_virus": 'Rotate crops, plant disease-resistant tomato varieties, practice good garden hygiene.',
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": 'Rotate crops, avoid overhead watering, prune lower leaves, maintain good air circulation.'
}
pesticides = {
    'Tomato___Bacterial_spot':'Copper-based fungicides, sulfur-based fungicides',
    'Tomato___Early_blight':'Copper fungicides, sulfur fungicides, chlorothalonil, mancozeb',
    'Tomato___healthy': 'No disease detected. No treatment required.',
    'Tomato___Late_blight':'Copper fungicides, chlorothalonil',
    'Tomato___Leaf_Mold':'Copper fungicides, chlorothalonil',
    'Tomato___Septoria_leaf_spot':'Copper fungicides, chlorothalonil',
    'Tomato___Spider_mites':'Neem oil, pyrethrin, spinosad, insecticidal soap',
    'Tomato___Target_Spot':'Copper fungicides, chlorothalonil',
    "Tomato___Tomato_mosaic_virus":'Neem oil, pyrethrin, spinosad, insecticidal soap',
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus":'Neem oil, pyrethrin, spinosad, insecticidal soap'
}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


@app.route('/')
def index():
    return render_template('index1.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        uploaded_file = request.files['image']

        # Secure the filename and save the file
        filename = secure_filename(uploaded_file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        uploaded_file.save(file_path)

        # Now pass this file_path to your model prediction code
        img = image.load_img(file_path, target_size=(299, 299))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        pred = model.predict(img_array)
        pred_class = np.argmax(pred)
        predicted_label = class_lab[pred_class]

        # You can add more logic here if needed

        return render_template('result2.html', predicted_label=predicted_label)

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return "An error occurred during prediction"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)