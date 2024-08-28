import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np

# Load the pre-trained model
model = load_model("plant_disease_classifier_model.h5")

# Define the class labels
class_lab = [
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___healthy',
    'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites', 'Two-spotted_spider_mite', 'Tomato___Target_Spot',
    "Tomato___Tomato_mosaic_virus", "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
]

def predict_disease(image_path):
    """Function to predict plant disease from an image."""
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(299, 299, 3))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Make a prediction
    pred = model.predict(img_array)
    pred_class = np.argmax(pred)

    # Return the predicted class label
    return class_lab[pred_class]

# Example usage
if __name__ == "__main__":
    # The path is now passed as an argument to the function
    path = "C:/Users/NAVEEN PRASATH/Downloads/Symptoms-and-causal-organism-of-leaf-mold-of-tomato-a-Typical-symptom-of-a-tomato-leaf_Q320.jpg"
    predicted_class = predict_disease(path)
    print("The predicted class:", predicted_class)
