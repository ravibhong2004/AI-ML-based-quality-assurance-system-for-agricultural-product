import json
import numpy as np
from PIL import Image
from keras.models import load_model

def test_model_proc(fn):
    IMAGE_SIZE = 64
    if fn != "":
        print("Loading model...")
        # Load the pre-trained model
        model = load_model("model.h5", compile=True)
        print("Model loaded.")

        # Load config for dynamic thresholds
        with open('config.json', 'r') as f:
            config = json.load(f)
        thresholds = config['quality_thresholds']
        qualities = config['qualities']
        print("Config loaded.")

        # Preprocess the image
        print("Preprocessing image...")
        img = Image.open(fn)
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
        img = np.array(img)
        img = img.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 3)
        img = img.astype('float32') / 255.0
        print("Image preprocessed.")

        # Predict the class
        print("Predicting...")
        prediction = model.predict(img)
        index = np.argmax(prediction)
        confidence_score = prediction[0][index]
        print(f"Prediction done: index={index}, confidence={confidence_score}")

        # Define the class-to-category mapping
        class_mapping = {
            0: {"type": "Corn seed", "qualities": ["Broken", "Discolored", "Pure", "Silkcut"]},
            1: {"type": "Maize seed", "qualities": ["Average", "Bad", "Excellent", "Good", "Worst"]},
            2: {"type": "Soybean seed", "qualities": ["Broken soybeans", "Immature soybeans", "Intact soybeans", "Skin-damaged soybeans", "Spotted soybeans"]}
        }

        if index in class_mapping:
            seed_info = class_mapping[index]
            seed_type = seed_info["type"]
            qualities = seed_info["qualities"]
            # Use the predicted index as the quality index
            if index < len(qualities):
                predicted_quality = qualities[index]
            else:
                predicted_quality = "Unknown"
            return f"Seed Type: {seed_type}\nPredicted Quality: {predicted_quality}\nConfidence: {confidence_score:.2f}\nPrediction Details: {prediction[0]}"
        else:
            return "Unknown seed type"
    else:
        return "No file selected for processing"

# Test on a pure image
image_path = r"testing_set/Corn seed/pure/16134.png"
print(f"Testing image: {image_path}")
result = test_model_proc(image_path)
print("Result:", result)