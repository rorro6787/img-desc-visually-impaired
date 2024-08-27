from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import gdown
import os
import sys

cd = os.getcwd()

def preprocess_image(img_path, img_size):
    # Load the image
    img = image.load_img(img_path, target_size=img_size)
    
    # Convert the image to a numpy array
    img_array = image.img_to_array(img)
    
    # Expand dimensions to match the model input shape (batch size dimension)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalize the image array (if required by your preprocessing)
    img_array /= 255.0  # Assuming images were scaled during training

    return img_array

def predict_image(model, img_array, class_names):
    # Make predictions
    predictions = model.predict(img_array)

    print(f"predictions: {predictions}")
    
    # Get the predicted class index
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    
    # Get the predicted class name
    predicted_class_name = class_names[predicted_class_index]
    
    return predicted_class_name, predictions

def inference_image(filename:str):

    if not os.path.exists(f"{cd}/trainedModel.h5"):
        print(f"{cd}/trainedModel.h5 does not exist. Downloading...")
        # Download the model file from Google Drive
        gdown.download("https://drive.google.com/uc?id=1vl7FGRt3Uq70cD2Ue787Pq5YbQUMsIY3", "trainedModel.h5", quiet=False)
    else:
        print(f"{cd}/trainedModel.h5 already exists. Skipping download.")

    new_model = load_model(f"{cd}/trainedModel.h5")

    # Define the image size (must match the size used during model training)
    img_size = (224, 224)

    # Define the path to the image you want to classify
    img_path = sys.argv[1]

    # Load and preprocess the image
    img_array = preprocess_image(img_path, img_size)

    class_names = ['Cloudy', 'Rain', 'Shine', 'Sunrise']

    # Make a prediction
    predicted_class_name, predictions = predict_image(new_model, img_array, class_names)

    # Print the results
    information = ""
    information += f'Predicted Class: {predicted_class_name}\n'
    information += f'Class Probabilities: {predictions}\n'

    print(f'Predicted Class: {predicted_class_name}')
    print(f'Class Probabilities: {predictions}')
    
    return information