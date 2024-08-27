# import system libs
import os
import sys
from PIL import Image

# import data handling tools
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('darkgrid')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# import Deep learning Libraries
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.optimizers import Adamax
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, BatchNormalization
from keras import regularizers
from keras.preprocessing import image
import zipfile
import gdown

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

cd = os.getcwd()

print ('modules loaded')

def preprocessing():
    # Generate data paths with labels

    zip_path = "archive.zip"

    # Download the file from Google Drive
    gdown.download(f"https://drive.google.com/uc?id=1rbWLaN20yoZ_JZKyMbbhDboTwss12Poo", zip_path, quiet=False)

    # Check if the dataset directory already exists to avoid re-extracting
    if not os.path.exists("dataset"):
        # Extract the archive
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(cd)

    data_dir = f"{cd}/Multi-class Weather Dataset"
    filepaths = []
    labels = []

    folds = os.listdir(data_dir)
    for fold in folds:
        foldpath = os.path.join(data_dir, fold)
        filelist = os.listdir(foldpath)
        for file in filelist:
            fpath = os.path.join(foldpath, file)

            filepaths.append(fpath)
            labels.append(fold)

    # Concatenate data paths with labels into one dataframe
    Fseries = pd.Series(filepaths, name= 'filepaths')
    Lseries = pd.Series(labels, name='labels')
    df = pd.concat([Fseries, Lseries], axis= 1)

    print(df)

    # train dataframe
    train_df, dummy_df = train_test_split(df,  train_size= 0.8, shuffle= True, random_state= 123)

    # valid and test dataframe
    valid_df, test_df = train_test_split(dummy_df,  train_size= 0.6, shuffle= True, random_state= 123)

    # crobed image size
    batch_size = 16
    img_size = (224, 224)
    channels = 3
    img_shape = (img_size[0], img_size[1], channels)

    tr_gen = ImageDataGenerator()
    ts_gen = ImageDataGenerator()

    train_gen = tr_gen.flow_from_dataframe( train_df, x_col= 'filepaths', y_col= 'labels', target_size= img_size, class_mode= 'categorical',
                                        color_mode= 'rgb', shuffle= True, batch_size= batch_size)

    valid_gen = ts_gen.flow_from_dataframe( valid_df, x_col= 'filepaths', y_col= 'labels', target_size= img_size, class_mode= 'categorical',
                                        color_mode= 'rgb', shuffle= True, batch_size= batch_size)

    test_gen = ts_gen.flow_from_dataframe( test_df, x_col= 'filepaths', y_col= 'labels', target_size= img_size, class_mode= 'categorical',
                                        color_mode= 'rgb', shuffle= False, batch_size= batch_size)

    g_dict = train_gen.class_indices        # defines dictionary {'class': index}
    classes = list(g_dict.keys())           # defines list of dictionary's kays (classes), classes names : string
    images, labels = next(train_gen)        # get a batch size samples from the generator

    plt.figure(figsize= (20, 20))

    for i in range(16):
        plt.subplot(4, 4, i + 1)
        image = images[i] / 255             # scales data to range (0 - 255)
        plt.imshow(image)
        index = np.argmax(labels[i])        # get image index
        class_name = classes[index]         # get class of image
        plt.title(class_name, color= 'blue', fontsize= 12)
        plt.axis('off')
    plt.show()

    return train_gen, valid_gen, test_gen

def createAndtrainModel(train_gen, valid_gen, test_gen):

    # Create Model Structure
    img_size = (224, 224)
    channels = 3
    img_shape = (img_size[0], img_size[1], channels)
    class_count = len(list(train_gen.class_indices.keys())) # to define number of classes in dense layer

    # create pre-trained model (you can built on pretrained model such as :  efficientnet, VGG , Resnet )
    # we will use efficientnetb3 from EfficientNet family.
    base_model = tf.keras.applications.efficientnet.EfficientNetB3(include_top= False, weights= "imagenet", 
                                                                input_shape= img_shape, pooling= 'max')

    model = Sequential([
        base_model,
        BatchNormalization(axis = -1, momentum = 0.99, epsilon = 0.001),
        Dense(256, kernel_regularizer = regularizers.l2(l = 0.016), activity_regularizer = regularizers.l1(0.006),
                    bias_regularizer = regularizers.l1(0.006), activation= 'relu'),
        Dropout(rate= 0.45, seed= 123),
        Dense(class_count, activation= 'softmax')
    ])

    model.compile(Adamax(learning_rate = 0.001), loss= 'categorical_crossentropy', 
                metrics= ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), 
                            tf.keras.metrics.AUC()])

    model.summary()

    # TRAINING

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    epochs = 25             # number of epochs to be trained

    model.fit(x=train_gen,
                epochs=epochs,
                verbose=1,
                validation_data=valid_gen, 
                validation_steps=None,
                shuffle=False)
    
    model.save("trainedModel.h5")  # Save the model

    return model


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
    
    # Get the predicted class index
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    
    # Get the predicted class name
    predicted_class_name = class_names[predicted_class_index]
    
    return predicted_class_name, predictions

tr_g, va_g, te_g = preprocessing()

model = createAndtrainModel(tr_g,va_g,te_g)

# Define the image size (must match the size used during model training)
img_size = (224, 224)

# Define the path to the image you want to classify
img_path = sys.argv[1]

# Load and preprocess the image
img_array = preprocess_image(img_path, img_size)

class_names = list(tr_g.class_indices.keys())

# Make a prediction
predicted_class_name, predictions = predict_image(model, img_array, class_names)

# Print the results
print(f'Predicted Class: {predicted_class_name}')
print(f'Class Probabilities: {predictions}')