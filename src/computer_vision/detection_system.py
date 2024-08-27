import cv2
import os
import sys
from ultralytics import YOLO
import shutil
import random
from zipfile import ZipFile
import gdown
import time
import computer_vision.heat_map as hm
import computer_vision.inference_weather as wm

def extract_entities_image(sourcePath:str):

    model = YOLO("yolov8x.pt", task="detect")
    cd = os.getcwd() 

    original_path = os.path.dirname(sourcePath)
    results = model(sourcePath, save = True, project=original_path)
    
    dimensions = "Image dimensions: (width=" + str(cv2.imread(sourcePath).shape[1]) + ") x (height="+ str(cv2.imread(sourcePath).shape[0]) + ")\n"
    information = []
    heat_map_path = hm.heat_map(sourcePath)
    heat_map_array = hm.load_npy(heat_map_path)
    max_deepth = "Max deepth: " + str(heat_map_array.max()) + "\n"
    min_deepth = "Min deepth: " + str(heat_map_array.min()) + "\n"
    weather = wm.inference_image(sourcePath)

    for result in results:
        boxes = result.boxes  # This contains the bounding boxes for detected objects
        # Indent the following code block
        for box in boxes:
            # Extract the bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()  # top-left and bottom-right corners
            
            # Extract other information
            confidence = box.conf[0].item()  # confidence score
            class_id = box.cls[0].item()  # class id
            class_name = model.names[int(class_id)]  # class name
            
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # Print or store the results
            information.append(f"{class_name} at coordinates: [{x1}, {y1}, {x2}, {y2}] with heat deepth of {heat_map_array[center_y][center_x]} in the centre of the image\n")
    
    carpeta = os.path.join(cd, '__pycache__')
    if os.path.isdir(carpeta):
        shutil.rmtree(carpeta)
        
    return dimensions, max_deepth, min_deepth, weather, information