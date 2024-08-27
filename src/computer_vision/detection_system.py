import cv2
import os
import sys
from ultralytics import YOLO
import shutil
import random
from zipfile import ZipFile
import gdown
import time
import heat_map as hm
import inference_weather as wm

model = YOLO("yolov8x.pt", task="detect")
cd = os.getcwd()          

def extract_entities_image(sourcePath:str):
    original_path = os.path.dirname(sourcePath)
    results = model(sourcePath, save = True, project=original_path)
    information = "Image of dimensions: " + str(cv2.imread(sourcePath).shape[1]) + "x"+ str(cv2.imread(sourcePath).shape[0]) + "\n"
    heat_map_path = hm.heat_map(sourcePath)
    heat_map_array = hm.load_npy(heat_map_path)



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
            information += f"{class_name} at coordinates: [{x1}, {y1}, {x2}, {y2}] with heat deepth of {heat_map_array[center_x][center_y]} in the centre of the image\n"
    
    carpeta = os.path.join(cd, '__pycache__')
    if os.path.isdir(carpeta):
        shutil.rmtree(carpeta)
        
    return information

if __name__ == "__main__":
    print(extract_entities_image("/home/rorro3382/Desktop/Universidad/4Carrera/img-desc-visually-impaired/src/computer_vision/tests/test1/download.jpeg"))