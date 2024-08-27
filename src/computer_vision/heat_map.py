import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def heat_map(filename: str):
    # Download image or simply use the image path
    # url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
    # urllib.request.urlretrieve(url, filename)

    # Configuring the model
    model_type = "DPT_Large"      # MiDaS v3 - Large      (highest accuracy, slowest inference speed)
    # model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
    # model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

    # Cargar el modelo de MiDaS
    midas = torch.hub.load("intel-isl/MiDaS", model_type)

    # Configuring device (GPU or CPU)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()

    # Download the necessary transformations
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    # Select the transformation according to the model type
    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    # Download image and convert to RGB
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Prepare the image for prediction  
    input_batch = transform(img).to(device)

    # Do the prediction without gradients
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    # Convert the prediction to a numpy array
    output = prediction.cpu().numpy()

    original_path = os.path.dirname(filename)

    # Save the depth image as an image file (heat map)
    depth_filename = os.path.join(original_path, "depth_image.png")
    plt.imsave(depth_filename, output, cmap='plasma')

    # Save the depth array as a .npy file
    npy_filename = os.path.join(original_path, "depth_array.npy")
    np.save(npy_filename, output)

    # return the path of the .npy
    return npy_filename

def show_heat_map(npy_filename: str):
    # Cargar el archivo .npy
    # npy_filename = "name.npy"
    depth_array = np.load(npy_filename)
    # Verificar las dimensiones del array cargado
    print("Array Dimensions:", depth_array.shape)

    # Mostrar la imagen usando matplotlib
    plt.imshow(depth_array, cmap='plasma')
    plt.colorbar()
    plt.title('Deepness Heat Map')
    plt.show()

def load_npy(npy_filename: str):
    # Download the .npy file
    depth_array = np.load(npy_filename)
    return depth_array
    