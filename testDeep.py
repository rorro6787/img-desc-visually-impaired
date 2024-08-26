"""
import cv2
import torch
import urllib.request
import numpy as np
import matplotlib.pyplot as plt

# Descargar la imagen de muestra
url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
urllib.request.urlretrieve(url, filename)

# Configuración del modelo
model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
# model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
# model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

# Cargar el modelo de MiDaS
midas = torch.hub.load("intel-isl/MiDaS", model_type)

# Configurar dispositivo (GPU o CPU)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

# Cargar las transformaciones necesarias
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

# Selección de la transformación según el tipo de modelo
if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

# Cargar la imagen y convertir a RGB
img = cv2.imread(filename)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Preparar la imagen para la predicción
input_batch = transform(img).to(device)

# Realizar la predicción sin gradientes
with torch.no_grad():
    prediction = midas(input_batch)
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

# Convertir la predicción a formato numpy
output = prediction.cpu().numpy()

# Guardar la imagen de profundidad como un archivo de imagen (mapa de calor)
depth_filename = "depth_image.png"
plt.imsave(depth_filename, output, cmap='plasma')
print(f"La imagen de profundidad se ha guardado como {depth_filename}")

# Guardar la matriz de profundidad como un archivo .npy
npy_filename = "depth_array.npy"
np.save(npy_filename, output)
print(f"Los datos de profundidad se han guardado como {npy_filename}")
"""

import numpy as np
import matplotlib.pyplot as plt

# Cargar el archivo .npy
npy_filename = "/home/rorro3382/Desktop/Universidad/4Carrera/ImageTracking/src/depth_array.npy"
depth_array = np.load(npy_filename)
print("siiii"+str(depth_array[600][600]))
# Verificar las dimensiones del array cargado
print("Dimensiones del array:", depth_array.shape)

# Mostrar la imagen usando matplotlib
plt.imshow(depth_array, cmap='plasma')
plt.colorbar()
plt.title('Mapa de Calor de Profundidad')
plt.show()
