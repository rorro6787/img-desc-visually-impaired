# Usar una imagen base de Python
FROM python:3.9-slim

# Establecer el directorio de trabajo en la imagen de Docker
WORKDIR /app

# Copiar el archivo de requisitos a la imagen de Docker
COPY requirements.txt .

# Instalar las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código fuente del proyecto a la imagen de Docker, incluyendo la carpeta adicional
COPY . .

# Exponer el puerto que usará la aplicación
EXPOSE 5000

# Comando por defecto para ejecutar la aplicación
CMD ["python", "app.py"]
