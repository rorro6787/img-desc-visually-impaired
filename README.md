# Automated Image Description for Visually Impaired People (YOLO + MiDaS + LangChain)

<div align="center">
  <p>
    <a href="https://kajabi-storefronts-production.kajabi-cdn.com/kajabi-storefronts-production/file-uploads/blogs/22606/images/f8d6362-3e5e-c73-a7a4-e54525b5431a_banner-yolov8.png" target="_blank">
      <img width="100%" src="https://kajabi-storefronts-production.kajabi-cdn.com/kajabi-storefronts-production/file-uploads/blogs/22606/images/f8d6362-3e5e-c73-a7a4-e54525b5431a_banner-yolov8.png" alt="YOLO Vision banner"></a>
  </p>
</div>

<div align="center">
  <p>
    <a href="https://yaksoy.github.io/images/hrdepthTeaser.jpg" target="_blank">
      <img width="100%" src="https://yaksoy.github.io/images/hrdepthTeaser.jpg" alt="LangChain banner"></a>
  </p>
</div>

<div align="center">
  <p>
    <a href="https://media.licdn.com/dms/image/D4E12AQHnLknj0EYfBA/article-cover_image-shrink_600_2000/0/1684267676484?e=2147483647&v=beta&t=PrMj5CmpRsqMecZwmySc3LSnQ9jkZNoer75YWJFzJBM" target="_blank">
      <img width="100%" src="https://media.licdn.com/dms/image/D4E12AQHnLknj0EYfBA/article-cover_image-shrink_600_2000/0/1684267676484?e=2147483647&v=beta&t=PrMj5CmpRsqMecZwmySc3LSnQ9jkZNoer75YWJFzJBM" alt="LangChain banner"></a>
  </p>
</div>

This repository contains code and resources for developing a computer vision system that automatically describes images for visually impaired individuals. The project integrates object detection and large language models (LLMs) to provide detailed, meaningful descriptions of visual content.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Docker Deployment](#docker-deployment) 
- [Contributing](#contributing)

## Introduction
The objective of this project is to create a system that provides comprehensive and accessible descriptions of images for people with visual impairments. By combining advanced object detection with natural language processing, the system can describe the contents of an image in detail, including identifying objects, their relationships, and contextual information.

## Features

- **Object Detection**: Utilizes YOLO (You Only Look Once) for identifying objects in images.
- **Natural Language Generation**: Employs large language models (LLMs) to create descriptive text based on detected objects.
- **Real-Time Processing**: Generates immediate descriptions through real-time image processing.
- **Evaluation Scripts**: Includes tools for assessing the quality and accuracy of generated descriptions.
- **Development Phases**: View our Trello board using this link [Link to our Trello Board](https://trello.com/invite/b/66cc95674da7ab502f627c06/ATTI256c41b0fb855982e0329a22659c13527927C238/image-description-visually-impaired-people) to explore the various phases and steps taken in building the system
- **Documentation**: Detailed PDF explaining the implementation and code in the following link [Download the PDF document](https://github.com/rorro6787/img-desc-visually-impaired/blob/main/Project_Documentation.pdf)

## Requirements

- Python 3.10.x
- Ultralytics YOLOv8+
- PyTorch
- Transformers (for LLMs)
- Other dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
   
    ```sh
    git clone https://github.com/yourusername/repository_name.git
    ```

2. Navigate to the project directory:
   
    ```sh
    cd repository_name
    ```

3. (Optional) Create a virtual environment:

    ```sh
    sudo apt-get install python3.10-venv
    python3.10 -m venv venv
    .\venv\Scripts\activate  # On macOS/Linux use 'source venv/bin/activate'
    ```

4. Select venv as your Python interpreter (in VSC):

    ```sh
    > Python: Select Interpreter
    .\venv\Scripts\python.exe  # On macOS/Linux use './venv/bin/python'
    ```

5. Install the required packages:
   
    ```sh
    pip install -r requirements.txt
    ```

6. If you add more dependencies, update the requirements file using:

    ```sh
    pip freeze > requirements.txt
    ```

## Usage

To use the system for generating image descriptions, follow these instructions:

1. **Configure the apikeys.txt**: Outside the src folder, you need to create an apikeys.txt file where you will insert your OpenAI API key. This step is mandatory; if not done, the application will not work. Your apikeys.txt has to look like this:

     ```sh
     openai: <your-api-key>
     ```

3. **Run the Application**: To use the application, first run the app.py script that hosts the service in your localhost:

     ```sh
     python app.py
     ```

4. **View and save results**: By default, we specified that the service would be deployed on your localhost at port 5000. Copy this into your browser, and you can then use the application without any issues.

## Dataset

The dataset should consist of diverse images that are representative of real-world scenes. You can use publicly available datasets or annotate your own dataset to train and evaluate the system. For best results, include a variety of objects and contexts.

## Docker Deployment
If you have Docker installed in your system and you do not want to install all the dependencies and the python version required to test this app, you can use Docker to run the app in a little virtual machine. The project has been included with a Dockerfile that process all the requirements and creates a docker image of a virtual machine that satisfies all the requiremenmts needed and has the project running perfecylt. Just run the followinf commands:
```ssh
sudo docker build -t prueba .
sudo docker run -it prueba
```
  

## Contributors

- [![GitHub](https://img.shields.io/badge/GitHub-100000?style=flat&logo=github&logoColor=white)](https://github.com/rorro6787) [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/emilio-rodrigo-carreira-villalta-2a62aa250/) **Emilio Rodrigo Carreira Villalta**
- [![GitHub](https://img.shields.io/badge/GitHub-100000?style=flat&logo=github&logoColor=white)](https://github.com/javimp2003uma) [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/javier-montes-p%C3%A9rez-a9765a279/) **Javier Montes Pérez**

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Create a new Pull Request

## Acknowledgements

- Inspired by the latest advancements in computer vision and natural language processing.
