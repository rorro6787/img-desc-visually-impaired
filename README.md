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
- [Contributing](#contributing)

## Introduction (tested on Python 3.10.12)

The objective of this project is to create a system that provides comprehensive and accessible descriptions of images for people with visual impairments. By combining advanced object detection with natural language processing, the system can describe the contents of an image in detail, including identifying objects, their relationships, and contextual information.

## Features

- **Object Detection**: Utilizes YOLO (You Only Look Once) for identifying objects in images.
- **Natural Language Generation**: Employs large language models (LLMs) to create descriptive text based on detected objects.
- **Real-Time Processing**: Generates immediate descriptions through real-time image processing.
- **Evaluation Scripts**: Includes tools for assessing the quality and accuracy of generated descriptions.
- **Development Phases**: View our Trello board using this link [Link to our Trello Board](https://trello.com/invite/b/66cc95674da7ab502f627c06/ATTI256c41b0fb855982e0329a22659c13527927C238/image-description-visually-impaired-people) to explore the various phases and steps taken in building the system
- **Documentation**: Detailed PDF explaining the implementation and code in the following link [Download the PDF document](https://github.com/rorro6787/ImageTracking/blob/gamma/Image_Tracking_YOLO.pdf)

## Requirements

- Python 3.x
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
    python -m venv venv
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

1. **Prepare the dataset**: Ensure your images are ready for processing.

2. **Run object detection and description generation**:

   - To generate descriptions for an image, use the following script:

     ```sh
     python generate_description.py --source=image_path
     ```

   This script processes the image at `image_path`, detects objects, and uses an LLM to generate a descriptive text of the image. 

3. **View and save results**:

   - The script will output the generated description to the console and save the processed image with annotations if required.

## Dataset

The dataset should consist of diverse images that are representative of real-world scenes. You can use publicly available datasets or annotate your own dataset to train and evaluate the system. For best results, include a variety of objects and contexts.

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
