import os
import sys
from computer_vision.detection_system import extract_entities_image
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain.chains import LLMChain
import pyttsx3

def load_api_keys(filepath):
    api_keys = {}
    with open(filepath, 'r') as file:
        for line in file:
            key_name, key_value = line.strip().split(':', 1)
            api_keys[key_name.strip()] = key_value.strip()
    return api_keys

# Load API keys from the file
api_keys = load_api_keys('apikeys.txt')
os.environ["OPENAI_API_KEY"] = api_keys.get('openai', '')

dimensions, maxD, minD, weather, info = extract_entities_image(sys.argv[1])

formatted_info = ""
for element in info:
    formatted_info += element
    formatted_info += '\n'


llm = OpenAI(model="gpt-3.5-turbo-instruct",
             temperature=0.3,
             max_tokens=2500)

prompt = PromptTemplate(
    template = """I have an image analysis where objects are detected and identified using a YOLO model, and depth information 
                is extracted using a MiDaS model. The data is structured as follows: each detected object is identified 
                by its type (e.g., 'car') and its color and is associated with a bounding box defined by coordinates (x1, y1, x2, y2). 
                Additionally, each object has a heatmap value representing the distance from the camera, with higher values indicating closer 
                proximity. Here is the array of detected objects with their corresponding bounding box coordinates and heatmap values: {info}
                You get as well to have some extra information about the image, such as the dimensions of the image {dimensions}, 
                the minimum and maximum depths of the image computed by MiDaS {minD} and {maxD}, and the weather on the image {weather}.
                Please analyze this information to determine in a nice description without talking about the coordendates or other
                types of numbers, the following:

                The weather conditions in the image.
                The total number of objects in the image.
                The type of each object.
                The relative positioning of the objects (left, right) based on their heatmap values to identify which objects are in front of others.
                Name them as people, animal, or whatever they are, not as objects and defenitely do not use any tecnical words like boxes and number meassures 
                in your description.
            """,
    input_variables=['dimensions', 'minD', 'maxD', 'weather','info']
)

chain = prompt | llm

input_data = {
    'dimensions': dimensions,
    'minD': minD,
    'maxD': maxD,
    'weather': weather,
    'info': formatted_info
}

# Run the chain with the input data
detailed_description = chain.invoke(input_data)

# Print or use the generated detailed description
print(detailed_description)

# Initialize the TTS engine
engine = pyttsx3.init()

# List all available voices
voices = engine.getProperty('voices')

# Select the voice by index
voice_index = 19                # Index for English (Great Britain) (male)

# Set the selected voice
engine.setProperty('voice', voices[voice_index].id)

# Speak the text
engine.say(detailed_description)
engine.runAndWait()