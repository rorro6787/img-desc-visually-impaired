import os
import sys
from computer_vision.detection_system import extract_entities_image
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain


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


llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.9)
prompt = PromptTemplate(
    template = """I have detected the objects and features of an image.
                Describe it in great detail for a visually impaired person.
                The dimensions of the image are {dimensions}, the minimum and maximum depths of 
                the image computed by MiDaS are {minD} and {maxD}. The weather on the image is {weather}
                and the here is the list of detected objects and their depths: {info}""",
    input_variables=['dimensions', 'minD', 'maxD', 'weather','info']
)

chain = LLMChain(llm=llm,
                 prompt=prompt)

input_data = {
    'dimensions': dimensions,
    'minD': minD,
    'maxD': maxD,
    'weather': weather,
    'info': formatted_info
}

# Run the chain with the input data
detailed_description = chain.run(input_data)

# Print or use the generated detailed description
print(detailed_description)