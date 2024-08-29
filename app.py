import json
import os
import time
import random
import string
from gtts import gTTS
from flask import Flask, render_template, request, send_file
import os
from playsound import playsound
from werkzeug.utils import secure_filename
# from image_processing import process_image  # Assume you have this function to handle image processing
from src.image_llm_description import process_image
app = Flask(__name__)
app.config['PROCESSED_FOLDER'] = 'processed'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}

description = ""

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def index():
    global description
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
            file.save(file_path)
            
            # Process the image and get results
            depth_image_path, objects_image_path, description, totalTime = process_image(file_path)
 
            return render_template('index.html',
                                   original_image=file_path,
                                   depth_image=depth_image_path,
                                   objects_image=objects_image_path,
                                   description=description,
                                   totalTime = totalTime)
    
    return render_template('index.html')

@app.route('/process_extra', methods=['POST'])
def process_extra():
    
    def generate_random_id(length=6):
        characters = string.ascii_letters + string.digits
        return ''.join(random.choice(characters) for _ in range(length))


    def initialize_json_file(file_name):
        with open(file_name, 'w') as json_file:
            json_file.write("[]")  # Write an empty list to the file


    def text_to_speech(text, lang, gender):
        # Create "Outputs" folder if it doesn't exist
        output_dir = "Outputs"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Generate a random ID
        id = generate_random_id()

        # Save the input text, ID, language, and gender in a JSON file
        data = {'id': id, 'text': text, 'lang': lang, 'gender': gender}
        with open('log.json', 'a+') as json_file:
            json_file.seek(0)
            try:
                logs = json.load(json_file)
            except json.decoder.JSONDecodeError:
                initialize_json_file('log.json')
                logs = []
            logs.append(data)
            json_file.seek(0)
            json_file.truncate()  # Truncate the file before writing
            json.dump(logs, json_file, indent=4)

        # Create a text-to-speech object
        tld = 'com'  # Default to Google's base voice
        if gender.lower() == 'male':
            if lang == 'en':
                tld = 'com'  # English male voice
            elif lang == 'es':
                tld = 'es'  # Spanish male voice
        elif gender.lower() == 'female':
            if lang == 'en':
                tld = 'co.uk'  # English female voice
            elif lang == 'es':
                tld = 'com.mx'  # Spanish female voice

        tts = gTTS(text=text, lang=lang, tld=tld, slow=False)

        # Save the speech in the "Outputs" folder with the ID in the filename
        output_file = os.path.join(output_dir, f"output_{id}.mp3")
        tts.save(output_file)

        # Print the path to the saved audio file
        print(f"Text converted to speech successfully. Audio file saved at: {output_file}")

        playsound(output_file)

    # Convert text to speech
    text_to_speech(description, 'en', 'female')

    time.sleep(10)

    return render_template('index.html')



@app.route('/display/<path:filename>')
def display_image(filename):
    # Use send_file to serve the file from the specified path
    return send_file(filename)

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host="0.0.0.0", debug=True)