from flask import Flask, render_template, request, send_file
import os
from werkzeug.utils import secure_filename
# from image_processing import process_image  # Assume you have this function to handle image processing
from src.image_llm_description import process_image
app = Flask(__name__)
app.config['PROCESSED_FOLDER'] = 'processed'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
            file.save(file_path)
            
            # Process the image and get results
            depth_image_path, objects_image_path, description = process_image(file_path)
 
            return render_template('index.html',
                                   original_image=file_path,
                                   depth_image=depth_image_path,
                                   objects_image=objects_image_path,
                                   description=description)
    
    return render_template('index.html')

@app.route('/display/<path:filename>')
def display_image(filename):
    # Use send_file to serve the file from the specified path
    return send_file(filename)

if __name__ == '__main__':
    app.run(debug=True)