from flask import Flask, render_template, request, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
from process_files import prepare_dataset
from train_llm import train_model
import torch

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)

# Set the CUDA_VISIBLE_DEVICES environment variable
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use the first GPU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"App.py: Using device: {device}")

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'json'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_and_train(file_type, file_path):
    try:
        dataset = prepare_dataset(file_type, file_path)
        if not dataset:
            flash('Error: Dataset preparation failed. The prepared dataset is empty.')
            return
        
        train_model(dataset, device)  # Pass the device to train_model
        flash('Training completed successfully')
    except Exception as e:
        flash(f'Error during processing or training: {str(e)}')
        app.logger.error(f'Error in process_and_train: {str(e)}', exc_info=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file_type = request.form['file_type']
        
        if file_type == 'youtube':
            youtube_url = request.form['youtube_url']
            flash('YouTube URL submitted successfully')
            process_and_train(file_type, youtube_url)
            return redirect(url_for('index'))
        
        elif file_type in ['pdf', 'json']:
            if 'file' not in request.files:
                flash('No file part')
                return redirect(request.url)
            file = request.files['file']
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                flash(f'{file_type.upper()} file uploaded successfully')
                process_and_train(file_type, file_path)
                return redirect(url_for('index'))
        
        flash('Invalid submission')
        return redirect(url_for('index'))
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)