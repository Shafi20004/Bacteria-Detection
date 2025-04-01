import gc
import torch
import shutil
from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
import pickle
from ultralytics import YOLO

# Clean up memory
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Optionally clean old YOLO prediction folders (like predict/)
PRED_FOLDER = 'static/predicted'
predict_subfolder = os.path.join(PRED_FOLDER, 'predict')
if os.path.exists(predict_subfolder):
    shutil.rmtree(predict_subfolder)




app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
PRED_FOLDER = 'static/predicted'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PRED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PRED_FOLDER'] = PRED_FOLDER

# Load YOLO model
with open("bacterial_model.pkl", "rb") as f:
    model = pickle.load(f)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            pred_path = os.path.join(app.config['PRED_FOLDER'], filename)

            file.save(upload_path)

            # Run YOLO and save output directly to pred folder
            model.predict(
                source=upload_path,
                conf=0.25,
                save=True,
                project=app.config['PRED_FOLDER'],
                name='',  # save as static/predicted/filename.jpg
                exist_ok=True
            )

            return render_template('index.html', uploaded=True, pred_img=filename)

    return render_template('index.html', uploaded=False)


if __name__ == '__main__':
    app.run(debug=True)
