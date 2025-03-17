from flask import Flask, request, jsonify, send_from_directory, render_template
import os
from werkzeug.utils import secure_filename
from flask_cors import CORS
from model import predict_cancer

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    prediction = predict_cancer(file_path)  # Call ML model

    return jsonify({'image': filename, 'prediction': prediction})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
