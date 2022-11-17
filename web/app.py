from flask import Flask, request
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = '../recordings'
app = Flask(__name__)
cors = CORS(app, resources={r"/sentiment": {"origins": "*"}})

@app.route('/sentiment', methods=(['POST']))
def get_sentiment():
    file = request.files.get("test")
    filename = secure_filename(file.filename)
    file.save(os.path.join(UPLOAD_FOLDER, filename))
    return "working"
