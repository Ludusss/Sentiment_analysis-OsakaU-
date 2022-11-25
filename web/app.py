from flask import Flask, request
from flask_cors import CORS
from werkzeug.utils import secure_filename
from sentence_transformers import SentenceTransformer
from google.cloud import speech
import sys
import os
from google.cloud import storage
import torch
import csv
import numpy as np

# Paths
UPLOAD_FOLDER = '../recordings'
WAV_FILE_PATH = "../recordings/test.wav"
BUCKET_NAME = "ludus_sentiment-analysis"

sys.path.append(os.path.abspath("/Users/ludus/Projects/Sentiment_analysis-OsakaU-/"))
from model import LSTM

# Initialize models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
client = speech.SpeechClient()    # Connect to google asr
bert = SentenceTransformer('all-mpnet-base-v2')    # Load s-bert model for text feature extractions
model = LSTM(1152, 300, 4, 200, 2, device)          # Initialize Sentiment Analysis Network
model.load_state_dict(torch.load("../saved_models/model_acc_45.61.at")["model_state_dict"])     # Load per-trained model

# Initialize app
app = Flask(__name__)
cors = CORS(app, resources={r"/sentiment": {"origins": "*"}})

def get_emotion(label):
    labels = {0: "Happy", 1: "Sad", 2: "Neutural", 3: "Angry", 4: "Excited", 5: "Frustrated"}
    return labels[label]

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

@app.route('/sentiment', methods=(['POST']))
def get_sentiment():
    file = request.files.get("test")
    filename = secure_filename(file.filename)
    file.save(os.path.join(UPLOAD_FOLDER, filename))
    upload_blob(BUCKET_NAME, WAV_FILE_PATH, WAV_FILE_PATH.split("/")[-1])
    audio = speech.RecognitionAudio(uri="gs://ludus_sentiment-analysis/test.wav")

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
        sample_rate_hertz=48000,
        language_code="en-US",
    )

    response = client.recognize(config=config, audio=audio)

    sentence = ""
    for res in response.results:
        print("Transcript: {}".format(res.alternatives[0].transcript))
        sentence += res.alternatives[0].transcript
    sentence_embedding = np.asarray(list(bert.encode(sentence, batch_size=1)))
    cmd = f"SMILExtract -C opensmile/config/is09-13/IS09_emotion.conf -I {WAV_FILE_PATH} -O ../extracted_data/test.csv -l 0"
    os.system(cmd)
    reader = csv.reader(open("../extracted_data/test.csv", 'r'))
    rows = [row for row in reader]
    last_line = rows[-1]
    audio_features = np.asarray(list(map(lambda x: float(x), last_line[1: 385])))
    input_data = np.concatenate((audio_features, sentence_embedding))
    input_data = torch.Tensor(input_data.reshape(1, 1, 1152)).to(device)
    output = model(input_data)
    output_label = torch.argmax(output)

    if sentence == "":
        return "null", "***Transcription failed***"
    else:
        return get_emotion(output_label.item()), sentence
