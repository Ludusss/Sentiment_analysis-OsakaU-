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
import librosa
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import moviepy.editor as moviepy
from sklearn import preprocessing
import time
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Paths
UPLOAD_FOLDER = '../recordings'
WEBM_FILE_PATH = "../recordings/test.webm"
WAV_FILE_PATH = "../recordings/test.wav"
BUCKET_NAME = "ludus_sentiment-analysis"
SAMP_RATE = 22050
SAMP_RATE_UPLOAD = 16000

sys.path.append(os.path.abspath("/Users/ludus/Projects/Sentiment_analysis-OsakaU-/"))
sys.path.append(os.path.abspath("C:/Projects/Sentiment_analysis-OsakaU-/"))
from model import LSTM, LSTM1, MLP

# Initialize models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
client = speech.SpeechClient()    # Connect to google asr
#bert = SentenceTransformer('all-mpnet-base-v2')    # Load s-bert model for text feature extractions
sentiment = SentimentIntensityAnalyzer()

"""model_text = LSTM1(input_feature_size=768, hidden_size=128, n_classes=4, n_layers=1, device=device)
text_model_info = torch.load("../saved_models/text_lstm/4_model_acc_67.44.t")"""

model_audio = MLP(input_feature_size=33, hidden_size=118, n_classes=3, n_layers=1, device=device)
audio_model_info = torch.load("../saved_models/audio_mlp/ESD/3_model_ESD_acc_84.85.a")
model_audio.load_state_dict(audio_model_info['model_state_dict'])
for name, param in model_audio.state_dict().items():
    if name == "fc.weight":
        param[:][0] = param[:][0] #+ 0.19045114591335294  # 0.23364777586901642 tuned on test
    if name == "fc1.weight":
        param[:][0] = param[:][0] #+ 0.014296908872680062  # 0.012529474944793237 tuned on test
model_audio.eval()
model_audio.zero_grad()

mean_arr = np.genfromtxt("../useful_variables/train/train_mean.csv")
variance_arr = np.genfromtxt("../useful_variables/train/train_variance.csv")
"""mean_arr = np.genfromtxt("../useful_variables/val/val_mean.csv")
variance_arr = np.genfromtxt("../useful_variables/val/val_variance.csv")
mean_arr = np.genfromtxt("../useful_variables/test/test_mean.csv")
variance_arr = np.genfromtxt("../useful_variables/test/test_variance.csv")"""

# Initialize app
app = Flask(__name__)
cors = CORS(app, resources={r"/sentiment": {"origins": "*"}, r"/sentiment_upload": {"origins": "*"}})


def get_audio_emotion_4(label):
    labels = {0: "Angry", 1: "Happy", 2: "Sad", 3: "Neutral"}
    #labels = {0: "Negative", 1: "Positive",  2: "Neutral"}
    return labels[label]


def get_audio_emotion_3(label):
    labels = {0: "Negative", 1: "Positive", 2: "Neutral"}
    return labels[label]


def get_text_emotion(label):
    labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return labels[label]


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

@app.route('/sentiment_upload', methods=(['POST']))
def get_sentiment_upload():
    file = request.files.get("test")
    filename = secure_filename(file.filename)
    file.save(os.path.join(UPLOAD_FOLDER, filename))
    upload_blob(BUCKET_NAME, WAV_FILE_PATH, WAV_FILE_PATH.split("/")[-1])
    audio = speech.RecognitionAudio(uri="gs://ludus_sentiment-analysis/test.wav")

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=SAMP_RATE_UPLOAD,
        language_code="en-US",
    )

    response = client.recognize(config=config, audio=audio)

    sentence = ""
    for res in response.results:
        print("Transcript: {}".format(res.alternatives[0].transcript))
        sentence = res.alternatives[0].transcript

    if sentence == "":
        return "null", "***Transcription failed***"

    """sentence_embedding = np.asarray(list(bert.encode(sentence, batch_size=1)))
    sentence_embedding = torch.Tensor(sentence_embedding.reshape(1, 1, 768)).to(device)
    output = torch.nn.functional.softmax(model_text(sentence_embedding), dim=1)"""
    output = sentiment.polarity_scores(sentence)
    print(output)
    print(sentence)
    sentiment_scores = np.array([output["neg"], output["neu"], output["pos"]])
    output_label_text = get_text_emotion(np.argmax(sentiment_scores))

    if output_label_text == "Neutral":
        audio_input = []
        y, _sr = librosa.load(WAV_FILE_PATH, sr=SAMP_RATE)
        f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C0'),
                                fmax=librosa.note_to_hz('C5'))
        if np.isnan(f0).all():  # If pitch extraction fails discard utterance
            print("***Librosa failed to extract audio features using text predicted label***")
            return [output_label_text, "fail"], sentence
        mfcc = librosa.feature.mfcc(y=y, sr=SAMP_RATE)
        chroma_cq = librosa.feature.chroma_cqt(y=y, sr=SAMP_RATE, fmin=librosa.note_to_hz('C2'), bins_per_octave=24)
        audio_input.append(np.nanmean(f0))
        audio_input.extend(np.mean(mfcc, axis=1))
        audio_input.extend(np.mean(chroma_cq, axis=1))
        audio_input = np.array(audio_input)
        audio_input = np.divide(np.subtract(audio_input, mean_arr), variance_arr).reshape(1, -1)
        print(audio_input)
        print(model_audio(torch.Tensor(audio_input)))
        output = torch.softmax(model_audio(torch.Tensor(audio_input)), dim=1)
        output_label = torch.argmax(output[0])
        return [output_label_text, get_audio_emotion_3(output_label.item())], sentence
    else:
        return [output_label_text, "not used"], sentence

@app.route('/sentiment', methods=(['POST']))
def get_sentiment():
    file = request.files.get("test")
    filename = secure_filename(file.filename)
    file.save(os.path.join(UPLOAD_FOLDER, filename))
    upload_blob(BUCKET_NAME, WEBM_FILE_PATH, WEBM_FILE_PATH.split("/")[-1])
    audio = speech.RecognitionAudio(uri="gs://ludus_sentiment-analysis/test.webm")

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
        sample_rate_hertz=48000,
        language_code="en-US",
    )

    response = client.recognize(config=config, audio=audio)

    sentence = ""
    for res in response.results:
        print("Transcript: {}".format(res.alternatives[0].transcript))
        sentence = res.alternatives[0].transcript

    if sentence == "":
        return "null", "***Transcription failed***"

    """sentence_embedding = np.asarray(list(bert.encode(sentence, batch_size=1)))
    sentence_embedding = torch.Tensor(sentence_embedding.reshape(1, 1, 768)).to(device)
    output = torch.nn.functional.softmax(model_text(sentence_embedding), dim=1)"""
    output = sentiment.polarity_scores(sentence)
    print(output)
    print(sentence)
    sentiment_scores = np.array([output["neg"], output["neu"], output["pos"]])
    output_label_text = get_text_emotion(np.argmax(sentiment_scores))

    if output_label_text == "Neutral":
        audio_input = []
        y, _sr = librosa.load(WEBM_FILE_PATH, sr=SAMP_RATE)
        f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C0'),
                                fmax=librosa.note_to_hz('C5'))
        if np.isnan(f0).all():  # If pitch extraction fails discard utterance
            print("***Librosa failed to extract audio features using text predicted label***")
            return [output_label_text, "fail"], sentence
        mfcc = librosa.feature.mfcc(y=y, sr=SAMP_RATE)
        chroma_cq = librosa.feature.chroma_cqt(y=y, sr=SAMP_RATE, fmin=librosa.note_to_hz('C2'), bins_per_octave=24)
        audio_input.append(np.nanmean(f0))
        audio_input.extend(np.mean(mfcc, axis=1))
        audio_input.extend(np.mean(chroma_cq, axis=1))
        audio_input = np.array(audio_input)
        audio_input = np.divide(np.subtract(audio_input, mean_arr), variance_arr).reshape(1, -1)
        print(audio_input)
        print(model_audio(torch.Tensor(audio_input)))
        output = torch.softmax(model_audio(torch.Tensor(audio_input)), dim=1)
        output_label_audio = torch.argmax(output[0])
        return [output_label_text, get_audio_emotion_3(output_label_audio.item())], sentence
    else:
        return [output_label_text, "not used"], sentence
