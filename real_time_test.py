import argparse
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from Recorder import Recorder
from model import LSTM
from google.cloud import speech
import os
import csv
from google.cloud import storage
from pynput.keyboard import Key, Listener
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "google_cloud_key.json"

RATE = 44100
CHUNK = 1024
WAV_FILE_PATH = "recordings/test.wav"
AVI_FILE_PATH = "/Users/ludus/Projects/Sentiment_analysis-OsakaU-/recordings/test.avi"
BUCKET_NAME = "ludus_sentiment-analysis"


def get_sentiment(label):
    labels = {0: "Happy", 1: "Sad", 2: "Neutural", 3: "Angry", 4: "Excited", 5: "Frustrated"}
    return labels[label]


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

def main():
    parser = argparse.ArgumentParser(description='MOSEI Sentiment Analysis')

    # Model Param
    parser.add_argument('--n_layers', type=int, default=2,
                        help='number of classes in the network (default: 2)')
    parser.add_argument('--output_size', type=int, default=4,
                        help='number of classes in the network (default: 4)')
    parser.add_argument('--hidden_dim', type=int, default=300,
                        help='dimension of hidden vector in LSTM (default: 300)')
    parser.add_argument('--fc_dim', type=int, default=200,
                        help='dimension of fc layer in LSTM (default: 200)')
    parser.add_argument('--dropout', type=int, default=0.5,
                        help='dropout rate lstm (default: 0.5)')
    # Tuning
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size (default: 1)')
    # Testing
    parser.add_argument('--full_dict', type=str, default="/Users/ludus/Projects/Sentiment_analysis-OsakaU-/luo/state_dict/full/full73.94.pt",
                        help='full model pretrained stat dict (default: 73.94)')
    parser.add_argument('--bl', type=bool, default=False,
                        help='Train on the baseline (default: Ture)')
    parser.add_argument('--classifier', type=str, default='mlp',
                        help='use mlp or lstm for decision level fusion (default: mlp)')
    args = parser.parse_args()

    # Initialize models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    client = speech.SpeechClient()    # Connect to google asr
    bert = SentenceTransformer('all-mpnet-base-v2')    # Load s-bert model for text feature extractions
    model = LSTM(1152, 300, 4, 200, 2, device)          # Initialize Sentiment Analysis Network
    model.load_state_dict(torch.load("saved_models/model_acc_45.61.at")["model_state_dict"])     # Load per-trained model

    while True:
        recorder = Recorder(RATE, CHUNK, WAV_FILE_PATH, AVI_FILE_PATH)
        #print("Press and hold the 'r' key to begin recording. Release the 'r' key to end recording. Press 'Escape' to exit")
        """with Recorder(RATE, CHUNK, WAV_FILE_PATH, AVI_FILE_PATH) as recorder:
            recorder.join()"""

        recorder.start()
        time.sleep(5)
        recorder.stop()
        ffmpeg_extract_subclip(AVI_FILE_PATH, t1=0, t2=4, targetname="online_test.avi")
        cmd = "openFace/OpenFace/build/bin/FeatureExtraction -f online_test.avi -out_dir online_test"
        os.system(cmd)
        time.sleep(100)

        upload_blob(BUCKET_NAME, WAV_FILE_PATH, WAV_FILE_PATH.split("/")[1])
        audio = speech.RecognitionAudio(uri="gs://ludus_sentiment-analysis/test.wav")

        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=44100,
            language_code="en-US",
        )

        response = client.recognize(config=config, audio=audio)

        sentence = ""
        for res in response.results:
            print("Transcript: {}".format(res.alternatives[0].transcript))
            sentence += res.alternatives[0].transcript

        sentence_embedding = np.asarray(list(bert.encode(sentence, batch_size=1)))
        cmd = f"SMILExtract -C opensmile/config/is09-13/IS09_emotion.conf -I {WAV_FILE_PATH} -O extracted_data/test.csv -l 0"
        os.system(cmd)
        reader = csv.reader(open("extracted_data/test.csv", 'r'))
        rows = [row for row in reader]
        last_line = rows[-1]
        audio_features = np.asarray(list(map(lambda x: float(x), last_line[1: 385])))
        input_data = np.concatenate((audio_features, sentence_embedding))
        input_data = torch.Tensor(input_data.reshape(1, 1, 1152)).to(device)
        output = model(input_data)
        output_label = torch.argmax(output)
        print(get_sentiment(output_label.item()))
        time.sleep(2)

def on_press(key):
    print('{0} pressed'.format(
        key))

def on_release(key):
    print('{0} release'.format(
        key))
    if key == Key.esc:
        # Stop listener
        return False

if __name__ == '__main__':
    print("Starting")
    main()
    print("Stopping program...")
