import argparse
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from AudioRecorder import AudioRecorder
from model import LSTM
from rev_ai import apiclient, JobStatus
import os
import csv

ACCESS_TOKEN = "028Y1kOBwGp6Jt1W2n7oR8Qzd3KWRqrY8G7rJZTKNOx-Ac_j16-BJ-v8viQh_8ELgOJMc85D1zdvJkTwhSoNRZomVq9Fw"
RATE = 44100
CHUNK = int(RATE / 10)
WAV_FILE_PATH = "recordings/test.wav"


def get_sentiment(label):
    labels = {0: "Happy", 1: "Sad", 2: "Neutural", 3: "Angry", 4: "Excited", 5: "Frustrated"}
    return labels[label]


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
    client = apiclient.RevAiAPIClient(ACCESS_TOKEN)    # Connect to rev ai ASR
    bert = SentenceTransformer('all-mpnet-base-v2')    # Load s-bert model for text feature extractions
    model = LSTM(1152, 300, 4, 200, 2, device)          # Initialize Sentiment Analysis Network
    model.load_state_dict(torch.load("saved_models/model_acc_45.61.at")["model_state_dict"])     # Load per-trained model

    while True:
        print("Press and hold the 'r' key to begin recording. Release the 'r' key to end recording. Press 'Escape' to exit")
        with AudioRecorder(RATE, CHUNK, WAV_FILE_PATH) as audio_recorder:
            audio_recorder.join()

        job = client.submit_job_local_file(WAV_FILE_PATH)

        # check job status
        while True:
            job_details = client.get_job_details(job.id)
            if job_details.status == JobStatus.TRANSCRIBED:
                break

        text = client.get_transcript_text(job.id)
        sentence = text.split("    ")[2].strip()
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


if __name__ == '__main__':
    print("Starting")
    main()
    print("Stopping program...")
