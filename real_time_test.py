import argparse
import os

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import random
from luo.model import hir_fullModel
from model import LSTM


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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    bert = SentenceTransformer('all-mpnet-base-v2')    # Load s-bert model for text feature extractions
    model = LSTM(712, 128, 4, 1, device)
    model.load_state_dict(torch.load("saved_models/model_acc_67.47")["model_state_dict"])
    """os.chdir("/Users/ludus/Projects/Sentiment_analysis-OsakaU-/luo")
    model = hir_fullModel(batch_size=args.batch_size, mode=args.bl, classifier=args.classifier,
                          output_size=args.output_size, hidden_dim=args.hidden_dim,
                          fc_dim=args.fc_dim, dropout=args.dropout, n_layers=args.n_layers)
    os.chdir("/Users/ludus/Projects/Sentiment_analysis-OsakaU-")
    model.load_state_dict(torch.load(args.full_dict, map_location='cpu'))"""

    while True:
        sentence = input("Write text to be analyzed: ")
        sentence_embedding = np.asarray(random.sample(list(bert.encode(sentence, batch_size=1)), 712))
        text_input = torch.Tensor(sentence_embedding.reshape(1, 1, 712)).to(device)
        output = model(text_input)
        output_label= torch.argmax(output)
        print(get_sentiment(output_label.item()))
        if sentence == "stop":
            return


if __name__ == '__main__':
    print("Starting")
    main()
    print("Stopping program...")
