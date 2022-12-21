import argparse
import time

import pandas as pd
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from utils import *
from model import LSTM, LSTM1, LSTM_ATTN, LSTMSep
import torch.utils.data
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='IEMOCAP Sentiment Analysis')
    parser.add_argument('--use_own', type=bool, default=False,
                        help='Use own extracted features (default: False')
    parser.add_argument('--alpha', type=float, default=0.001,
                        help='initial learning rate (default: 0.001)')
    parser.add_argument('--n_classes', type=int, default=4,
                        help='number of classes in the network (default: 4)')
    parser.add_argument('--n_epochs', type=int, default=35,
                        help='number of epochs (default: 100)')
    parser.add_argument('--hidden_size', type=int, default=128,
                        help='dimension of hidden layer in LSTM (default: 128)')
    parser.add_argument('--n_layers', type=int, default=1,
                        help='number of hidden layers (default: 1)')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='batch size (default: 10)')
    parser.add_argument('--save_model_threshold_text', type=float, default=61,
                        help='threshold for saving text model (default: 61)')
    parser.add_argument('--save_model_threshold_audio', type=float, default=36,
                        help='threshold for saving audio model (default: 36)')
    parser.add_argument('--use_pretrained', type=bool, default=True,
                        help='Use pretrained model (default: False)')
    parser.add_argument('--use_pretrained_text', type=bool, default=False,
                        help='Use pretrained text model (default: False)')
    parser.add_argument('--use_pretrained_audio', type=bool, default=False,
                        help='Use pretrained audio model (default: False)')
    parser.add_argument('--fc_dim', type=int, default=200,
                        help='dimension of fc layer in LSTM (default: 200)')
    parser.add_argument('--lr_decay', type=float, default=0.999,
                        help='Learning rate decay rate (default: 0.99)')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #process_data()

    #print("done")


    # Load Dataset
    if args.use_own:
        print("Uses own features")
        data_train, data_test, train_text, train_audio, test_text, test_audio, train_label, test_label, train_seq_len, test_seq_len, train_mask, test_mask = get_extracted_data()
    else:
        print("Uses pre-extracted features")
        text_features_train, text_labels_train, text_mask_train, text_features_test, text_labels_test, text_mask_test, audio_features_train, audio_labels_train, audio_mask_train, audio_features_test, audio_labels_test, audio_mask_test, _, _ = process_features()

    # Initialize Tensors for test set
    text_target_test = torch.Tensor(text_labels_test).to(device)
    text_target_test = text_target_test.view(-1).long()

    audio_target_test = torch.Tensor(audio_labels_test).to(device)
    audio_target_test = audio_target_test.view(-1).long()

    # Define model
    """model = LSTM(input_feature_size=data_train.shape[-1], hidden_size=args.hidden_size, n_classes=args.n_classes,
                 n_layers=args.n_layers, device=device, fc_dim=args.fc_dim)"""
    """model = LSTM_ATTN(input_feature_size=data_train.shape[-1], hidden_size=args.hidden_size, n_classes=args.n_classes,
                  n_layers=args.n_layers, device=device)"""
    """model = LSTMSep(input_feature_size_text=train_text.shape[-1],input_feature_size_audio=train_audio.shape[-1], hidden_size=args.hidden_size, n_classes=args.n_classes,
                    n_layers=args.n_layers, device=device, fc_dim=args.fc_dim)"""
    model_text = LSTM1(input_feature_size=text_features_train.shape[-1], hidden_size=args.hidden_size, n_classes=args.n_classes,
                       n_layers=args.n_layers, device=device)
    model_audio = LSTM1(input_feature_size=audio_features_train.shape[-1], hidden_size=args.hidden_size,
                        n_classes=args.n_classes,
                        n_layers=args.n_layers, device=device)

    # Setting loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_text.parameters(), lr=args.alpha)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.lr_decay)

    best_epoch_text = 0
    best_acc_text = 0
    train_acc = 0

    if not args.use_pretrained:
        if not args.use_pretrained_text:
            print("Now training text classifier...")
            for epoch in range(args.n_epochs):
                batches = gen_bashes(text_features_train, text_labels_train, text_mask_train, args.batch_size)
                for idx, batch in enumerate(batches):
                    model_text.train()  # Indicate training started
                    b_train_text, b_train_label, b_train_mask = zip(*batch)
                    input_text = torch.Tensor(np.array(b_train_text)).to(device)
                    target_train = torch.Tensor(np.array(b_train_label)).to(device)
                    target_train = target_train.view(-1).long()
                    # Forward pass
                    output_text = model_text(input_text)
                    loss = criterion(output_text, target_train)

                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    # Calculate the unweighted accuracy of train pre_extracted_features
                    b_train_mask = np.asarray(b_train_mask).reshape(-1)
                    train_acc, f1_train, _, _ = report_acc(output_text, target_train, b_train_mask)

                    if (idx + 1) % args.batch_size == 0:
                        print(f'Epoch [{epoch + 1}/{args.n_epochs}], Loss: {loss.item():.4f}', end=" ")

                # Evaluation after each epoch on test set
                model_text.eval()
                with torch.no_grad():
                    output_test_text = model_text(torch.Tensor(text_features_test).to(device))
                    acc_test_text, f1_test_text, conf_matrix_text, classify_report_text = report_acc(output_test_text, text_target_test, text_mask_test)
                    print(f"Accuracy (test): {acc_test_text} (text_lstm)")

                if acc_test_text > best_acc_text:
                    best_acc_text = acc_test_text
                    best_epoch_text = epoch
                    if best_acc_text > args.save_model_threshold_text:
                        torch.save({
                                    'best_epoch': best_epoch_text,
                                    'model_state_dict': model_text.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict()
                                    },
                                   "saved_models/text_lstm/" + "model_acc_" + "{:0.2f}".format(best_acc_text) + ".t")

            print("Text LSTM:")
            print('Best Epoch: {}/{}.............'.format(best_epoch_text, args.n_epochs), end=" ")
            print("Train accuracy: {:.2f}% Test accuracy: {:.2f}%".format(train_acc, best_acc_text))
            print(classify_report_text)

        if not args.use_pretrained_audio:
            # Setting loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model_audio.parameters(), lr=args.alpha)
            scheduler = StepLR(optimizer, step_size=1, gamma=args.lr_decay)

            best_epoch_audio = 0
            best_acc_audio = 0
            train_acc = 0

            print("\nNow training audio classifier...")
            for epoch in range(args.n_epochs):
                batches = gen_bashes(audio_features_train, audio_labels_train, audio_mask_train, args.batch_size)
                for idx, batch in enumerate(batches):
                    model_audio.train()  # Indicate training started
                    b_train_audio, b_train_label, b_train_mask = zip(*batch)
                    input_audio = torch.Tensor(np.array(b_train_audio)).to(device)
                    target_train = torch.Tensor(np.array(b_train_label)).to(device)
                    target_train = target_train.view(-1).long()
                    # Forward pass
                    output_audio = model_audio(input_audio)
                    loss = criterion(output_audio, target_train)

                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    # Calculate the unweighted accuracy of train pre_extracted_features
                    b_train_mask = np.asarray(b_train_mask).reshape(-1)
                    train_acc, f1_train, _, _ = report_acc(output_audio, target_train, b_train_mask)

                    if (idx + 1) % args.batch_size == 0:
                        print(f'Epoch [{epoch + 1}/{args.n_epochs}], Loss: {loss.item():.4f}', end=" ")

                # Evaluation after each epoch on test set
                model_audio.eval()
                with torch.no_grad():
                    output_test_audio = model_audio(torch.Tensor(audio_features_test).to(device))
                    acc_test_audio, f1_test_audio, conf_matrix_audio, classify_report_audio = report_acc(output_test_audio,
                                                                                                     audio_target_test,
                                                                                                     audio_mask_test)
                    print(f"Accuracy (test): {acc_test_audio} (audio_lstm)")

                if acc_test_audio > best_acc_audio:
                    best_acc_audio = acc_test_audio
                    best_epoch_audio = epoch
                    if best_acc_audio > args.save_model_threshold_audio:
                        torch.save({
                            'best_epoch': best_epoch_audio,
                            'model_state_dict': model_audio.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()
                        },
                            "saved_models/audio_lstm/" + "model_acc_" + "{:0.2f}".format(best_acc_audio) + ".a")
            print("Audio LSTM:")
            print('Best Epoch: {}/{}.............'.format(best_epoch_audio, args.n_epochs), end=" ")
            print("Train accuracy: {:.2f}% Test accuracy: {:.2f}%".format(train_acc, best_acc_audio))
            print(classify_report_audio)
    else:
        text_model_info = torch.load("saved_models/text_lstm/model_acc_94.91.t")
        model_text.load_state_dict(text_model_info['model_state_dict'])
        model_text.eval()
        with torch.no_grad():
            output_test = model_text(torch.Tensor(text_features_test).to(device))
            acc_test, f1_test, conf_matrix, classify_report = report_acc(output_test, text_target_test, text_mask_test)

            print("Accuracy of text model loaded: {:.2f}%".format(acc_test))
            print(classify_report)

        audio_model_info = torch.load("saved_models/audio_lstm/model_acc_91.87.a")
        model_audio.load_state_dict(audio_model_info['model_state_dict'])
        model_audio.eval()
        with torch.no_grad():
            output_test = model_audio(torch.Tensor(np.array(audio_features_test)).to(device))
            acc_test, f1_test, conf_matrix, classify_report = report_acc(output_test, audio_target_test, audio_mask_test)

            print("Accuracy of audio model loaded: {:.2f}%".format(acc_test))
            print(classify_report)


if __name__ == '__main__':
    print("Starting")
    main()
