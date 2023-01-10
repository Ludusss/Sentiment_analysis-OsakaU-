import argparse
import time

import pandas as pd
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from utils import *
from model import LSTM, LSTM1, LSTM_ATTN, LSTMSep, MLP, MLP_2
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler


def main():
    parser = argparse.ArgumentParser(description='IEMOCAP Sentiment Analysis')
    parser.add_argument('--use_quad_classes', type=bool, default=False,
                        help='Use 4 class classifier (default: True')
    parser.add_argument('--alpha', type=float, default=0.01,
                        help='initial learning rate (default: 0.01)')
    parser.add_argument('--lr_decay', type=float, default=0.999,
                        help='Learning rate decay rate (default: 0.99)')
    parser.add_argument('--n_epochs_text', type=int, default=20,
                        help='number of epochs text model (default: 100)')
    parser.add_argument('--hidden_size_text', type=int, default=200,
                        help='dimension of hidden layer in text model (default: 128)')
    parser.add_argument('--n_layers_text', type=int, default=2,
                        help='number of hidden layers text model (default: 1)')
    parser.add_argument('--batch_size_text', type=int, default=32,
                        help='batch size text model (default: 32)')
    parser.add_argument('--fc_dim_text', type=int, default=200,
                        help='dimension of fc layer in text model (default: 200)')
    parser.add_argument('--n_epochs_audio', type=int, default=35,
                        help='number of epochs audio model (default: 100)')
    parser.add_argument('--hidden_size_audio', type=int, default=128,
                        help='dimension of hidden layer in audio model (default: 128)')
    parser.add_argument('--n_layers_audio', type=int, default=1,
                        help='number of hidden layers audio model (default: 1)')
    parser.add_argument('--batch_size_audio', type=int, default=32,
                        help='batch size audio model (default: 32)')
    parser.add_argument('--save_model_threshold_text', type=float, default=63,
                        help='threshold for saving text model (default: 63)')
    parser.add_argument('--save_model_threshold_audio', type=float, default=69,
                        help='threshold for saving audio model (default: 69)')
    parser.add_argument('--use_pretrained', type=bool, default=False,
                        help='Use pretrained model (default: False)')
    parser.add_argument('--use_pretrained_text', type=bool, default=True,
                        help='Use pretrained text model (default: False)')
    parser.add_argument('--use_pretrained_audio', type=bool, default=False,
                        help='Use pretrained audio model (default: False)')
    parser.add_argument('--fc_dim_audio', type=int, default=200,
                        help='dimension of fc layer in audio model (default: 200)')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    text_train, text_labels_train, text_test, text_labels_test, text_val, text_labels_val = process_twitter()   # Text data
    audio_train, audio_labels_train, audio_test, audio_labels_test, audio_val, audio_labels_val = process_ESD_features(args.use_quad_classes)   # Audio data

    # Initialize Tensors for train, val and test sets
    text_target_train = torch.Tensor(text_labels_train).to(device)
    text_target_val = torch.Tensor(text_labels_val).to(device)
    text_target_test = torch.Tensor(text_labels_test).to(device)

    audio_target_train = torch.Tensor(audio_labels_train).to(device)
    audio_target_val = torch.Tensor(audio_labels_val).to(device)
    audio_target_test = torch.Tensor(audio_labels_test).to(device)

    # Define models
    if args.use_quad_classes:
        print("4-class models used")
        model_text = MLP(input_feature_size=text_train.shape[-1], hidden_size=args.hidden_size_text, n_classes=13,
                            n_layers=args.n_layers_text, device=device)

        model_audio = MLP(input_feature_size=audio_train.shape[-1], hidden_size=args.hidden_size_audio, n_classes=4,
                              n_layers=args.n_layers_audio, device=device)
    else:
        print("3-class models used")
        model_text = MLP(input_feature_size=text_train.shape[-1], hidden_size=args.hidden_size_text, n_classes=3,
                         n_layers=args.n_layers_text, device=device)

        model_audio = MLP(input_feature_size=audio_train.shape[-1], hidden_size=args.hidden_size_audio, n_classes=3, n_layers=args.n_layers_audio, device=device)

    if not args.use_pretrained:
        if not args.use_pretrained_text:
            # Setting loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model_text.parameters(), lr=args.alpha)
            scheduler = StepLR(optimizer, step_size=1, gamma=args.lr_decay)

            best_epoch_text = 0
            best_acc_text = 0

            # variables for plotting
            train_acc_epoch = []
            val_acc_epoch = []
            test_acc_epoch = []
            train_f1_epoch = []
            val_f1_epoch = []
            test_f1_epoch = []

            print("Now training text classifier...")
            for epoch in range(args.n_epochs_text):
                batches = gen_batches_mlp(text_train, text_labels_train, args.batch_size_text)
                for idx, batch in enumerate(batches):
                    model_text.train()  # Indicate training started
                    b_train_text, b_train_label = zip(*batch)
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

                # Evaluation after each epoch on val set
                model_text.eval()
                with torch.no_grad():
                    output_train_text = model_text(torch.Tensor(text_train).to(device))
                    output_val_text = torch.softmax(model_text(torch.Tensor(text_val).to(device)), dim=1)
                    output_test_text = torch.softmax(model_text(torch.Tensor(text_test).to(device)), dim=1)

                    acc_train_text, f1_train_text, _, _ = report_acc_mlp(
                        torch.softmax(output_train_text, dim=1), text_target_train)
                    acc_val_text, f1_val_text, _, _ = report_acc_mlp(
                        output_val_text, text_target_val)
                    acc_test_text, f1_test_text, _, _ = report_acc_mlp(
                        output_test_text, text_target_test)

                    val_acc_epoch.append(acc_val_text)
                    val_f1_epoch.append(f1_val_text)
                    test_acc_epoch.append(acc_test_text)
                    test_f1_epoch.append(f1_test_text)
                    train_f1_epoch.append(f1_train_text)
                    train_acc_epoch.append(acc_train_text)

                    training_loss = criterion(output_train_text, text_target_train.view(-1).long())
                    print(f'Epoch [{epoch + 1}/{args.n_epochs_text}], Loss: {training_loss.item():.4f}', end=" ")
                    print(f"Accuracy (evaluation): {acc_val_text} (text_model)")

                if acc_val_text > best_acc_text:
                    best_acc_text = acc_val_text
                    best_epoch_text = epoch
                    train_acc = acc_train_text
                    if best_acc_text > args.save_model_threshold_text:
                        if args.use_quad_classes:
                            torch.save({
                                'best_epoch': best_epoch_text,
                                'model_state_dict': model_text.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict()
                            },
                                "saved_models/text_mlp/twitter" + "4_twitter_model_mlp_acc_" + "{:0.2f}".format(
                                    best_acc_text) + ".t")
                        else:
                            torch.save({
                                'best_epoch': best_epoch_text,
                                'model_state_dict': model_text.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict()
                            },
                                "saved_models/text_mlp/twitter" + "3_twitter_model_mlp_acc_" + "{:0.2f}".format(
                                    best_acc_text) + ".t")

            output_test_text = model_text(torch.Tensor(text_test).to(device))
            acc_test_text, f1_test_text, conf_matrix_text_test, classify_report_text_test = report_acc_mlp(
                output_test_text, text_target_test)
            print("Text model:")
            print('Best Epoch: {}/{}.............'.format(best_epoch_text, args.n_epochs_text), end=" ")
            print("Train accuracy: {:.2f}% Evaluation accuracy: {:.2f}% Test accuracy: {:.2f}%".format(train_acc,
                                                                                                       best_acc_text,
                                                                                                       acc_test_text))
            print(classify_report_text_test)

            #   Plotting
            figure, ax = plt.subplots(2, 1)
            ax[0].set_title("Text accuracy on Training, Test and Validation sets")
            ax[0].set_xlabel("Epoch")
            ax[0].set_ylabel("Accuracy")
            ax[0].plot(np.arange(len(test_acc_epoch)), test_acc_epoch, color="green", label="test")
            ax[0].plot(np.arange(len(val_acc_epoch)), val_acc_epoch, color="orange", label="validation")
            ax[0].plot(np.arange(len(train_acc_epoch)), train_acc_epoch, color="blue", label="training")
            ax[0].legend()
            ax[0].set_xticks(np.arange(len(test_acc_epoch)))

            ax[1].set_title("Text F1-score on Test and Validation sets")
            ax[1].set_xlabel("Epoch")
            ax[1].set_ylabel("Accuracy")
            ax[1].plot(np.arange(len(test_f1_epoch)), test_f1_epoch, color="green", label="test")
            ax[1].plot(np.arange(len(val_f1_epoch)), val_f1_epoch, color="orange", label="validation")
            ax[1].plot(np.arange(len(train_f1_epoch)), train_f1_epoch, color="blue", label="training")
            ax[1].legend()
            ax[1].set_xticks(np.arange(len(test_f1_epoch)))

            plt.show()

        if not args.use_pretrained_audio:
            # Setting loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model_audio.parameters(), lr=args.alpha)
            scheduler = StepLR(optimizer, step_size=1, gamma=args.lr_decay)

            best_epoch_audio = 0
            best_acc_audio = 0

            # variables for plotting
            train_acc_epoch = []
            val_acc_epoch = []
            test_acc_epoch = []
            train_f1_epoch = []
            val_f1_epoch = []
            test_f1_epoch = []

            print("\nNow training audio classifier...")
            for epoch in range(args.n_epochs_audio):
                batches = gen_batches_mlp(audio_train, audio_labels_train, args.batch_size_audio)
                for idx, batch in enumerate(batches):
                    model_audio.train()  # Indicate training started
                    b_train_audio, b_train_label = zip(*batch)
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

                # Evaluation after each epoch on test set
                model_audio.eval()
                with torch.no_grad():
                    output_train_audio = model_audio(torch.Tensor(audio_train).to(device))
                    output_val_audio = torch.softmax(model_audio(torch.Tensor(audio_val).to(device)), dim=1)
                    output_test_audio = torch.softmax(model_audio(torch.Tensor(audio_test).to(device)), dim=1)

                    acc_train_audio, f1_train_audio, _, _ = report_acc_mlp(
                        torch.softmax(output_train_audio, dim=1), audio_target_train)
                    acc_val_audio, f1_val_audio, _, _ = report_acc_mlp(
                        output_val_audio, audio_target_val)
                    acc_test_audio, f1_test_audio, _, _ = report_acc_mlp(
                        output_test_audio, audio_target_test)

                    val_acc_epoch.append(acc_val_audio)
                    val_f1_epoch.append(f1_val_audio)
                    test_acc_epoch.append(acc_test_audio)
                    test_f1_epoch.append(f1_test_audio)
                    train_f1_epoch.append(f1_train_audio)
                    train_acc_epoch.append(acc_train_audio)

                    training_loss = criterion(output_train_audio, audio_target_train.view(-1).long())
                    print(f'Epoch [{epoch + 1}/{args.n_epochs_audio}], Loss: {training_loss.item():.4f}', end=" ")
                    print(f"Accuracy (evaluation): {acc_val_audio} (audio_model)")

                if acc_val_audio > best_acc_audio:
                    best_acc_audio = acc_val_audio
                    best_epoch_audio = epoch
                    train_acc = acc_train_audio
                    if best_acc_audio > args.save_model_threshold_audio:
                        if args.use_quad_classes:
                            torch.save({
                                'best_epoch': best_epoch_audio,
                                'model_state_dict': model_audio.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict()
                            },
                                "saved_models/audio_mlp/ESD/" + "4_model_ESD_acc_" + "{:0.2f}".format(best_acc_audio) + ".a")
                        else:
                            torch.save({
                                'best_epoch': best_epoch_audio,
                                'model_state_dict': model_audio.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict()
                            },
                                "saved_models/audio_mlp/ESD/" + "3_model_ESD_acc_" + "{:0.2f}".format(best_acc_audio) + ".a")

            output_test_audio = model_audio(torch.Tensor(audio_test).to(device))
            acc_test_audio, f1_test_audio, conf_matrix_audio_test, classify_report_audio_test = report_acc_mlp(
                output_test_audio, audio_target_test)
            print("Audio model:")
            print('Best Epoch: {}/{}.............'.format(best_epoch_audio, args.n_epochs_audio), end=" ")
            print("Train accuracy: {:.2f}% Evaluation accuracy: {:.2f}% Test accuracy: {:.2f}%".format(train_acc,
                                                                                                       best_acc_audio,
                                                                                                       acc_test_audio))
            print(classify_report_audio_test)

            #   Plotting
            figure, ax = plt.subplots(2, 1)
            ax[0].set_title("Text accuracy on Training, Test and Validation sets")
            ax[0].set_xlabel("Epoch")
            ax[0].set_ylabel("Accuracy")
            ax[0].plot(np.arange(len(test_acc_epoch)), test_acc_epoch, color="green", label="test")
            ax[0].plot(np.arange(len(val_acc_epoch)), val_acc_epoch, color="orange", label="validation")
            ax[0].plot(np.arange(len(train_acc_epoch)), train_acc_epoch, color="blue", label="training")
            ax[0].legend()
            ax[0].set_xticks(np.arange(len(test_acc_epoch)))

            ax[1].set_title("Text F1-score on Test and Validation sets")
            ax[1].set_xlabel("Epoch")
            ax[1].set_ylabel("Accuracy")
            ax[1].plot(np.arange(len(test_f1_epoch)), test_f1_epoch, color="green", label="test")
            ax[1].plot(np.arange(len(val_f1_epoch)), val_f1_epoch, color="orange", label="validation")
            ax[1].plot(np.arange(len(train_f1_epoch)), train_f1_epoch, color="blue", label="training")
            ax[1].legend()
            ax[1].set_xticks(np.arange(len(test_f1_epoch)))

            plt.show()
    else:
        if args.use_quad_classes:
            text_model_info = torch.load("saved_models/text_lstm/4_model_acc_86.72.t")
        else:
            text_model_info = torch.load("saved_models/text_lstm/3_model_acc_84.06.t")
        model_text.load_state_dict(text_model_info['model_state_dict'])
        model_text.eval()
        with torch.no_grad():
            output_test = model_text(torch.Tensor(text_features_test).to(device))
            acc_test, f1_test, conf_matrix, classify_report = report_acc(output_test, text_target_test, text_mask_test)

            print("Accuracy of text model loaded: {:.2f}%".format(acc_test))
            print(classify_report)
        if not args.use_esd:
            if args.use_quad_classes:
                audio_model_info = torch.load("saved_models/audio_lstm/4_model_acc_89.72.a")
            else:
                audio_model_info = torch.load("saved_models/audio_lstm/3_model_acc_89.72.a")
            model_audio.load_state_dict(audio_model_info['model_state_dict'])
            model_audio.eval()
            with torch.no_grad():
                output_test = model_audio(torch.Tensor(np.array(audio_features_test)).to(device))
                acc_test, f1_test, conf_matrix, classify_report = report_acc(output_test, audio_target_test, audio_mask_test)

                print("Accuracy of audio model loaded: {:.2f}%".format(acc_test))
                print(classify_report)
        else:
            if args.use_quad_classes:
                audio_model_info = torch.load("saved_models/audio_lstm/ESD/4_model_ESD_acc_74.97.a")
            else:
                audio_model_info = torch.load("saved_models/audio_lstm/ESD/3_model_ESD_acc_89.72.a")
            model_audio.load_state_dict(audio_model_info['model_state_dict'])
            model_audio.eval()
            with torch.no_grad():
                output_test = model_audio(torch.Tensor(np.array(audio_test)).to(device))
                acc_test, f1_test, conf_matrix, classify_report = report_acc_mlp(output_test, audio_target_test)

                print("Accuracy of audio model loaded: {:.2f}%".format(acc_test))
                print(classify_report)


if __name__ == '__main__':
    print("Starting")
    main()
