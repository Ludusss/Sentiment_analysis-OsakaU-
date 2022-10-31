import argparse
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR

from utils import *
from model import LSTM, LSTM1
import torch.utils.data
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='IEMOCAP Sentiment Analysis')

    parser.add_argument('--alpha', type=float, default=0.001,
                        help='initial learning rate (default: 0.001)')
    parser.add_argument('--n_classes', type=int, default=4,
                        help='number of classes in the network (default: 4)')
    parser.add_argument('--n_epochs', type=int, default=35,
                        help='number of epochs (default: 100)')
    parser.add_argument('--hidden_size', type=int, default=300,
                        help='dimension of hidden layer in LSTM (default: 128)')
    parser.add_argument('--n_layers', type=int, default=2,
                        help='number of hidden layers (default: 1)')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='batch size (default: 10)')
    parser.add_argument('--save_model_threshold', type=float, default=60,
                        help='threshold for saving model (default: 69)')
    parser.add_argument('--use_pretrained', type=bool, default=True,
                        help='Use pretrained model (default: False)')
    parser.add_argument('--fc_dim', type=int, default=200,
                        help='dimension of fc layer in LSTM (default: 200)')
    parser.add_argument(
        "--modalities",
        type=str,
        default="t",
        choices=["a", "t", "v", "at", "tv", "av", "atv"],
        help="Modalities",
    )
    parser.add_argument('--lr_decay', type=float, default=0.999,
                        help='Learning rate decay rate (default: 0.99)')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    # Load Dataset
    data_train, data_test, train_text, test_text, train_label, test_label, train_seq_len, test_seq_len, train_mask, test_mask = get_extracted_data()
    #data_train1, data_test1, audio_train, audio_test, text_train1, text_test1, video_train, video_test, train_label1, test_label1, seqlen_train1, seqlen_test1, train_mask1, test_mask1 = get_iemocap_data()

    # Define model - using concatenated multimodal features (video, audio, transcript)
    model = LSTM(input_feature_size=data_train.shape[-1], hidden_size=args.hidden_size, n_classes=args.n_classes,
                 n_layers=args.n_layers, device=device, fc_dim=args.fc_dim)
    """model = LSTM1(input_feature_size=data_train.shape[-1], hidden_size=args.hidden_size, n_classes=args.n_classes,
                 n_layers=args.n_layers, device=device)"""

    # Setting loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.alpha)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.lr_decay)

    # Initialize Tensors for test set
    input_test = torch.Tensor(np.array(data_test)).to(device)
    target_test = torch.Tensor(np.array(test_label)).to(device)
    target_test = target_test.view(-1).long()

    best_epoch = 0
    best_acc = 0
    train_acc = 0

    if not args.use_pretrained:
        for epoch in range(args.n_epochs):
            batches = gen_bashes(data_train, train_label, train_mask, args.batch_size)
            for idx, batch in enumerate(batches):
                model.train()  # Indicate training started
                b_train_data, b_train_label, b_train_mask = zip(*batch)
                input_train = torch.Tensor(np.array(b_train_data)).to(device)
                target_train = torch.Tensor(np.array(b_train_label)).to(device)
                target_train = target_train.view(-1).long()

                # Forward pass
                output = model(input_train)
                loss = criterion(output, target_train)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                # Calculate the unweighted accuracy of train data
                b_train_mask = np.asarray(b_train_mask).reshape(-1)
                train_acc, f1_train, _, _ = report_acc(output, target_train, b_train_mask)

                if (idx + 1) % args.batch_size == 0:
                    print(f'Epoch [{epoch + 1}/{args.n_epochs}], Loss: {loss.item():.4f}', end=" ")

            # Evaluation after each epoch on test set
            model.eval()
            with torch.no_grad():
                output_test = model(input_test).to(device)
                acc_test, f1_test, conf_matrix, classify_report = report_acc(output_test, target_test, test_mask)
                print(f"Accuracy (test): {acc_test}")

            if acc_test > best_acc:
                best_acc = acc_test
                best_epoch = epoch
                if best_acc > args.save_model_threshold:
                    torch.save({
                                'best_epoch': best_epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict()
                                },
                               "saved_models/" + "model_acc_" + "{:0.2f}".format(best_acc) + "." + str(args.modalities))


        print('Best Epoch: {}/{}.............'.format(best_epoch, args.n_epochs), end=" ")
        print("Train accuracy: {:.2f}% Test accuracy: {:.2f}%".format(train_acc, best_acc))
        print(classify_report)


    else:
        model_info = torch.load("saved_models/model_acc_64.91.t")
        model.load_state_dict(model_info['model_state_dict'])
        model.eval()
        with torch.no_grad():
            output_test = model(input_test).to(device)
            acc_test, f1_test, conf_matrix, classify_report = report_acc(output_test, target_test, test_mask)

            print("Accuracy of model loaded: {:.2f}%".format(acc_test))
            print(classify_report)










if __name__ == '__main__':
    print("Starting")
    main()
