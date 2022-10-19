import argparse
import torch
from torch import nn
from utils import *
from model import LSTM
import torch.utils.data
import numpy as np
from sklearn import metrics

def main():
    parser = argparse.ArgumentParser(description='IEMOCAP Sentiment Analysis')

    parser.add_argument('--alpha', type=float, default=0.001,
                        help='initial learning rate (default: 0.001)')
    parser.add_argument('--n_classes', type=int, default=4,
                        help='number of classes in the network (default: 4)')
    parser.add_argument('--n_epochs', type=int, default=100,
                        help='number of epochs (default: 100)')
    parser.add_argument('--hidden_size', type=int, default=128,
                        help='dimension of hidden layer in LSTM (default: 128)')
    parser.add_argument('--n_layers', type=int, default=1,
                        help='number of hidden layers (default: 1)')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='batch size (default: 10)')
    parser.add_argument('--save_model_threshold', type=float, default=67.34,
                        help='threshold for saving model (default: 67.34)')
    parser.add_argument('--use_pretrained', type=bool, default=False,
                        help='Use pretrained model (default: False)')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    # Load Dataset
    data_train, data_test, audio_train, audio_test, text_train, text_test, video_train, video_test, train_label, test_label, seqlen_train, seqlen_test, train_mask, test_mask = get_iemocap_data()

    # Define model - using concatenated multimodal features (video, audio, transcript)
    model = LSTM(input_feature_size=data_train.shape[-1], hidden_size=args.hidden_size, n_classes=args.n_classes, n_layers=args.n_layers, device=device)

    # Setting loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.alpha)

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

                # Calculate the unweighted accuracy of train data
                b_train_mask = np.asarray(b_train_mask).reshape(-1)
                train_acc = get_accuracy(output, target_train, b_train_mask)

                if (idx + 1) % args.batch_size == 0:
                    print(f'Epoch [{epoch + 1}/{args.n_epochs}], Loss: {loss.item():.4f}, Accuracy (test): {get_accuracy(model(input_test), target_test, test_mask)}')

            # Evaluation after each epoch on test set
            model.eval()
            with torch.no_grad():
                output_test = model(input_test).to(device)
                acc_test = get_accuracy(output_test, target_test, test_mask)

            if acc_test > best_acc:
                best_acc = acc_test
                best_epoch = epoch
                if best_acc > args.save_model_threshold:
                    torch.save({
                                'best_epoch': best_epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict()
                                },
                               "saved_models/" + "model_acc_" + "{:0.2f}".format(best_acc))


        print('Best Epoch: {}/{}.............'.format(best_epoch, args.n_epochs), end=' ')
        print("Train accuracy:{:.2f}% Test accuracy: {:.2f}%".format(train_acc, best_acc))

    else:
        model_info = torch.load("saved_models/model_acc_67.47")
        model.load_state_dict(model_info['model_state_dict'])
        model.eval()
        with torch.no_grad():
            output_test = model(input_test).to(device)
            acc_test = get_accuracy(output_test, target_test, test_mask)

            print("Accuracy of model loaded: {:.2f}%".format(acc_test))










if __name__ == '__main__':
    print("Starting")
    main()
