import torch
from torch import nn
from utils import *
from model import LSTM
import torch.utils.data


# Hyper-parameters
alpha = 0.001
n_classes = 4
n_epochs = 100
hidden_size = 128
n_layers = 1
batch_size = 10


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    # Load Dataset
    data_train, data_test, audio_train, audio_test, text_train, text_test, video_train, video_test, train_label, test_label, seqlen_train, seqlen_test, train_mask, test_mask = get_iemocap_data(4)

    # Define model - using concatenated multimodal features (video, audio, transcript)
    model = LSTM(input_feature_size=data_train.shape[-1], hidden_size=hidden_size, n_classes=n_classes, n_layers=n_layers, device=device)

    # Setting loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=alpha)

    # Initialize Tensors for test set
    input_test = torch.Tensor(np.array(data_test)).to(device)
    target_test = torch.Tensor(np.array(test_label)).to(device)
    target_test = target_test.view(-1).long()

    best_epoch = 0
    best_acc = 0
    train_acc = 0

    for epoch in range(n_epochs):
        batches = gen_bashes(data_train, train_label, train_mask, batch_size)
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

            if (idx + 1) % batch_size == 0:
                print(f'Epoch [{epoch + 1}/{n_epochs}], Loss: {loss.item():.4f}, Accuracy (test): {get_accuracy(model(input_test), target_test, test_mask)}')

        # Evaluation after each epoch on test set
        model.eval()
        with torch.no_grad():
            output_test = model(input_test).to(device)
            acc_test = get_accuracy(output_test, target_test, test_mask)

        if acc_test > best_acc:
            best_acc = acc_test
            best_epoch = epoch

    print('Best Epoch: {}/{}.............'.format(best_epoch, n_epochs), end=' ')
    print("Train accuracy:{:.2f}% Test accuracy: {:.2f}%".format(train_acc, best_acc))


if __name__ == '__main__':
    print("Starting")
    main()
