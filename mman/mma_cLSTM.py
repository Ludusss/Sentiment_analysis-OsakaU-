import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR

import numpy as np 
import argparse

from utils import *
from model import *

import seaborn as sns
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description='MOSEI Sentiment Analysis')

    # LSTM Param
    parser.add_argument('--output_size', type=int, default=4,
                        help='number of classes in the network (default: 4)')
    parser.add_argument('--hidden_dim', type=int, default=300,
                        help='dimension of hidden vector in LSTM (default: 300)')
    parser.add_argument('--fc_dim', type=int, default=200,
                        help='dimension of fc layer in LSTM (default: 200)')
    parser.add_argument('--n_layers', type=int, default=1,
                        help='no. layrss in LSTM (default: 2)')
    parser.add_argument('--dropout_lstm', type=int, default=0.5,
                        help='dropout rate LSTM (default: 0.5)')


    # Transformer param
    parser.add_argument('--nhidden', type=int, default=300,
                        help='dimension of hidden vector in transformer (default: 300)')
    parser.add_argument('--nLayers', type=int, default=1,
                        help='no. layrss in transformer (default: 1)')
    parser.add_argument('--nhead', type=int, default=1,
                        help='no. heads in transformer (default: 1)')
    parser.add_argument('--input_feat_size', type=int, default=150,
                        help='input standardised feat_size in transformer (default: 150)')
    parser.add_argument('--dropout', type=int, default=0.1,
                        help='dropout rate transformer (default: 0.5)')


    # Tuning
    parser.add_argument('--batch_size', type=int, default=5,
                        help='batch size (default: 10)')
    parser.add_argument('--clip', type=float, default=1e-3,
                        help='gradient clip value (default: 1e-3)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='initial learning rate (default: 0.001)')
    parser.add_argument('--lr_decay', type=float, default=1,
                        help='Learning rate decay rate (default: 0.99)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs (default: 100)')

    # Testing
    parser.add_argument('--load_dict', type=bool, default=False,
                        help='Load a pretrained state_dict (default: False)')
    parser.add_argument('--test_mode', type=bool, default=True,
                        help='test_mode, plot confusion matrix (default: False)')
    parser.add_argument('--ctf_dict', type=str, default="test/trs_lstm70.86.pt",
                        help='ctf pretrained stat dict (default: 70.67)')


    args = parser.parse_args()

    # Use GPU if it is available
    if torch.cuda.is_available():
        print(torch.cuda.current_device())
        device = torch.device("cuda")
        # print("GPU is used")
    else:
        device = torch.device("cpu")
        print("cpu is used")

    # Load the dataset
    _, _, audio_train, audio_test, text_train, text_test, video_train, video_test, train_label, test_label, seqlen_train, seqlen_test, train_mask, test_mask = get_iemocap_raw(4)

    # Define the model
    model = MAN(batch_size = args.batch_size, input_feat_size = args.input_feat_size , audio_dim = 100, visual_dim = 512, text_dim = 100, nhead = args.nhead, nhidden = args.nhidden,  nLayers = args.nLayers, dropout = args.dropout,  output_size = args.output_size, hidden_dim= args.hidden_dim , fc_dim = args.fc_dim, n_layers = args.n_layers, dropout_lstm = args.dropout_lstm)
    if args.load_dict or args.test_mode:
        model.load_state_dict(torch.load(args.ctf_dict))
    model = model.to(device)

    # print(sum(p.numel() for p in model.parameters()))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr= args.lr)
    scheduler = StepLR(optimizer, step_size = 1, gamma = args.lr_decay)

    input_audio_test = torch.Tensor(audio_test).to(device)
    input_visual_test = torch.Tensor(video_test).to(device)
    input_text_test = torch.Tensor(text_test).to(device)
    target_test = torch.Tensor(test_label).to(device)
    target_test = target_test.view(-1).long()

    best_test_epoch = 0
    best_test_acc = 0
    train_acc = 0

    if args.test_mode:
        model.eval()
        with torch.no_grad():
            output_tst, _ = model(input_audio_test,input_text_test,input_visual_test)
            output_tst = output_tst.to(device)
            loss_tst = criterion(output_tst, target_test)
            accuracy_tst = cal_acc(output_tst, target_test, test_mask)
            precision, recall,f1,  wa , wf1= cal_metrics(output_tst, target_test, test_mask)
            cm = report_acc(output_tst, target_test, test_mask)
            cm = cm.astype('float')
            tt=cm.sum(axis=1)[:, np.newaxis]
            cm = cm/tt *100
            ax= plt.subplot()
            sns.set(font_scale=1.5)
            sns.heatmap(cm, annot=True, ax = ax,cmap="Blues", cbar = False, square = True,fmt=".2f"); #annot=True to annotate cells

            # labels, title and ticks
            ax.set_xlabel('Predicted labels',fontsize = 25.0);
            ax.set_ylabel('True labels', fontsize = 25.0);
            ax.set_title('cLSTM-MMA\n',fontsize = 25.0);
            ax.xaxis.set_ticklabels(['HPY', 'SAD', 'NEU', 'ANG'],fontsize = 20.0);
            ax.yaxis.set_ticklabels(['HPY', 'SAD', 'NEU', 'ANG'],fontsize = 20.0, rotation = 0);
            # ax.get_yaxis().set_visible(False)
            plt.savefig("test/cm_mafn.pdf", bbox_inches = 'tight', pad_inches = 0)

            precision, recall,f1,  wa , wf1= cal_metrics(output_tst, target_test, test_mask)

            print(precision)
            print(wa)
            print(wf1)

    else: 
        for epoch in range(1, args.epochs + 1):
            batches = batch_iter(list(zip(audio_train, text_train, video_train, train_mask, seqlen_train, train_label)), args.batch_size)
            for idx, batch in enumerate(batches):
                model.train()
                b_audio_train, b_text_train, b_visual_train, b_train_mask, b_seqlen_train, b_train_label = zip(*batch)

                input_audio_train = torch.Tensor(b_audio_train).to(device)
                input_text_train = torch.Tensor(b_text_train).to(device)
                input_visual_train = torch.Tensor(b_visual_train).to(device)
                target_train = torch.Tensor(b_train_label).to(device)
                target_train = target_train.view(-1).long()

                optimizer.zero_grad()
                output, _ = model(input_audio_train, input_text_train, input_visual_train)
                output = output.to(device)

                loss = criterion(output, target_train)
                loss.backward() 
                clip_gradient(model, args.clip)
                optimizer.step() 
                scheduler.step()

                # Calculate the unweighted accuracy of train data
                b_train_mask = np.asarray(b_train_mask).reshape(-1)
                train_acc = cal_acc(output, target_train, b_train_mask)


            # if epoch%1 == 0:
                # Evaluate on the test data:
                model.eval()
                with torch.no_grad():
                    output_tst, _ = model(input_audio_test,input_text_test,input_visual_test)
                    output_tst = output_tst.to(device)
                    loss_tst = criterion(output_tst, target_test)
                    accuracy_tst = cal_acc(output_tst, target_test, test_mask)

                if accuracy_tst > best_test_acc:
                    best_test_acc = accuracy_tst
                    best_test_epoch = epoch
                    if accuracy_tst > 0.7:
                        torch.save(model.state_dict(), "state_dict/mafn/trs_lstm{:0.2f}.pt".format(100*accuracy_tst))
                # precision, recall,f1,  wa , wf1= cal_metrics(output_tst, target_test, test_mask)
                # print(wf1*100)
                print('Epoch: {}/{}.............'.format(epoch, args.epochs), end=' ')
                print("Train:{:.2f}%, Loss:{:.3f}, Test:{:.2f}% , Best Test:{:.2f}".format(100. * train_acc, loss, 100*accuracy_tst, 100*best_test_acc))

        print('Best Epoch: {}/{}.............'.format(best_test_epoch, args.epochs), end=' ')
        print("Accuracy:{:.2f} {:.2f}%".format(100*train_acc, 100. * best_test_acc))

if __name__ == '__main__':
    main()
