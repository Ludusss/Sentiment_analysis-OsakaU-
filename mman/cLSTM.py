import time

import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR

import numpy as np 
import argparse

from utils import *
from model import *

import seaborn as sns
import matplotlib.pyplot as plt

from utils import process_features


def main():
    parser = argparse.ArgumentParser(description='IEMOCAP Emotion Analysis')

    # Modal selection
    parser.add_argument('--model', type=str, default='text',
                        help='choose the modal to work with audio/visual/text/fusion (default: audio)')

    # Model Param
    parser.add_argument('--n_layers', type=int, default=2,
                        help='number of classes in the network (default: 2)')
    parser.add_argument('--output_size', type=int, default=4,
                        help='number of classes in the network (default: 4)')
    parser.add_argument('--hidden_dim', type=int, default=300,
                        help='dimension of hidden vector in LSTM (default: 300, 512 for fusion)')
    parser.add_argument('--fc_dim', type=int, default=200,
                        help='dimension of fc layer in LSTM (default: 200)')
    parser.add_argument('--dropout', type=int, default=0.5,
                        help='dropout rate lstm (default: 0.5)')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='batch size (default: 10)')
    parser.add_argument('--clip', type=float, default=1e-3,
                        help='gradient clip value (default: 1e-3)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='initial learning rate (default: 0.001)')
    parser.add_argument('--lr_decay', type=float, default=1,
                        help='Learning rate decay rate (default: 0.99)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs (default: 100)')
    parser.add_argument('--save_dict_treshold', type=float, default=0.50,
                        help='treshold for saving a dict (default: 0.57)')

    # Testing
    parser.add_argument('--load_dict', type=bool, default=False,
                        help='Load a pretrained state_dict (default: False)')
    parser.add_argument('--test_mode', type=bool, default=False,
                        help='test_mode, plot confusion matrix (default: False)')
    parser.add_argument('--audio_dict', type=str, default="state_dict/audioRNN/audioRNN57.12.pt",
                        help='audio pretrained stat dict (default: 57.12)')
    parser.add_argument('--visual_dict', type=str, default="state_dict/visualRNN/visualRNN56.01.pt",
                        help='visual pretrained stat dict (default: 56.01)')
    parser.add_argument('--text_dict', type=str, default="state_dict/textRNN/textRNN68.95.pt",
                        help='text pretrained stat dict (default: 68.95)')
    parser.add_argument('--fusion_dict', type=str, default="state_dict/fusionRNN/fusionRNN69.75.pt",
                        help='text pretrained stat dict (default: 69.75)')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load the dataset
    if args.model == 'audio':
        path = 'audioRNN'
        #_, _, data_train, data_test, _, _, _, _, train_label, test_label, seqlen_train, seqlen_test, train_mask, test_mask = get_iemocap_raw(4)
        _, _, _, _, _, _, data_train, train_label, train_mask, data_test, test_label, test_mask, _, _, seqlen_train, seqlen_test, _, _ = process_features(True)
    elif args.model == 'visual':
        _, _, _, _, _, _, data_train, data_test, train_label, test_label, seqlen_train, seqlen_test, train_mask, test_mask = get_iemocap_raw(4)	
        path = 'visualRNN'	
    elif args.model == 'text':
        data_train, train_label, train_mask, data_test, test_label, test_mask, _, _, _, _, _, _, _, _, _, _, seqlen_train, seqlen_test = process_features(True)
        path = 'textRNN'
    elif args.model == 'fusion':
        data_train, data_test, _, _, _, _, _, _, train_label, test_label, seqlen_train, seqlen_test, train_mask, test_mask = get_iemocap_raw(4)
        path = 'fusionRNN'
    else:
        print("model not found")
        return
    print(args.model + " model loaded")

    # Define the model
    model = LstmModel(input_feat_size = data_train.shape[-1] , output_size = args.output_size, hidden_dim = args.hidden_dim, fc_dim = args.fc_dim , dropout = args.dropout)
    print("model created")

    if args.load_dict or args.test_mode:
        if args.model == 'audio':
            model.load_state_dict(torch.load(args.audio_dict, map_location="cpu"))
        elif args.model == 'visual':
            model.load_state_dict(torch.load(args.visual_dict))
        elif args.model == 'text':
            model.load_state_dict(torch.load(args.text_dict));
        elif args.model == 'fusion':
            model.load_state_dict(torch.load(args.fusion_dict));
        print("stat_dict loaded")

    model = model.to(device)

    # print(sum(p.numel() for p in model.parameters()))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr= args.lr)
    scheduler = StepLR(optimizer, step_size = 1, gamma = args.lr_decay)

    input_test = torch.Tensor(data_test).to(device)
    target_test = torch.Tensor(test_label).to(device)
    target_test = target_test.view(-1).long()

    best_test_epoch = 0
    best_test_acc = 0
    train_acc = 0

    if args.test_mode:
        model.eval()
        with torch.no_grad():
            correct_tst = 0
            output_tst, _ = model(input_test)
            output_tst = output_tst.to(device)
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
            ax.set_title('cLSTM-Speech\n',fontsize = 25.0);
            ax.set_xlabel('Predicted labels',fontsize = 25.0);
            ax.set_ylabel('True labels', fontsize = 25.0); 
            ax.yaxis.set_ticklabels(['HPY', 'SAD', 'NEU', 'ANG'],fontsize = 20.0, rotation = 0);
            ax.xaxis.set_ticklabels(['HPY', 'SAD', 'NEU', 'ANG'],fontsize = 20.0);
            ax.set_aspect("equal")
            plt.savefig("Figure/cm_audio.pdf", bbox_inches = 'tight', pad_inches = 0)
    else:
        for epoch in range(1, args.epochs + 1):
            batches = batch_iter(list(zip(data_train, train_mask, seqlen_train, train_label)), args.batch_size)
            for idx, batch in enumerate(batches):
                model.train()
                b_train_data, b_train_mask, b_seqlen_train, b_train_label = zip(*batch)

                input_train = torch.Tensor(b_train_data).to(device)
                target_train = torch.Tensor(b_train_label).to(device)
                target_train = target_train.view(-1).long()

                optimizer.zero_grad()
                output, hidden = model(input_train)
                output = output.to(device)

                loss = criterion(output, target_train)
                loss.backward() 
                clip_gradient(model, args.clip)
                optimizer.step() 
                scheduler.step()

                # Calculate the unweighted accuracy of train data
                b_train_mask = np.asarray(b_train_mask).reshape(-1)
                train_acc, _ = cal_acc(output, target_train, b_train_mask)
                # cal_metrics(output, target_train, b_train_mask)

            if epoch%1 == 0:
                # Evaluate on the test data:
                model.eval()
                with torch.no_grad():
                    correct_tst = 0
                    output_tst, hidden_tst = model(input_test)
                    output_tst = output_tst.to(device)
                    loss_tst = criterion(output_tst, target_test)
                    accuracy_tst, _ = cal_acc(output_tst, target_test, test_mask)
                    precision, recall,f1,  wa , wf1= cal_metrics(output_tst, target_test, test_mask)
                if accuracy_tst > best_test_acc:
                    best_test_acc = accuracy_tst
                    best_test_epoch = epoch
                    if accuracy_tst > args.save_dict_treshold:
                        torch.save(model.state_dict(), "state_dict/"+path+"/"+path+"{:0.2f}.pt".format(100*accuracy_tst))

                print('Epoch: {}/{}......'.format(epoch, args.epochs), end=' ')
                print("Train Acc:{:.2f}%, Loss:{:.3f}, Test Acc:{:.2f}, Test Best:{:.2f},  Test w_acc:{:.2f}, Test w_f1:{:.2f}%".format(100. * train_acc, loss,  100*accuracy_tst,100*best_test_acc, wa, wf1))

        print('Best Epoch: {}/{}.............'.format(best_test_epoch, args.epochs), end=' ')
        print("Train accuracy:{:.2f} Test accuracy: {:.2f}%".format(100*train_acc, 100. * best_test_acc))

if __name__ == '__main__':
    print("starting")
    main()


