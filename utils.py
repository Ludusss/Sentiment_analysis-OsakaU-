import numpy as np
import pickle
import torch
import sys
from sklearn import metrics
from operator import eq


def report_acc(output, target, mask):
    pred = output.argmax(dim=1, keepdim=True)
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    pred = np.extract(mask, pred)
    target = np.extract(mask, target)

    acc = (sum(map(eq, target, pred)) / len(pred)) * 100.0
    f1 = metrics.f1_score(target, pred, average='weighted')
    conf_matrix = metrics.confusion_matrix(target, pred)
    classify_report = metrics.classification_report(target, pred, digits=4)

    return acc, f1, conf_matrix, classify_report

def gen_bashes(features, labels, mask, batch_size, shuffle=True):
    permutation = torch.randperm(len(features))

    for i in range(0, len(features), batch_size):
        indices = permutation[i:i+batch_size]
        yield zip(features[indices], labels[indices], mask[indices])


def get_iemocap_data():
    f = open("data/IEMOCAP_features_raw.pkl", "rb")
    if sys.version_info[0] == 2:
        videoIDs, videoSpeakers, videoLabels, videoText, videoAudio, videoVisual, videoSentence, trainVid, testVid = pickle.load(f)
    else:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        videoIDs, videoSpeakers, videoLabels, videoText, videoAudio, videoVisual, videoSentence, trainVid, testVid = u.load()
    '''
    label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
    '''
    # print(len(trainVid))
    # for vid in trainVid:
    # videoIDs[vid] = List of utterance IDs in this video in the order of occurrences
    # videoSpeakers[vid] = List of speaker turns. e.g. [M, M, F, M, F]. here M = Male, F = Female
    # videoText[vid] = List of textual features for each utterance in video vid
    # videoAudio[vid] = List of audio features for each utterance in video vid
    # videoVisual[vid] = List of visual features for each utterance in video vid
    # videoLabels[vid] = List of label indices for each utterance in video vid
    # videoSentence[vid] = List of sentences for each utterance in video vid
    train_audio = []
    train_text = []
    train_visual = []
    train_seq_len = []
    train_label = []

    test_audio = []
    test_text = []
    test_visual = []
    test_seq_len = []
    test_label = []

    for vid in trainVid:
        train_seq_len.append(len(videoIDs[vid]))
    for vid in testVid:
        test_seq_len.append(len(videoIDs[vid]))

    max_len = max(max(train_seq_len), max(test_seq_len))

    for vid in trainVid:
        train_label.append(videoLabels[vid] + [0] * (max_len - len(videoIDs[vid])))

        pad = [np.zeros(videoText[vid][0].shape)] * (max_len - len(videoIDs[vid]))
        text = np.stack(videoText[vid] + pad, axis=0)
        train_text.append(text)

        pad = [np.zeros(videoAudio[vid][0].shape)] * (max_len - len(videoIDs[vid]))
        audio = np.stack(videoAudio[vid] + pad, axis=0)
        train_audio.append(audio)

        pad = [np.zeros(videoVisual[vid][0].shape)] * (max_len - len(videoIDs[vid]))
        video = np.stack(videoVisual[vid] + pad, axis=0)
        train_visual.append(video)

    for vid in testVid:
        test_label.append(videoLabels[vid] + [0] * (max_len - len(videoIDs[vid])))

        pad = [np.zeros(videoText[vid][0].shape)] * (max_len - len(videoIDs[vid]))
        text = np.stack(videoText[vid] + pad, axis=0)
        test_text.append(text)

        pad = [np.zeros(videoAudio[vid][0].shape)] * (max_len - len(videoIDs[vid]))
        audio = np.stack(videoAudio[vid] + pad, axis=0)
        test_audio.append(audio)

        pad = [np.zeros(videoVisual[vid][0].shape)] * (max_len - len(videoIDs[vid]))
        video = np.stack(videoVisual[vid] + pad, axis=0)
        test_visual.append(video)

    train_text = np.stack(train_text, axis=0)
    train_audio = np.stack(train_audio, axis=0)
    train_visual = np.stack(train_visual, axis=0)

    test_text = np.stack(test_text, axis=0)
    test_audio = np.stack(test_audio, axis=0)
    test_visual = np.stack(test_visual, axis=0)

    train_label = np.array(train_label)
    test_label = np.array(test_label)
    train_seq_len = np.array(train_seq_len)
    test_seq_len = np.array(test_seq_len)

    train_mask = np.zeros((train_text.shape[0], train_text.shape[1]), dtype='float')
    for i in range(len(train_seq_len)):
        train_mask[i, :train_seq_len[i]] = 1.0

    test_mask = np.zeros((test_text.shape[0], test_text.shape[1]), dtype='float')
    for i in range(len(test_seq_len)):
        test_mask[i, :test_seq_len[i]] = 1.0

    for i in range(train_label.shape[0]):
        for j in range(train_label.shape[1]):
            if train_label[i][j] == 4:  # set excited to happy
                train_label[i][j] = 0
            if train_label[i][j] == 5:  # set frustrated to sad
                train_label[i][j] = 1
                # train_mask[i][j]=0

    for i in range(test_label.shape[0]):
        for j in range(test_label.shape[1]):
            if test_label[i][j] == 4:  # set excited to happy
                test_label[i][j] = 0
            if test_label[i][j] == 5:  # set frustrated to sad
                test_label[i][j] = 1
                # test_mask[i][j]=0

    train_data = np.concatenate((train_audio, train_visual, train_text), axis=-1)
    test_data = np.concatenate((test_audio, test_visual, test_text), axis=-1)

    test_mask = test_mask.reshape(3410)

    return train_data, test_data, train_audio, test_audio, train_text, test_text, train_visual, test_visual, train_label, test_label, train_seq_len, test_seq_len, train_mask, test_mask
