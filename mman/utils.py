import time

import torch
from torch import nn
from sklearn import metrics
import numpy as np
import pickle
import sys
import os
from sklearn import preprocessing
import pandas as pd


def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)

def cal_acc_without_mask(output, target):
    pred = output.argmax(dim=1, keepdims=True)
    correct = 0
    total = 0
    for i in range(len(output)):
        total += 1
        if (pred[i] == target[i]):
            correct += 1
    return correct/total

def cal_acc(output, target, mask):
    pred = output.argmax(dim=1, keepdim=True)
    target = target.reshape(-1)
    correct = 0
    total = 0
    misclassified_mask = []
    for i in range(len(mask)):
        if (mask[i] ==1):
            total += 1
            if (pred[i] == target[i]):
                correct += 1
            else:
                misclassified_mask.append(i)
    return correct/total, np.array(misclassified_mask)

def cal_metrics(output, target, mask):
    pred = output.argmax(dim=1, keepdim=True)
    correct = np.zeros([4])
    total = np.zeros([4])
    predict = np.zeros([4])
    for i in range(len(mask)):
        if (mask[i] ==1):
            if (target[i] == 0): total[0] += 1
            elif (target[i] == 1): total[1] += 1
            elif (target[i] == 2): total[2] += 1
            elif (target[i] == 3): total[3] += 1
            else: print("error 1")
            if (pred[i] == 0): predict[0] += 1
            elif (pred[i] == 1): predict[1] += 1
            elif (pred[i] == 2): predict[2] += 1
            elif (pred[i] == 3): predict[3]+= 1
            else: print("error 2")
            if (pred[i] == target[i]):
                if(pred[i] == 0):
                    correct[0] += 1
                elif (pred[i] == 1):
                    correct[1] += 1
                elif (pred[i] == 2):
                    correct[2]+= 1
                elif (pred[i] == 3):
                    correct[3] += 1
                else: print("error 3")

    # print(correct)
    # print(predict)
    # print(total)
    precision = correct/predict
    recall = correct/total
    f1 = 2 * (precision * recall)/(precision + recall)

    w_a = np.sum(precision * total) / np.sum(total)
    w_f1 = np.sum(f1 * total) / np.sum(total)
    return precision, recall, f1, w_a, w_f1

def report_acc(output, target, mask):
    pred = output.argmax(dim=1, keepdim=True)
    #pred = pred.cpu().numpy()
    #target = target.cpu().numpy()
    pred = np.extract(mask, pred)
    target = np.extract(mask, target)
    cm = metrics.confusion_matrix(target, pred)
    print(metrics.classification_report(target, pred))
    
    return cm

import pickle
import sys
import os
import numpy as np


def createOneHot(train_label, test_label):
    maxlen = int(max(train_label.max(), test_label.max()))

    train = np.zeros((train_label.shape[0], train_label.shape[1], maxlen + 1))
    test = np.zeros((test_label.shape[0], test_label.shape[1], maxlen + 1))

    for i in range(train_label.shape[0]):
        for j in range(train_label.shape[1]):
            train[i, j, train_label[i, j]] = 1

    for i in range(test_label.shape[0]):
        for j in range(test_label.shape[1]):
            test[i, j, test_label[i, j]] = 1

    return train, test


def batch_iter(data, batch_size, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    # Shuffle the data at each epoch
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
    else:
        shuffled_data = data
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield shuffled_data[start_index:end_index]

def process_features(quad_class=False):
    """labels = {"neg": 0,
              "neu": 1,
              "pos": 2}"""

    """emotion_dict = {'ang': 0,
                    'hap': 1,
                    'exc': 2,
                    'sad': 3,
                    'fru': 4,
                    'fea': 5,
                    'sur': 6,
                    'neu': 7,
                    'dis': 8,
                    'xxx': 9,
                    'oth': 9}"""

    text_features_df = pd.read_csv("extracted_data/text_features.csv")
    audio_features_df = pd.read_csv("extracted_data/audio_features.csv")

    # Remove unused labels 4 classes
    if quad_class:
        text_features_df = text_features_df[text_features_df.label != "fea"]
        text_features_df = text_features_df[text_features_df.label != "sur"]
        text_features_df = text_features_df[text_features_df.label != "dis"]
        text_features_df = text_features_df[text_features_df.label != "xxx"]
        text_features_df = text_features_df[text_features_df.label != "oth"]
        audio_features_df = audio_features_df[audio_features_df.label != 5]
        audio_features_df = audio_features_df[audio_features_df.label != 6]
        audio_features_df = audio_features_df[audio_features_df.label != 8]
        audio_features_df = audio_features_df[audio_features_df.label != 9]

        # Reduce classes via concatenation 4 classes
        text_features_df['label'] = text_features_df['label'].replace(['ang', 'hap', 'exc', 'sad', 'fru', 'neu'],
                                                                      [0, 1, 1, 2, 2, 3])
        audio_features_df['label'] = audio_features_df['label'].replace([2, 3, 4, 7],
                                                                        [1, 2, 2, 3])

    # Remove unused labels 3 classes
    else:
        text_features_df = text_features_df[text_features_df.label != "dis"]
        text_features_df = text_features_df[text_features_df.label != "xxx"]
        text_features_df = text_features_df[text_features_df.label != "oth"]
        audio_features_df = audio_features_df[audio_features_df.label != 8]
        audio_features_df = audio_features_df[audio_features_df.label != 9]

        # reduce classes via concatenation 3 classes
        text_features_df['label'] = text_features_df['label'].replace(['ang', 'sad', 'fea', 'fru', 'hap', 'exc', 'sur', 'neu'],
                                                                      [0, 0, 0, 0, 1, 1, 1, 2])
        audio_features_df['label'] = audio_features_df['label'].replace([3, 4, 5, 2, 6, 7],
                                                                        [0, 0, 0, 1, 1, 2])

    cond = text_features_df['utterance_id'].isin(audio_features_df['utterance_id']) != True
    text_features_df.drop(text_features_df[cond].index, inplace=True)
    # Get max sequence length
    batch = 0
    prev_batch = "Ses01F_impro01"
    sequence_lengths = [0]
    for index, row in text_features_df.iterrows():
        row_info = row["utterance_id"].split("_")
        if len(row_info) == 3:
            row_info = "_".join(row_info[:2])
        elif len(row_info) == 4:
            row_info = "_".join(row_info[:3])
        if prev_batch != row_info:
            sequence_lengths.append(0)
            batch += 1
            prev_batch = row_info
        sequence_lengths[batch] += 1

    batch_size = len(sequence_lengths)
    max_seq = max(sequence_lengths)

    text_features = []
    text_labels = []
    text_mask = []
    prev_idx = 0
    seq_lens_text = []
    for i, seq_len in enumerate(sequence_lengths):
        seq_lens_text.append(seq_len)
        # Labels (batch, seq) with padded seq
        text_labels.append(np.pad(text_features_df[prev_idx:seq_len+prev_idx]["label"].T.to_numpy(), (0, max_seq - seq_len), "constant", constant_values=0))

        # Features (batch, seq, feature) with padded seq
        pad = [np.zeros(np.fromstring(text_features_df.iloc[0, 3][1:-1], sep=" ").shape[0])] * (max_seq - seq_len)
        cleaned_features = list([elem.replace('\n', '')[2:-1] for elem in text_features_df[prev_idx:seq_len+prev_idx]["b_features"].values])  # Remove inner paranthesis and \n tokens from strings
        cleaned_features = [np.fromstring(features, sep=" ") for features in cleaned_features]
        text = np.stack(cleaned_features + pad, axis=0)
        text_features.append(text)

        # Text mask (batch, seq) with padded seq
        text_mask.append(np.zeros(max_seq))
        text_mask[i][:seq_len] = 1
        prev_idx = prev_idx + seq_len

    # Get max sequence length
    batch = 0
    prev_batch = "Ses01F_impro01"
    sequence_lengths = [0]
    for index, row in audio_features_df.iterrows():
        row_info = row["utterance_id"].split("_")
        if len(row_info) == 3:
            row_info = "_".join(row_info[:2])
        elif len(row_info) == 4:
            row_info = "_".join(row_info[:3])
        if prev_batch != row_info:
            sequence_lengths.append(0)
            batch += 1
            prev_batch = row_info
        sequence_lengths[batch] += 1

    batch_size = len(sequence_lengths)
    max_seq = max(sequence_lengths)

    audio_features = []
    audio_labels = []
    audio_mask = []
    prev_idx = 0
    seq_lens_audio = []
    for i, seq_len in enumerate(sequence_lengths):
        seq_lens_audio.append(seq_len)
        # Labels (batch, seq) with padded seq
        audio_labels.append(
            np.pad(audio_features_df[prev_idx:seq_len + prev_idx]["label"].T.to_numpy(), (0, max_seq - seq_len),
                   "constant", constant_values=0))

        # Features (batch, seq, feature) with padded seq
        pad = [np.zeros(33)] * (max_seq - seq_len)
        f0 = audio_features_df[prev_idx:seq_len+prev_idx]["f0"].values
        mfcc = [elem.replace("\n", "")[1:-1] for elem in audio_features_df[prev_idx:seq_len+prev_idx]["mfcc"].values]
        cqt = [elem.replace("\n", "")[1:-1] for elem in audio_features_df[prev_idx:seq_len+prev_idx]["cqt"].values]
        cleaned_features = np.hstack((f0.reshape(seq_len, 1), [np.fromstring(feature, sep=" ") for feature in mfcc],
                                          [np.fromstring(feature, sep=" ") for feature in cqt])).tolist()
        cleaned_features = preprocessing.StandardScaler().fit_transform(cleaned_features).tolist()
        audio = np.stack(cleaned_features + pad, axis=0)
        audio_features.append(audio)

        # Text mask (batch, seq) with padded seq
        audio_mask.append(np.zeros(max_seq))
        audio_mask[i][:seq_len] = 1
        prev_idx = prev_idx + seq_len

    # Split text features into train/test
    text_features_train, text_labels_train, text_mask_train, text_train_seq_len = np.array(text_features)[:120], np.array(text_labels)[:120], np.array(text_mask)[:120], np.array(seq_lens_text)[:120]
    text_features_test, text_labels_test, text_mask_test, text_test_seq_len = np.array(text_features)[120:151], np.array(text_labels)[120:151], np.array(text_mask)[120:151], np.array(seq_lens_text)[120:151]

    # Split audio features into train/test
    audio_features_train, audio_labels_train, audio_mask_train, audio_train_seq_len = np.array(audio_features)[:120], np.array(audio_labels)[:120], np.array(audio_mask)[:120], np.array(seq_lens_audio)[:120]
    audio_features_test, audio_labels_test, audio_mask_test, audio_test_seq_len = np.array(audio_features)[120:151], np.array(audio_labels)[120:151], np.array(audio_mask)[120:151], np.array(seq_lens_audio)[120:151]

    audio_mask_test = audio_mask_test.reshape(3286)
    text_mask_test = text_mask_test.reshape(3286)

    return text_features_train, text_labels_train, text_mask_train, text_features_test, text_labels_test, text_mask_test, audio_features_train, audio_labels_train, audio_mask_train, audio_features_test, audio_labels_test, audio_mask_test, text_features, audio_features, audio_train_seq_len, audio_test_seq_len, text_train_seq_len, text_test_seq_len

def get_iemocap_raw(classes):
    if sys.version_info[0] == 2:
        f = open("raw_data/IEMOCAP_features_raw.pkl", "rb")
        videoIDs, videoSpeakers, videoLabels, videoText, videoAudio, videoVisual, videoSentence, trainVid, testVid = pickle.load(
            f)

        '''
        label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
        '''
    else:
        f = open("raw_data/IEMOCAP_features_raw.pkl", "rb")
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        videoIDs, videoSpeakers, videoLabels, videoText, videoAudio, videoVisual, videoSentence, trainVid, testVid = u.load()
        '''
        label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
        '''

    # print(len(trainVid))
    # print(len(testVid))

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

    # print(test_seq_len)

    max_len = max(max(train_seq_len), max(test_seq_len))
    # print('max_len', max_len)
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

    # print(len(test_text))
    # print(len(train_audio))
    # print(len(train_visual))

    train_text = np.stack(train_text, axis=0)
    train_audio = np.stack(train_audio, axis=0)
    train_visual = np.stack(train_visual, axis=0)
    # print(train_text.shape)
    # print(train_audio.shape)
    # print(train_visual.shape)

    # print()
    test_text = np.stack(test_text, axis=0)
    test_audio = np.stack(test_audio, axis=0)
    test_visual = np.stack(test_visual, axis=0)
    # print(test_text.shape)
    # print(test_audio.shape)
    # print(test_visual.shape)

    train_label = np.array(train_label)
    test_label = np.array(test_label)
    train_seq_len = np.array(train_seq_len)
    test_seq_len = np.array(test_seq_len)
    # print(train_label.shape)
    # print(test_label.shape)
    # print(np.sum(train_seq_len))


    train_mask = np.zeros((train_text.shape[0], train_text.shape[1]), dtype='float')
    for i in range(len(train_seq_len)):
        train_mask[i, :train_seq_len[i]] = 1.0

    test_mask = np.zeros((test_text.shape[0], test_text.shape[1]), dtype='float')
    for i in range(len(test_seq_len)):
        test_mask[i, :test_seq_len[i]] = 1.0

    for i in range(train_label.shape[0]):
        for j in range(train_label.shape[1]):
            if train_label[i][j]==4: train_label[i][j]=0
            if train_label[i][j]==5: 
                train_label[i][j]=1
                # train_mask[i][j]=0

    for i in range(test_label.shape[0]):
        for j in range(test_label.shape[1]):
            if test_label[i][j]==4: test_label[i][j]=0
            if test_label[i][j]==5: 
                test_label[i][j]=1
                # test_mask[i][j]=0


    # train_label, test_label = createOneHot(train_label, test_label)

    train_data = np.concatenate((train_audio, train_visual, train_text), axis=-1)
    test_data = np.concatenate((test_audio, test_visual, test_text), axis=-1)

    # print(train_audio.shape)
    # print(train_text.shape)
    # print(train_visual.shape)
    # print(test_mask.shape)
    # print(train_mask.shape)
    test_mask = test_mask.reshape(3410)
    # print(test_mask.shape)
    # train_mask = train_mask.reshape(13200)

    return train_data, test_data, train_audio, test_audio, train_text, test_text, train_visual, test_visual, train_label, test_label, train_seq_len, test_seq_len, train_mask, test_mask



if __name__ == '__main__':
    train_data, test_data, audio_train, audio_test, text_train, text_test, video_train, video_test, train_label, test_label, seqlen_train, seqlen_test, train_mask, test_mask = get_iemocap_raw(4)
    # print(test_mask.shape)
    test_mask = test_mask.reshape(3410)
    # print(test_mask.shape)

    mask =   np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1])
    y_true = np.array([1,1,1,1,1,1,0,0,0,0,0,3,3,3,0,0,0,0,0,0,2,2,2,0,1,0,2,0,1,0,1,0])
    y_pred = np.array([1,1,1,2,2,1,1,1,0,0,0,0,3,3,1,0,1,2,2,0,0,2,2,0,1,0,1,0,1,0,1,0])
    print(metrics.classification_report(y_true, y_pred))

    # precision, recall,f1,  wa , wf1= cal_metrics(y_pred, y_true, mask)

    report_acc(y_pred, y_true, mask)

    # print(precision)
    # print(recall)
    # print(f1)
    # print(wa)
    # print(wf1)