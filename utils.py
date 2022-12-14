import random
import time
import numpy as np
import pickle
import torch
import sys
from sklearn import metrics
from operator import eq
import pandas as pd
import itertools


def report_acc(output, target, mask):
    pred = output.argmax(dim=1, keepdim=True)
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    pred = np.extract(mask, pred)
    target = np.extract(mask, target)

    acc = (sum(map(eq, target, pred)) / len(pred)) * 100.0
    f1 = metrics.f1_score(target, pred, average='weighted')
    conf_matrix = metrics.confusion_matrix(target, pred)
    classify_report = metrics.classification_report(target, pred, digits=4, zero_division=0)

    return acc, f1, conf_matrix, classify_report


def gen_bashes(features, labels, mask, batch_size):
    permutation = torch.randperm(len(features))

    for i in range(0, len(features), batch_size):
        indices = permutation[i:i+batch_size]
        yield zip(features[indices], labels[indices], mask[indices])


def process_features():
    """labels = {"neg": 0,
              "neu": 1,
              "pos": 2}"""

    text_features_df = pd.read_csv("extracted_data/text_features.csv")
    audio_features_df = pd.read_csv("extracted_data/audio_features.csv")
    audio_features = []
    audio_labels = []

    # Remove unused labels
    text_features_df = text_features_df[text_features_df.label != "xxx"]
    text_features_df = text_features_df[text_features_df.label != "fea"]
    text_features_df = text_features_df[text_features_df.label != "oth"]
    text_features_df = text_features_df[text_features_df.label != "dis"]
    audio_features_df = audio_features_df[audio_features_df.label != 9]
    audio_features_df = audio_features_df[audio_features_df.label != 8]
    audio_features_df = audio_features_df[audio_features_df.label != 6]
    audio_features_df = audio_features_df[audio_features_df.label != 5]

    # Reduce classes via concatenation
    text_features_df['label'] = text_features_df['label'].replace(['fru', 'ang', 'sad', 'neu', 'exc', 'hap', 'sur'], [0, 0, 0, 1, 2, 2, 2])
    audio_features_df['label'] = audio_features_df['label'].replace([3, 4, 7, 1, 6],
                                                            [0, 0, 1, 2, 2])
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
    for i, seq_len in enumerate(sequence_lengths):
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
        prev_idx = seq_len - 1

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
    for i, seq_len in enumerate(sequence_lengths):
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
        audio = np.stack(cleaned_features + pad, axis=0)
        audio_features.append(audio)

        # Text mask (batch, seq) with padded seq
        audio_mask.append(np.zeros(max_seq))
        audio_mask[i][:seq_len] = 1
        prev_idx = seq_len - 1

def get_extracted_data():
    train_text = []
    train_audio = []
    train_seq_len = []
    train_labels = []

    test_text = []
    test_audio = []
    test_seq_len = []
    test_labels = []

    f = open("extracted_data/combined/IEMOCAP_features_raw.pkl", "rb")
    utterance_ids, text_features, audio_features, labels, train_set, test_set = pickle.load(f)

    """f = open("pre_extracted_features/IEMOCAP_features_raw.pkl", "rb")
    if sys.version_info[0] == 2:
        videoIDs, videoSpeakers, videoLabels, videoText, videoAudio, videoVisual, videoSentence, trainVid, testVid = pickle.load(
            f)
    else:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        videoIDs, videoSpeakers, videoLabels, videoText, videoAudio, videoVisual, videoSentence, trainVid, testVid = u.load()

    for video in videoIDs:
        if len(videoLabels[video]) == len(labels[video]):
            continue
        else:
            print(video)
            print(videoIDs[video])
            print(utterance_ids[video])
            print(videoLabels[video])
            print(labels[video])"""

    for train_dialog_id in train_set.keys():
        train_seq_len.append(len(utterance_ids[train_dialog_id]))
    for test_dialog_id in test_set.keys():
        test_seq_len.append(len(utterance_ids[test_dialog_id]))

    max_len = max(max(train_seq_len), max(test_seq_len))

    for train_dialog_id in train_set:
        train_labels.append(labels[train_dialog_id] + [0] * (max_len - len(utterance_ids[train_dialog_id])))

        pad = [np.zeros(text_features[train_dialog_id][0].shape)] * (max_len - len(utterance_ids[train_dialog_id]))
        text = np.stack(text_features[train_dialog_id] + pad, axis=0)
        train_text.append(text)

        pad = [np.zeros(np.asarray(audio_features[train_dialog_id][0]).shape)] * (max_len - len(utterance_ids[train_dialog_id]))
        audio = np.stack(audio_features[train_dialog_id] + pad, axis=0)
        train_audio.append(audio)

    for test_dialog_id in test_set:
        test_labels.append(labels[test_dialog_id] + [0] * (max_len - len(utterance_ids[test_dialog_id])))

        pad = [np.zeros(text_features[test_dialog_id][0].shape)] * (max_len - len(utterance_ids[test_dialog_id]))
        text = np.stack(text_features[test_dialog_id] + pad, axis=0)
        test_text.append(text)

        pad = [np.zeros(np.asarray(audio_features[test_dialog_id][0]).shape)] * (max_len - len(utterance_ids[test_dialog_id]))
        text = np.stack(audio_features[test_dialog_id] + pad, axis=0)
        test_audio.append(text)

    train_text = np.stack(train_text, axis=0)
    train_audio = np.stack(train_audio, axis=0)

    test_text = np.stack(test_text, axis=0)
    test_audio = np.stack(test_audio, axis=0)

    train_label = np.array(train_labels)
    test_label = np.array(test_labels)
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

    train_data = np.concatenate((train_audio, train_text), axis=-1)
    test_data = np.concatenate((test_audio, test_text), axis=-1)

    test_mask = test_mask.reshape(3410)

    return train_data, test_data, train_text, train_audio, test_text, test_audio, train_label, test_label, train_seq_len, test_seq_len, train_mask, test_mask


def get_iemocap_data():
    f = open("pre_extracted_features/IEMOCAP_features_raw.pkl", "rb")
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

    train_data = np.concatenate((train_audio, train_text), axis=-1)
    test_data = np.concatenate((test_audio, test_text), axis=-1)

    test_mask = test_mask.reshape(3410)

    return train_data, test_data, train_audio, test_audio, train_text, test_text, train_visual, test_visual, train_label, test_label, train_seq_len, test_seq_len, train_mask, test_mask
