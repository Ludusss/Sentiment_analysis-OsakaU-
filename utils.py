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
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing


def report_acc_mlp(output, target):
    pred = output.argmax(dim=1)
    acc = (torch.eq(target, pred).sum().item() / len(pred)) * 100
    f1 = metrics.f1_score(target, pred, average='weighted')
    conf_matrix = metrics.confusion_matrix(target, pred)
    classify_report = metrics.classification_report(target, pred, digits=4, zero_division=0)

    return acc, f1, conf_matrix, classify_report


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


def gen_batches_mlp(features, labels, batch_size):
    permutation = torch.randperm(len(features))

    for i in range(0, len(features), batch_size):
        indices = permutation[i:i + batch_size]
        yield zip(features[indices], labels[indices])


def gen_batches(features, labels, mask, batch_size):
    permutation = torch.randperm(len(features))

    for i in range(0, len(features), batch_size):
        indices = permutation[i:i+batch_size]
        yield zip(features[indices], labels[indices], mask[indices])


def process_ESD_features(quad_class=False):
    emo_dict = {
        "Angry": 0,
        "Happy": 1,
        "Neutral": 2,
        "Sad": 3,
        "Surprise": 4
    }

    X_train = []
    X_test = []
    X_val = []
    y_train = []
    y_test = []
    y_val = []

    audio_features = pd.read_csv("extracted_data/ESD/ESD_audio_features_combined.csv")
    """labels = []
    X = []
    for feature_row in audio_features.values:
        labels.append(feature_row[1])
        f0 = feature_row[2]
        mfcc = np.fromstring(feature_row[3].replace("\n", "")[1:-1], sep=" ")
        cqt = np.fromstring(feature_row[4].replace("\n", "")[1:-1], sep=" ")
        X.append(np.hstack((f0, mfcc, cqt)))
    pca = PCA(n_components=3)
    X_r = pca.fit_transform(X)
    cdict = {0: 'red', 1: 'red', 2: 'gray', 3: "green", 4: "green"}
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    for g in np.unique(labels):
        ix = np.where(labels == g)
        ax.scatter(X_r[ix[0][0:50], 0], X_r[ix[0][0:50], 1], X_r[ix[0][0:50], 2], c=cdict[g], label=g, s=100)
    ax.legend()
    plt.show()"""
    neutral_samples = audio_features.label.value_counts()[2]
    oversampled_neutral = audio_features[audio_features["label"] == 2].sample(neutral_samples)
    audio_features = pd.concat([audio_features, oversampled_neutral], axis=0)
    audio_features = audio_features.sample(frac=1)  # Shuffle features
    train_features = audio_features.loc[audio_features['set'] == "train"]
    train_features = train_features.drop(['set'], axis=1)
    test_features = audio_features.loc[audio_features['set'] == "test"]
    test_features = test_features.drop(['set'], axis=1)
    val_features = audio_features.loc[audio_features['set'] == "val"]
    val_features = val_features.drop(['set'], axis=1)

    if quad_class:
        train_features = train_features[train_features.label != 4]
        test_features = train_features[train_features.label != 4]
        val_features = train_features[train_features.label != 4]
        train_features['label'] = train_features['label'].replace([2, 3], [3, 2])
        test_features['label'] = train_features['label'].replace([2, 3], [3, 2])
        val_features['label'] = train_features['label'].replace([2, 3], [3, 2])
        """
        "Angry": 0,
        "Happy": 1,
        "Sad": 2,
        "Neutral": 3,
        """
    else:
        train_features['label'] = train_features['label'].replace([3, 4], [0, 1])
        test_features['label'] = test_features['label'].replace([3, 4], [0, 1])
        val_features['label'] = val_features['label'].replace([3, 4], [0, 1])

    for feature_row in train_features.values:
        y_train.append(feature_row[0])
        f0 = feature_row[1]
        mfcc = np.fromstring(feature_row[2].replace("\n", "")[1:-1], sep=" ")
        cqt = np.fromstring(feature_row[3].replace("\n", "")[1:-1], sep=" ")
        X_train.append(np.hstack((f0, mfcc, cqt)))

    for feature_row in test_features.values:
        y_test.append(feature_row[0])
        f0 = feature_row[1]
        mfcc = np.fromstring(feature_row[2].replace("\n", "")[1:-1], sep=" ")
        cqt = np.fromstring(feature_row[3].replace("\n", "")[1:-1], sep=" ")
        X_test.append(np.hstack((f0, mfcc, cqt)))

    for feature_row in val_features.values:
        y_val.append(feature_row[0])
        f0 = feature_row[1]
        mfcc = np.fromstring(feature_row[2].replace("\n", "")[1:-1], sep=" ")
        cqt = np.fromstring(feature_row[3].replace("\n", "")[1:-1], sep=" ")
        X_val.append(np.hstack((f0, mfcc, cqt)))

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    X_val = np.array(X_val)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_val = np.array(y_val)

    """scaler_train = preprocessing.StandardScaler().fit(X_train)
    scaler_test = preprocessing.StandardScaler().fit(X_test)
    scaler_val = preprocessing.StandardScaler().fit(X_val)
    np.savetxt("useful_variables/train/train_mean.csv", scaler_train.mean_)
    np.savetxt("useful_variables/train/train_variance.csv", scaler_train.scale_)
    np.savetxt("useful_variables/test/test_mean.csv", scaler_test.mean_)
    np.savetxt("useful_variables/test/test_variance.csv", scaler_test.scale_)
    np.savetxt("useful_variables/val/val_mean.csv", scaler_val.mean_)
    np.savetxt("useful_variables/val/val_variance.csv", scaler_val.scale_)

    time.sleep(10000)"""

    scaled_train = preprocessing.StandardScaler().fit_transform(X_train)
    scaled_test = preprocessing.StandardScaler().fit_transform(X_test)
    scaled_val = preprocessing.StandardScaler().fit_transform(X_val)

    return scaled_train, y_train, scaled_test, y_test, scaled_val, y_val

def process_twitter():
    emo_dict_sample = {
        "anger": 0,
        "happiness": 1,
        "sadness": 2,
        "neutral": 3,
        "worry": 4,
        "love": 5,
        "surprise": 6,
        "fun": 7,
        "relief": 8,
        "hate": 9,
        "empty": 10,
        "enthusiasm": 11,
        "boredom": 12
    }
    emo_dict = {
        "neg": 0,
        "pos": 1,
        "neu": 2
    }

    X_train = []
    X_test = []
    X_val = []
    y_train = []
    y_test = []
    y_val = []

    sentiment_data_df = pd.read_csv("extracted_data/twitter/twitter_text_features.csv")
    sentiment_data_df["sentiment"] = sentiment_data_df["sentiment"].replace(["neutral", "anger", "happiness", "worry", "fun", "relief"],
                                                                            ["neu", "neg", "pos", "neg", "pos", "pos"])
    sentiment_data_df = sentiment_data_df[~sentiment_data_df['sentiment'].isin(["sadness", "love", "surprise", "hate", "empty", "enthusiasm", "boredom"])]
    sentiment_data_df = sentiment_data_df.sample(frac=1)
    indices = np.arange(sentiment_data_df.shape[0])
    train_idx, test_idx, val_idx = np.split(indices, [int(len(indices)*0.8), int(len(indices)*0.9)])

    for i in train_idx:
        X_train.append(np.fromstring(sentiment_data_df.iloc[i]["content"].replace("\n", "")[1:-1], sep=" "))
        y_train.append(emo_dict[sentiment_data_df.iloc[i]["sentiment"]])
    for i in test_idx:
        X_test.append(np.fromstring(sentiment_data_df.iloc[i]["content"].replace("\n", "")[1:-1], sep=" "))
        y_test.append(emo_dict[sentiment_data_df.iloc[i]["sentiment"]])
    for i in val_idx:
        X_val.append(np.fromstring(sentiment_data_df.iloc[i]["content"].replace("\n", "")[1:-1], sep=" "))
        y_val.append(emo_dict[sentiment_data_df.iloc[i]["sentiment"]])
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test), np.array(X_val), np.array(y_val)


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
        prev_idx = prev_idx + seq_len

    # Split text features into train/test
    rand_batches = np.random.permutation(len(text_features))
    text_features_train, text_labels_train, text_mask_train = np.array(text_features)[rand_batches[:120]], np.array(text_labels)[rand_batches[:120]], np.array(text_mask)[rand_batches[:120]]
    text_features_test, text_labels_test, text_mask_test = np.array(text_features)[rand_batches[120:151]], np.array(text_labels)[rand_batches[120:151]], np.array(text_mask)[rand_batches[120:151]]

    # Split audio features into train/test
    rand_batches = np.random.permutation(len(audio_features))
    audio_features_train, audio_labels_train, audio_mask_train = np.array(audio_features)[rand_batches[:120]], np.array(audio_labels)[rand_batches[:120]], np.array(audio_mask)[rand_batches[:120]]
    audio_features_test, audio_labels_test, audio_mask_test = np.array(audio_features)[rand_batches[120:151]], np.array(audio_labels)[rand_batches[120:151]], np.array(audio_mask)[rand_batches[120:151]]

    return text_features_train, text_labels_train, text_mask_train, text_features_test, text_labels_test, text_mask_test, audio_features_train, audio_labels_train, audio_mask_train, audio_features_test, audio_labels_test, audio_mask_test, text_features, audio_features


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
