import numpy as np
import pickle
import sys

def get_iemocap_data(classes):
    f = open("data/IEMOCAP_features_raw.pkl", "rb")
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    videoIDs, videoSpeakers, videoLabels, videoText, videoAudio, videoVisual, videoSentence, trainVid, testVid = u.load()
    '''
    label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
    '''
    #print(len(trainVid))
    # for vid in trainVid:
	# videoIDs[vid] = List of utterance IDs in this video in the order of occurance
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

    #print(test_seq_len)
    #print(train_seq_len)

    max_len = max(max(train_seq_len), max(test_seq_len))
    #print('max_len', max_len)
    t = 0
    for vid in trainVid:
        train_label.append(videoLabels[vid] + [0] * (max_len - len(videoIDs[vid])))
        if t == 0:
            print(np.array(train_label))
            t += 1

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

    #print(len(train_text))
    #print(len(train_audio))
    #print(len(train_visual))
    
    train_text = np.stack(train_text, axis=0)
    train_audio = np.stack(train_audio, axis=0)
    train_visual = np.stack(train_visual, axis=0)
    #print(train_text.shape)
    #print(train_audio.shape)
    #print(train_visual.shape)

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
    #print(train_label.shape)
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