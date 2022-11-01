import csv
import time
import sys
import pickle
import os
import glob
import re
from sentence_transformers import SentenceTransformer
import random
import numpy as np


def get_video_features(file_paths, timestamps):
    pass


def get_audio_features(file_paths, timestamps):
    if os.path.isfile("/Users/ludus/Projects/Sentiment_analysis-OsakaU-/extracted_data/audio_features.pkl"):
        print("Loaded audio-features")
        return pickle.load(open("/Users/ludus/Projects/Sentiment_analysis-OsakaU-/extracted_data/audio_features.pkl", "rb"))
    print("Extracting audio-features...")
    audio_features = {}

    for i, file_name in enumerate(file_paths):
        dialog_id = re.split(r"[/.]", file_name)[-2]
        audio_features[dialog_id] = []
        for timestamp in timestamps[dialog_id]:
            cmd = f"SMILExtract -C opensmile/config/is09-13/IS09_emotion.conf -start {timestamp[0]} -end {timestamp[1]} -I {file_name} -O extracted_data/test.csv -l 0"
            os.system(cmd)
            reader = csv.reader(open("extracted_data/test.csv", 'r'))
            rows = [row for row in reader]
            last_line = rows[-1]
            utterance_audio_features = list(map(lambda x: float(x), last_line[1: 385]))
            audio_features[dialog_id].append(utterance_audio_features)

        print(f"File {i + 1} / {len(file_paths)} files done")

    pickle.dump(audio_features, open("/Users/ludus/Projects/Sentiment_analysis-OsakaU-/extracted_data/audio_features.pkl", "wb"), pickle.HIGHEST_PROTOCOL)
    print("Audio extraction done. Saved features.")
    time.sleep(0.5)

    return audio_features


def get_text_features(file_paths, dialog_utterance_ids):
    if os.path.isfile("/Users/ludus/Projects/Sentiment_analysis-OsakaU-/extracted_data/text_features.pkl"):
        print("Loaded text-features")
        return pickle.load(open("/Users/ludus/Projects/Sentiment_analysis-OsakaU-/extracted_data/text_features.pkl", "rb"))
    print("Extracting text-features...")
    model = SentenceTransformer('all-mpnet-base-v2')    # Load s-bert model for text feature extractions
    text_features = {}  # key: dialog id | value: list of utterances in dialog

    for i, file_name in enumerate(file_paths):
        dialog_id = re.split(r"[/.]", file_name)[-2]
        text_features[dialog_id] = []
        file = open(file_name, "r")
        lines = file.readlines()
        for line in lines:
            line_arr = line.split(" ", 2)
            if line_arr[0] in dialog_utterance_ids[dialog_id]:
                sentence = line_arr[2].strip()
                sentence_embedding = model.encode(sentence)
                text_features[dialog_id].append(sentence_embedding)
        print(f"File {i+1} / {len(file_paths)} files done")

    pickle.dump(text_features, open("/Users/ludus/Projects/Sentiment_analysis-OsakaU-/extracted_data/text_features.pkl","wb"), protocol=pickle.HIGHEST_PROTOCOL)
    print("Text extraction done. Saved features.")
    time.sleep(0.5)
    return text_features


def get_labels(dialog_utterance_ids, label_lookup_list):
    dialog_utterance_labels = {}  # key: dialog id | value: list of label for each utterance in dialog

    for (dialog_id, utterance_ids_elem) in dialog_utterance_ids.items():
        dialog_utterance_labels[dialog_id] = []
        for u_id in utterance_ids_elem:
            dialog_utterance_labels[dialog_id].append(label_lookup_list[u_id])

    return dialog_utterance_labels


def get_utterance_ids(file_paths, label_lookup_dict):
    dialog_utterance_ids = {}  # key: dialog id | value: list of utterances in dialog

    for file_name in file_paths:
        dialog_id = re.split(r"[/.]", file_name)[-2]
        dialog_utterance_ids[dialog_id] = []
        file = open(file_name, "r")
        lines = file.readlines()
        for line in lines:
            utterance_id = line.split(" ")[0]
            if utterance_id in label_lookup_dict:
                dialog_utterance_ids[dialog_id].append(line.split(" ")[0])

    return dialog_utterance_ids


def extract_label_lookup_dict_and_timestamps(file_paths):
    labels = {'hap': 0, 'sad': 1, 'neu': 2, 'ang': 3, 'exc': 4, 'fru': 5}
    label_lookup_dict = {}
    timestamps = {}     # Timestamps [from, to] for each labeled utterance in dialog (in-order)

    for file_name in file_paths:
        dialog_id = re.split(r"[/.]", file_name)[-2]
        timestamps[dialog_id] = []
        file = open(file_name, "r")
        lines = file.readlines()
        for line in lines:
            if line[0] == "[":
                evaluation = re.split(r"[\t]", line)  # [timestamp, utterance_id, label, [V, A, D]\n]
                if evaluation[2] != 'xxx' and evaluation[2] != 'sur' and evaluation[2] != 'fea' \
                        and evaluation[2] != 'dis'\
                        and evaluation[2] != 'oth':  # Remove labels and utterances not used for experiment
                    label_lookup_dict[evaluation[1]] = labels[evaluation[2]]
                    timestamps_arr = evaluation[0].strip("[]").split(" - ")
                    timestamps[dialog_id].append(timestamps_arr)

    return label_lookup_dict, timestamps


def get_file_paths():
    transcript_file_names = []
    evaluation_file_names = []
    audio_file_names = []
    # get file paths
    for i in range(1, 6):
        evaluation_file_names.extend(
            glob.glob("raw_data/IEMOCAP_full_release/Session" + str(i) + "/dialog/EmoEvaluation/*.txt"))
        transcript_file_names.extend(
            glob.glob("raw_data/IEMOCAP_full_release/Session" + str(i) + "/dialog/transcriptions/*.txt"))
        audio_file_names.extend(
            glob.glob("raw_data/IEMOCAP_full_release/Session" + str(i) + "/dialog/wav/*.wav")
        )

    return evaluation_file_names, transcript_file_names, audio_file_names


def train_test_split(file_paths):
    random.shuffle(file_paths)
    train_set = {}
    test_set = {}

    for i, file_name in enumerate(file_paths):
        dialog_id = re.split(r"[/.]", file_name)[-2]
        if i < 120:
            train_set[dialog_id] = []
        else:
            test_set[dialog_id] = []

    return train_set, test_set

def balanced_sampling(file_paths, labels, text_features, audio_features):
    sad = 0
    hap = 0
    angry = 0
    neu = 0
    exc = 0
    fru = 0
    sampling_rates = []

    for i, elem in enumerate(labels):
        for j, _ in enumerate(labels[elem]):
            if labels[elem][j] == 0:
                hap += 1
            if labels[elem][j] == 1:
                sad += 1
            if labels[elem][j] == 2:
                neu += 1
            if labels[elem][j] == 3:
                angry += 1
            if labels[elem][j] == 4:
                exc += 1
            if labels[elem][j] == 5:
                fru += 1

    sampling_rates.append(hap/(hap+exc))
    sampling_rates.append(hap/(sad+fru))
    sampling_rates.append(hap/neu)
    sampling_rates.append(hap/angry)
    sampling_rates.append(hap/(exc+hap))
    sampling_rates.append(hap/(fru+sad))

    random.shuffle(file_paths)
    train_set = {}
    test_set = {}
    train_labels = {}
    test_labels = {}

    for i, file_name in enumerate(file_paths):
        dialog_id = re.split(r"[/.]", file_name)[-2]
        if i < 120:
            train_set[dialog_id] = []
            train_labels[dialog_id] = []
        else:
            test_set[dialog_id] = []
            test_labels[dialog_id] = []

        for j, label in enumerate(labels[dialog_id]):
            match label:
                case 0:
                    if random.random() < sampling_rates[0]:
                        if i < 120:
                            train_set[dialog_id].append(np.concatenate((text_features[dialog_id][j], audio_features[dialog_id][j])))
                            train_labels[dialog_id].append(label)
                        else:
                            test_set[dialog_id].append(np.concatenate((text_features[dialog_id][j], audio_features[dialog_id][j])))
                            test_labels[dialog_id].append(label)
                case 1:
                    if random.random() < sampling_rates[1]:
                        if i < 120:
                            train_set[dialog_id].append(np.concatenate((text_features[dialog_id][j], audio_features[dialog_id][j])))
                            train_labels[dialog_id].append(label)
                        else:
                            test_set[dialog_id].append(np.concatenate((text_features[dialog_id][j], audio_features[dialog_id][j])))
                            test_labels[dialog_id].append(label)
                case 2:
                    if random.random() < sampling_rates[2]:
                        if i < 120:
                            train_set[dialog_id].append(np.concatenate((text_features[dialog_id][j], audio_features[dialog_id][j])))
                            train_labels[dialog_id].append(label)
                        else:
                            test_set[dialog_id].append(np.concatenate((text_features[dialog_id][j], audio_features[dialog_id][j])))
                            test_labels[dialog_id].append(label)
                case 3:
                    if random.random() < sampling_rates[3]:
                        if i < 120:
                            train_set[dialog_id].append(np.concatenate((text_features[dialog_id][j], audio_features[dialog_id][j])))
                            train_labels[dialog_id].append(label)
                        else:
                            test_set[dialog_id].append(np.concatenate((text_features[dialog_id][j], audio_features[dialog_id][j])))
                            test_labels[dialog_id].append(label)
                case 4:
                    if random.random() < sampling_rates[4]:
                        if i < 120:
                            train_set[dialog_id].append(np.concatenate((text_features[dialog_id][j], audio_features[dialog_id][j])))
                            train_labels[dialog_id].append(label)
                        else:
                            test_set[dialog_id].append(np.concatenate((text_features[dialog_id][j], audio_features[dialog_id][j])))
                            test_labels[dialog_id].append(label)
                case 5:
                    if random.random() < sampling_rates[5]:
                        if i < 120:
                            train_set[dialog_id].append(np.concatenate((text_features[dialog_id][j], audio_features[dialog_id][j])))
                            train_labels[dialog_id].append(label)
                        else:
                            test_set[dialog_id].append(np.concatenate((text_features[dialog_id][j], audio_features[dialog_id][j])))
                            test_labels[dialog_id].append(label)
    """sad = 0
    hap = 0
    angry = 0
    neu = 0
    exc = 0
    fru = 0

    for i, elem in enumerate(labels):
        for j, _ in enumerate(labels[elem]):
            if labels[elem][j] == 0:
                hap += 1
            if labels[elem][j] == 1:
                sad += 1
            if labels[elem][j] == 2:
                neu += 1
            if labels[elem][j] == 3:
                angry += 1
            if labels[elem][j] == 4:
                exc += 1
            if labels[elem][j] == 5:
                fru += 1
    print(f"sad: {sad} happy: {hap} angry: {angry} neutural: {neu} excited: {exc} frutrated: {fru}")
    sad = 0
    hap = 0
    angry = 0
    neu = 0
    exc = 0
    fru = 0

    for i, elem in enumerate(train_labels):
        for j, _ in enumerate(train_labels[elem]):
            if train_labels[elem][j] == 0:
                hap += 1
            if train_labels[elem][j] == 1:
                sad += 1
            if train_labels[elem][j] == 2:
                neu += 1
            if train_labels[elem][j] == 3:
                angry += 1
            if train_labels[elem][j] == 4:
                exc += 1
            if train_labels[elem][j] == 5:
                fru += 1
    print(f"sad: {sad} happy: {hap} angry: {angry} neutural: {neu} excited: {exc} frutrated: {fru}")
    sad = 0
    hap = 0
    angry = 0
    neu = 0
    exc = 0
    fru = 0

    for i, elem in enumerate(test_labels):
        for j, _ in enumerate(test_labels[elem]):
            if test_labels[elem][j] == 0:
                hap += 1
            if test_labels[elem][j] == 1:
                sad += 1
            if test_labels[elem][j] == 2:
                neu += 1
            if test_labels[elem][j] == 3:
                angry += 1
            if test_labels[elem][j] == 4:
                exc += 1
            if test_labels[elem][j] == 5:
                fru += 1
    print(f"sad: {sad} happy: {hap} angry: {angry} neutural: {neu} excited: {exc} frutrated: {fru}")"""

    return train_set, train_labels, test_set, test_labels


def main():
    evaluation_file_names, transcript_file_names, audio_file_names = get_file_paths()
    label_lookup_dict, timestamps = extract_label_lookup_dict_and_timestamps(evaluation_file_names)
    dialog_utterance_ids = get_utterance_ids(transcript_file_names, label_lookup_dict)
    dialog_utterance_labels = get_labels(dialog_utterance_ids, label_lookup_dict)
    dialog_text_features = get_text_features(transcript_file_names, dialog_utterance_ids)
    dialog_audio_features = get_audio_features(audio_file_names, timestamps)
    #train_set, test_set, train_labels, test_labels = balanced_sampling(transcript_file_names, dialog_utterance_labels, dialog_text_features, dialog_audio_features)
    train_set, test_set = train_test_split(transcript_file_names)


    pickle.dump([dialog_utterance_ids, dialog_text_features, dialog_audio_features, dialog_utterance_labels, train_set, test_set],
                open("/Users/ludus/Projects/Sentiment_analysis-OsakaU-/extracted_data/combined/IEMOCAP_features_raw.pkl", "wb"),
                protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    animation = ["[        ]",
                 "[=       ]",
                 "[===     ]",
                 "[====    ]",
                 "[=====   ]",
                 "[======  ]",
                 "[======= ]",
                 "[========]\n"]
    for anim in animation:
        sys.stdout.write("\rStarting Extractor " + anim)
        sys.stdout.flush()
        time.sleep(0.5)
    main()
