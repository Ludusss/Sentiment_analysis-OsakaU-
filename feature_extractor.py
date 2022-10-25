import time
import sys
import pickle
import os
import glob
import re
from sentence_transformers import SentenceTransformer
import audiofile
import opensmile
import random


def get_video_features(file_paths, timestamps):
    pass


def get_audio_features(file_paths, timestamps):
    if os.path.isfile("/Users/ludus/Projects/Sentiment_analysis-OsakaU-/extracted_data/audio_features.pkl"):
        print("Loaded audio-features")
        return pickle.load(open("/Users/ludus/Projects/Sentiment_analysis-OsakaU-/extracted_data/audio_features.pkl", "rb"))
    print("Extracting audio-features...")
    audio_features = {}
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
    )

    for i, file_name in enumerate(file_paths):
        dialog_id = re.split(r"[/.]", file_name)[-2]
        audio_features[dialog_id] = []
        for timestamp in timestamps[dialog_id]:
            signal, sampling_rate = audiofile.read(file_name, offset=float(timestamp[0]), duration=(float(timestamp[1]) - float(timestamp[0])))
            utterance_audio_features = smile.process_signal(
                signal,
                sampling_rate
            )
            audio_features[dialog_id].append(utterance_audio_features.iloc[0].values)
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


def get_dataset_ids(file_paths):
    random.shuffle(file_paths)
    train_set_ids = set()
    test_set_ids = set()

    for i, file_name in enumerate(file_paths):
        dialog_id = re.split(r"[/.]", file_name)[-2]
        if i > 119:
            print(len(train_set_ids))
            test_set_ids.add(dialog_id)
        else:
            train_set_ids.add(dialog_id)

    return train_set_ids, test_set_ids


def main():
    evaluation_file_names, transcript_file_names, audio_file_names = get_file_paths()
    train_set, test_set = get_dataset_ids(transcript_file_names)
    label_lookup_dict, timestamps = extract_label_lookup_dict_and_timestamps(evaluation_file_names)
    dialog_utterance_ids = get_utterance_ids(transcript_file_names, label_lookup_dict)
    dialog_utterance_labels = get_labels(dialog_utterance_ids, label_lookup_dict)
    dialog_text_features = get_text_features(transcript_file_names, dialog_utterance_ids)
    dialog_audio_features = get_audio_features(audio_file_names, timestamps)

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
