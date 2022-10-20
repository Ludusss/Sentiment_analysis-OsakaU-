import time
import sys
import pickle
import glob
import re
from sentence_transformers import SentenceTransformer
import numpy as np

def get_text_features(file_paths, dialog_utterance_ids):
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


def extract_label_lookup_dict(file_paths):
    labels = {'hap': 0, 'sad': 1, 'neu': 2, 'ang': 3, 'exc': 4, 'fru': 5}
    label_lookup_dict = {}

    for file_name in file_paths:
        file = open(file_name, "r")
        lines = file.readlines()
        for line in lines:
            if line[0] == "[":
                evaluation = re.split(r"[\t]", line)  # [timestamp, utterance_id, label, [V, A, D]\n]
                if evaluation[2] != 'xxx' and evaluation[2] != 'sur' and evaluation[2] != 'fea' \
                        and evaluation[2] != 'dis'\
                        and evaluation[2] != 'oth':  # Remove labels and utterances not used for experiment
                    label_lookup_dict[evaluation[1]] = labels[evaluation[2]]

    return label_lookup_dict


def get_file_paths():
    transcript_file_names = []
    evaluation_file_names = []

    # get file paths
    for i in range(1, 6):
        evaluation_file_names.extend(
            glob.glob("raw_data/IEMOCAP_full_release/Session" + str(i) + "/dialog/EmoEvaluation/*.txt"))
        transcript_file_names.extend(
            glob.glob("raw_data/IEMOCAP_full_release/Session" + str(i) + "/dialog/transcriptions/*.txt"))

    return evaluation_file_names, transcript_file_names


def main():
    evaluation_file_names, transcript_file_names = get_file_paths()
    label_lookup_dict = extract_label_lookup_dict(evaluation_file_names)
    dialog_utterance_ids = get_utterance_ids(transcript_file_names, label_lookup_dict)
    dialog_utterance_labels = get_labels(dialog_utterance_ids, label_lookup_dict)
    dialog_text_features = get_text_features(transcript_file_names, dialog_utterance_ids)
    print(len(dialog_text_features))



if __name__ == '__main__':
    """animation = ["[        ]",
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
        time.sleep(0.5)"""
    main()
