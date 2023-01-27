import re
import os
import time
import pandas as pd
import librosa
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import math
import librosa.display
import random
import sklearn
import soundfile as sf
from sentence_transformers import SentenceTransformer
from sys import platform

SAMP_RATE = 22050


def extract_iemocap_info():
    if platform == "darwin":
        if os.path.isfile("/Users/ludus/Projects/Sentiment_analysis-OsakaU-/extracted_data/df_iemocap.csv"):
            print("Loaded iemocap_info")
            return pd.read_csv("/Users/ludus/Projects/Sentiment_analysis-OsakaU-/extracted_data/df_iemocap.csv")
    elif platform == "win32":
        if os.path.isfile("C:/Projects/Sentiment_analysis-OsakaU-/extracted_data/df_iemocap.csv"):
            print("Loaded iemocap_info")
            return pd.read_csv("C:/Projects/Sentiment_analysis-OsakaU-/extracted_data/df_iemocap.csv")

    info_line = re.compile(r'\[.+\]\n', re.IGNORECASE)

    start_times, end_times, utterance_ids, emotions, vals, acts, doms = [], [], [], [], [], [], []
    df_iemocap = pd.DataFrame(columns=['utterance_id', 'start_time', 'end_time', 'emotion', 'val', 'act', 'dom'])
    ordered_utterance_ids = []

    print("Extraction info from IEMOCAP...")
    for sess in range(1, 6):
        emo_transcription_dir = 'raw_data/IEMOCAP_full_release/Session{}/dialog/transcriptions/'.format(sess)
        transcription_files = [l for l in os.listdir(emo_transcription_dir) if 'Ses' in l]
        for file in transcription_files:
            with open(emo_transcription_dir + file) as f:
                lines = f.readlines()
            for line in lines:
                if not line.startswith("Ses") or "XX" in line:
                    continue
                ordered_utterance_ids.append(line.split(" ")[0])
    for sess in range(1, 6):
        emo_evaluation_dir = 'raw_data/IEMOCAP_full_release/Session{}/dialog/EmoEvaluation/'.format(sess)
        evaluation_files = [l for l in os.listdir(emo_evaluation_dir) if 'Ses' in l]
        for file in evaluation_files:
            with open(emo_evaluation_dir + file) as f:
                content = f.read()
            info_lines = re.findall(info_line, content)
            for line in info_lines[1:]:  # the first line is a header
                start_end_time, utterance_id, emotion, val_act_dom = line.strip().split('\t')
                start_time, end_time = start_end_time[1:-1].split('-')
                val, act, dom = val_act_dom[1:-1].split(',')
                val, act, dom = float(val), float(act), float(dom)
                start_time, end_time = float(start_time), float(end_time)
                start_times.append(start_time)
                end_times.append(end_time)
                utterance_ids.append(utterance_id)
                emotions.append(emotion)
                vals.append(val)
                acts.append(act)
                doms.append(dom)



    df_iemocap['start_time'] = start_times
    df_iemocap['end_time'] = end_times
    df_iemocap['utterance_id'] = utterance_ids
    df_iemocap['emotion'] = emotions
    df_iemocap['val'] = vals
    df_iemocap['act'] = acts
    df_iemocap['dom'] = doms

    df_iemocap = df_iemocap.set_index('utterance_id')
    df_iemocap = df_iemocap.loc[ordered_utterance_ids]
    df_iemocap = df_iemocap.reset_index()
    df_iemocap.to_csv('extracted_data/df_iemocap.csv', index=False)

    print("Done extracting info")
    return df_iemocap


def build_audio_vector(iemocap_info_df):
    audio_vectors = {}
    count = 0
    for sess in tqdm(range(1, 6)):
        if platform == "darwin":
            if os.path.isfile("/Users/ludus/Projects/Sentiment_analysis-OsakaU-/extracted_data/audio_vectors/audio_vectors_{}.pkl".format(sess)):
                count += 1
                continue
        elif platform == "win32":
            if os.path.isfile("C:/Projects/Sentiment_analysis-OsakaU-/extracted_data/audio_vectors/audio_vectors_{}.pkl".format(sess)):
                count += 1
                continue

        wav_file_path = 'raw_data/IEMOCAP_full_release/Session{}/dialog/wav/'.format(sess)
        orig_wav_files = os.listdir(wav_file_path)
        for orig_wav_file in tqdm(orig_wav_files):
            try:
                orig_wav_vector, _sr = librosa.load(wav_file_path + orig_wav_file, sr=SAMP_RATE)
                orig_wav_file, file_format = orig_wav_file.split('.')
                for index, row in iemocap_info_df[iemocap_info_df['utterance_id'].str.contains(orig_wav_file)].iterrows():
                    start_time, end_time, truncated_wav_file_name, emotion, val, act, dom = row['start_time'], row[
                        'end_time'], row['utterance_id'], row['emotion'], row['val'], row['act'], row['dom']
                    start_frame = math.floor(start_time * SAMP_RATE)
                    end_frame = math.floor(end_time * SAMP_RATE)
                    truncated_wav_vector = orig_wav_vector[start_frame:end_frame + 1]
                    audio_vectors[truncated_wav_file_name] = truncated_wav_vector
            except:
                print('An exception occured for {}'.format(orig_wav_file))
        with open('extracted_data/audio_vectors/audio_vectors_{}.pkl'.format(sess), 'wb') as f:
            pickle.dump(audio_vectors, f)
    if count == 5:
        print("Loaded audio_vectors")


def extract_audio_features(iemocap_info_df):
    if platform == "darwin":
        if os.path.isfile("/Users/ludus/Projects/Sentiment_analysis-OsakaU-/extracted_data/audio_features.csv"):
            print("Loaded audio features")
            return pd.read_csv("/Users/ludus/Projects/Sentiment_analysis-OsakaU-/extracted_data/audio_features.csv")
    elif platform == "win32":
        if os.path.isfile("C:/Projects/Sentiment_analysis-OsakaU-/extracted_data/audio_features.csv"):
            print("Loaded audio features")
            return pd.read_csv("C:/Projects/Sentiment_analysis-OsakaU-/extracted_data/audio_features.csv")
    emotion_dict = {'ang': 0,
                    'hap': 1,
                    'exc': 2,
                    'sad': 3,
                    'fru': 4,
                    'fea': 5,
                    'sur': 6,
                    'neu': 7,
                    'dis': 8,
                    'xxx': 9,
                    'oth': 9}

    audio_vectors_path = 'extracted_data/audio_vectors/audio_vectors_'
    columns = ['utterance_id', 'label', 'mfcc', 'cqt', 'f0']
    df_features = pd.DataFrame(columns=columns)
    for sess in range(1, 6):
        audio_vectors = pickle.load(open('{}{}.pkl'.format(audio_vectors_path, sess), 'rb'))
        for index, row in tqdm(iemocap_info_df[iemocap_info_df['utterance_id'].str.contains('Ses0{}'.format(sess))].iterrows()):
            try:
                utterance_id = row['utterance_id']
                label = emotion_dict[row['emotion']]
                y = audio_vectors[utterance_id]

                feature_list = [utterance_id, label]

                # Extract F0-Score
                f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C0'),
                                        fmax=librosa.note_to_hz('C5'))

                if np.isnan(f0).all():     # If pitch extraction fails discard utterance
                    continue

                feature_list.append(np.nanmean(f0))

                # Extract mfcc
                mfcc = librosa.feature.mfcc(y=y, sr=SAMP_RATE)
                feature_list.append(np.mean(mfcc, axis=1))

                # Extract Constant-Q chromagram
                chroma_cq = librosa.feature.chroma_cqt(y=y, sr=SAMP_RATE,  fmin=librosa.note_to_hz('C2'), bins_per_octave=24)
                feature_list.append(np.mean(chroma_cq, axis=1))

                df_features = pd.concat([df_features, pd.DataFrame(feature_list, index=columns).transpose()])

            except:
                print('Some exception occured')

    df_features.to_csv('extracted_data/audio_features.csv', index=False)

    return df_features


def extract_text_features(iemocap_info_df):
    if platform == "darwin":
        if os.path.isfile("/Users/ludus/Projects/Sentiment_analysis-OsakaU-/extracted_data/text_features.csv"):
            print("Loaded text features")
            return pd.read_csv("/Users/ludus/Projects/Sentiment_analysis-OsakaU-/extracted_data/text_features.csv")
    elif platform == "win32":
        if os.path.isfile("C:/Projects/Sentiment_analysis-OsakaU-/extracted_data/text_features.csv"):
            print("Loaded text features")
            return pd.read_csv("C:/Projects/Sentiment_analysis-OsakaU-/extracted_data/text_features.csv")

        useful_regex = re.compile(r'^(\w+)', re.IGNORECASE)
        columns = ['utterance_id', 'label', 'text', 'b_features']
        df_features = pd.DataFrame(columns=columns)
        model = SentenceTransformer('all-mpnet-base-v2')  # Load s-bert model for text feature extractions
        for sess in tqdm(range(1, 6)):
            transcripts_path = 'raw_data/IEMOCAP_full_release/Session{}/dialog/transcriptions/'.format(sess)
            transcript_files = os.listdir(transcripts_path)
            for f in tqdm(transcript_files):
                with open('{}{}'.format(transcripts_path, f), 'r') as f:
                    all_lines = f.readlines()
                for l in all_lines:
                    transcript_code = useful_regex.match(l).group()
                    row_info = iemocap_info_df.loc[iemocap_info_df['utterance_id'] == transcript_code]

                    if not row_info.empty:
                        transcription = l.split(':')[-1].strip()
                        new_row = pd.Series({columns[0]: transcript_code,  columns[1]: row_info.loc[row_info.index[0], "emotion"],
                                             columns[2]: transcription, columns[3]: model.encode(transcription)})
                        df_features = pd.concat([df_features, new_row.to_frame().T], ignore_index=True)

        df_features.to_csv('extracted_data/text_features.csv', index=False)

        return df_features

def extract_ESD_audio_features():
    emo_dict = {
        "Angry": 0,
        "Happy": 1,
        "Neutral": 2,
        "Sad": 3,
        "Surprise": 4
    }
    columns = ['set', 'label', 'f0', 'mfcc', 'cqt']
    df_features = pd.DataFrame(columns=columns)
    start_sess = 11

    if platform == "darwin":
        if os.path.isfile("/extracted_data/ESD/male_features/ESD_audio_features_11.csv"):
            print("Loaded ESD audio features")
            df_features = pd.read_csv("extracted_data/ESD/ESD_audio_features_combined.csv")
            return df_features
        elif os.path.isfile("/extracted_data/ESD/male_features/ESD_audio_features_11.csv"):
            feature_dir = "Users/ludus/Projects/Sentiment_analysis-OsakaU-/extracted_data/ESD/"
            features_files = os.listdir(feature_dir)
            for features_file in features_files:
                df_features = pd.concat([df_features, pd.read_csv(feature_dir + features_file)])
                start_sess += 1
            if start_sess == 21:
                print("Loaded ESD audio features")
                return df_features
    elif platform == "win32":
        if os.path.isfile("C:/Projects/Sentiment_analysis-OsakaU-/extracted_data/ESD/ESD_audio_features_combined.csv"):
            print("Loaded ESD audio features")
            df_features = pd.read_csv("extracted_data/ESD/ESD_audio_features_combined.csv")
            return df_features
        elif os.path.isfile("C:/Projects/Sentiment_analysis-OsakaU-/extracted_data/ESD/ESD_audio_features_11.csv"):
            feature_dir = "C:/Projects/Sentiment_analysis-OsakaU-/extracted_data/ESD/"
            features_files = os.listdir(feature_dir)
            for features_file in features_files:
                df_features = pd.concat([df_features, pd.read_csv(feature_dir + features_file)])
                start_sess += 1
            if start_sess == 21:
                print("Loaded ESD audio features")
                return df_features

    for sess in tqdm(range(start_sess, 21)):
        df_features = pd.DataFrame(columns=columns)
        wav_file_dirs = []
        wav_ang_val = 'raw_data/Emotional Speech Dataset (ESD)/00{}/Angry/evaluation'.format(sess)
        wav_ang_test = 'raw_data/Emotional Speech Dataset (ESD)/00{}/Angry/test'.format(sess)
        wav_ang_train = 'raw_data/Emotional Speech Dataset (ESD)/00{}/Angry/train'.format(sess)
        wav_hap_val = 'raw_data/Emotional Speech Dataset (ESD)/00{}/Happy/evaluation'.format(sess)
        wav_hap_test = 'raw_data/Emotional Speech Dataset (ESD)/00{}/Happy/test'.format(sess)
        wav_hap_train = 'raw_data/Emotional Speech Dataset (ESD)/00{}/Happy/train'.format(sess)
        wav_neu_val = 'raw_data/Emotional Speech Dataset (ESD)/00{}/Neutral/evaluation'.format(sess)
        wav_neu_test = 'raw_data/Emotional Speech Dataset (ESD)/00{}/Neutral/test'.format(sess)
        wav_neu_train = 'raw_data/Emotional Speech Dataset (ESD)/00{}/Neutral/train'.format(sess)
        wav_sad_val = 'raw_data/Emotional Speech Dataset (ESD)/00{}/Sad/evaluation'.format(sess)
        wav_sad_test = 'raw_data/Emotional Speech Dataset (ESD)/00{}/Sad/test'.format(sess)
        wav_sad_train = 'raw_data/Emotional Speech Dataset (ESD)/00{}/Sad/train'.format(sess)
        wav_sur_val = 'raw_data/Emotional Speech Dataset (ESD)/00{}/Surprise/evaluation'.format(sess)
        wav_sur_test = 'raw_data/Emotional Speech Dataset (ESD)/00{}/Surprise/test'.format(sess)
        wav_sur_train = 'raw_data/Emotional Speech Dataset (ESD)/00{}/Surprise/train'.format(sess)
        wav_file_dirs.append(wav_ang_val)
        wav_file_dirs.append(wav_ang_test)
        wav_file_dirs.append(wav_ang_train)
        wav_file_dirs.append(wav_hap_val)
        wav_file_dirs.append(wav_hap_test)
        wav_file_dirs.append(wav_hap_train)
        wav_file_dirs.append(wav_neu_val)
        wav_file_dirs.append(wav_neu_test)
        wav_file_dirs.append(wav_neu_train)
        wav_file_dirs.append(wav_sad_val)
        wav_file_dirs.append(wav_sad_test)
        wav_file_dirs.append(wav_sad_train)
        wav_file_dirs.append(wav_sur_val)
        wav_file_dirs.append(wav_sur_test)
        wav_file_dirs.append(wav_sur_train)

        for wav_file_dir in wav_file_dirs:
            data_set = "val" if wav_file_dir.split("/")[-1] == "evaluation" else wav_file_dir.split("/")[-1]
            label = emo_dict[os.path.dirname(wav_file_dir).split("/")[-1]]
            wav_files = os.listdir(wav_file_dir)
            for wav_file in tqdm(wav_files):
                wav_vec, _sr = librosa.load(wav_file_dir + "/" + wav_file, sr=SAMP_RATE)
                feature_list = [data_set, label]

                # Extract F0-Score
                f0, _, _ = librosa.pyin(wav_vec, fmin=librosa.note_to_hz('C0'),
                                        fmax=librosa.note_to_hz('C5'))

                if np.isnan(f0).all():  # If pitch extraction fails discard utterance
                    continue

                feature_list.append(np.nanmean(f0))

                # Extract mfcc
                mfcc = librosa.feature.mfcc(y=wav_vec, sr=SAMP_RATE)
                feature_list.append(np.mean(mfcc, axis=1))

                # Extract Constant-Q chromagram
                chroma_cq = librosa.feature.chroma_cqt(y=wav_vec, sr=SAMP_RATE, fmin=librosa.note_to_hz('C2'),
                                                       bins_per_octave=24)
                feature_list.append(np.mean(chroma_cq, axis=1))

                df_features = pd.concat([df_features, pd.DataFrame(feature_list, index=columns).transpose()])

        df_features.to_csv('extracted_data/ESD/ESD_audio_features_{}.csv'.format(sess), index=False)

    df_features = pd.DataFrame(columns=columns)
    if platform == "darwin":
        feature_dir = "Users/ludus/Projects/Sentiment_analysis-OsakaU-/extracted_data/ESD/"
        features_files = os.listdir(feature_dir)
        for features_file in features_files:
            df_features = pd.concat([df_features, pd.read_csv(feature_dir + features_file)])
        df_features.to_csv('extracted_data/ESD/ESD_audio_features_combined.csv', index=False)
    elif platform == "win32":
        feature_dir = "C:/Projects/Sentiment_analysis-OsakaU-/extracted_data/ESD/"
        features_files = os.listdir(feature_dir)
        for features_file in features_files:
            df_features = pd.concat([df_features, pd.read_csv(feature_dir + features_file)])
        df_features.to_csv('extracted_data/ESD/ESD_audio_features_combined.csv', index=False)

    return df_features


def extract_twitter_text_features():
    model = SentenceTransformer('all-mpnet-base-v2')  # Load s-bert model for text feature extractions
    twitter_df = pd.read_csv("raw_data/twitter/text_emotion.csv")
    twitter_df = twitter_df.drop(["tweet_id", "author"], axis=1)
    twitter_df["content"] = twitter_df["content"].apply(lambda x:  model.encode(x))
    twitter_df.to_csv("extracted_data/twitter/twitter_text_features.csv")
    return twitter_df


def main():
    iemocap_df = extract_iemocap_info()
    print(iemocap_df.head())
    print("Number of samples extracted: " + str(len(iemocap_df.index)))
    build_audio_vector(iemocap_df)
    audio_features = extract_audio_features(iemocap_df)
    text_features = extract_text_features(iemocap_df)
    esd_audio_features = extract_ESD_audio_features()
    twitter_features = extract_twitter_text_features()
    print("Audio features:\n" + str(audio_features.head()))
    print("Text features:\n" + str(text_features.head()))
    print("ESD audio features:\n" + str(esd_audio_features.head()))
    print("Twitter text features:\n" + str(twitter_features.head()))


if __name__ == '__main__':
    print("Starting")
    main()
