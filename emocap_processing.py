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

SAMP_RATE = 22050


def extract_iemocap_info():
    if os.path.isfile("/Users/ludus/Projects/Sentiment_analysis-OsakaU-/extracted_data/df_iemocap.csv"):
        print("Loaded iemocap_info")
        return pd.read_csv("/Users/ludus/Projects/Sentiment_analysis-OsakaU-/extracted_data/df_iemocap.csv")
    else:
        info_line = re.compile(r'\[.+\]\n', re.IGNORECASE)

        start_times, end_times, wav_file_names, emotions, vals, acts, doms = [], [], [], [], [], [], []

        print("Extraction info from IEMOCAP...")
        for sess in range(1, 6):
            emo_evaluation_dir = 'raw_data/IEMOCAP_full_release/Session{}/dialog/EmoEvaluation/'.format(sess)
            evaluation_files = [l for l in os.listdir(emo_evaluation_dir) if 'Ses' in l]
            for file in evaluation_files:
                with open(emo_evaluation_dir + file) as f:
                    content = f.read()
                info_lines = re.findall(info_line, content)
                for line in info_lines[1:]:  # the first line is a header
                    start_end_time, wav_file_name, emotion, val_act_dom = line.strip().split('\t')
                    start_time, end_time = start_end_time[1:-1].split('-')
                    val, act, dom = val_act_dom[1:-1].split(',')
                    val, act, dom = float(val), float(act), float(dom)
                    start_time, end_time = float(start_time), float(end_time)
                    start_times.append(start_time)
                    end_times.append(end_time)
                    wav_file_names.append(wav_file_name)
                    emotions.append(emotion)
                    vals.append(val)
                    acts.append(act)
                    doms.append(dom)

        df_iemocap = pd.DataFrame(columns=['start_time', 'end_time', 'wav_file', 'emotion', 'val', 'act', 'dom'])

        df_iemocap['start_time'] = start_times
        df_iemocap['end_time'] = end_times
        df_iemocap['wav_file'] = wav_file_names
        df_iemocap['emotion'] = emotions
        df_iemocap['val'] = vals
        df_iemocap['act'] = acts
        df_iemocap['dom'] = doms

        df_iemocap.to_csv('extracted_data/df_iemocap.csv', index=False)

        print("Done extracting info")
        return df_iemocap


def build_audio_vector():
    iemocap_info_df = pd.read_csv('extracted_data/df_iemocap.csv')
    audio_vectors = {}
    for sess in range(1, 6):
        if os.path.isfile("/Users/ludus/Projects/Sentiment_analysis-OsakaU-/extracted_data/audio_vectors/audio_vectors_{}.pkl".format(sess)):
            continue
        else:
            wav_file_path = 'raw_data/IEMOCAP_full_release/Session{}/dialog/wav/'.format(sess)
            orig_wav_files = os.listdir(wav_file_path)
            for orig_wav_file in tqdm(orig_wav_files):
                try:
                    orig_wav_vector, _sr = librosa.load(wav_file_path + orig_wav_file, sr=SAMP_RATE)
                    orig_wav_file, file_format = orig_wav_file.split('.')
                    for index, row in iemocap_info_df[iemocap_info_df['wav_file'].str.contains(orig_wav_file)].iterrows():
                        start_time, end_time, truncated_wav_file_name, emotion, val, act, dom = row['start_time'], row[
                            'end_time'], row['wav_file'], row['emotion'], row['val'], row['act'], row['dom']
                        start_frame = math.floor(start_time * SAMP_RATE)
                        end_frame = math.floor(end_time * SAMP_RATE)
                        truncated_wav_vector = orig_wav_vector[start_frame:end_frame + 1]
                        audio_vectors[truncated_wav_file_name] = truncated_wav_vector
                except:
                    print('An exception occured for {}'.format(orig_wav_file))
            with open('extracted_data/audio_vectors/audio_vectors_{}.pkl'.format(sess), 'wb') as f:
                pickle.dump(audio_vectors, f)


def extract_audio_features():
    if os.path.isfile("/Users/ludus/Projects/Sentiment_analysis-OsakaU-/extracted_data/audio_features.csv"):
        return pd.read_csv("/Users/ludus/Projects/Sentiment_analysis-OsakaU-/extracted_data/audio_features.csv")
    else:
        iemocap_info_df = pd.read_csv('extracted_data/df_iemocap.csv')
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
        columns = ['wav_file', 'label', 'mfcc', 'cqt', 'f0']
        df_features = pd.DataFrame(columns=columns)
        for sess in (range(1, 6)):
            audio_vectors = pickle.load(open('{}{}.pkl'.format(audio_vectors_path, sess), 'rb'))
            for index, row in tqdm(iemocap_info_df[iemocap_info_df['wav_file'].str.contains('Ses0{}'.format(sess))].iterrows()):
                try:
                    wav_file_name = row['wav_file']
                    label = emotion_dict[row['emotion']]
                    y = audio_vectors[wav_file_name]


                    #time.sleep(1000)

                    feature_list = [wav_file_name, label]  # wav_file, label

                    # Extract mfcc
                    mfcc = librosa.feature.mfcc(y=y, sr=SAMP_RATE)
                    feature_list.append(np.mean(mfcc, axis=1))


                    # Extract Constant-Q chromagram
                    chroma_cq = librosa.feature.chroma_cqt(y=y, sr=SAMP_RATE,  fmin=librosa.note_to_hz('C2'), bins_per_octave=24)
                    feature_list.append(np.mean(chroma_cq, axis=1))

                    # Extract F0-Score
                    f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C0'),
                                            fmax=librosa.note_to_hz('C5'))

                    feature_list.append(np.nanmean(f0))

                    df_features = pd.concat([pd.DataFrame(feature_list, index=columns).transpose(), df_features])

                except:
                    print('Some exception occured')

        df_features.to_csv('extracted_data/audio_features.csv', index=False)

        return df_features


def extract_text_features():
    useful_regex = re.compile(r'^(\w+)', re.IGNORECASE)

    file2transcriptions = {}

    for sess in range(1, 6):
        transcripts_path = 'raw_data/IEMOCAP_full_release/Session{}/dialog/transcriptions/'.format(sess)
        transcript_files = os.listdir(transcripts_path)
        for f in transcript_files:
            with open('{}{}'.format(transcripts_path, f), 'r') as f:
                all_lines = f.readlines()

            for l in all_lines:
                audio_code = useful_regex.match(l).group()
                transcription = l.split(':')[-1].strip()
                # assuming that all the keys would be unique and hence no `try`
                file2transcriptions[audio_code] = transcription
    # save dict
    with open('data/t2e/audiocode2text.pkl', 'wb') as file:
        pickle.dump(file2transcriptions, file)

def main():
    iemocap_df = extract_iemocap_info()
    print(iemocap_df.tail())
    print("Number of samples extracted: " + str(len(iemocap_df.index)))
    build_audio_vector()
    audio_features = extract_audio_features()
    print(audio_features.tail())
    #extract_text_features()

if __name__ == '__main__':
    print("Starting")
    main()
