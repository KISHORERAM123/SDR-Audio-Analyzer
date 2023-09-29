import os
import sys
import numpy as np
import librosa
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from tqdm import tqdm
RADIO_ART = """
                    _ _                           _                     
     /\            | (_)        /\               | |                    
    /  \  _   _  __| |_  ___   /  \   _ __   __ _| |_   _ ___  ___ _ __ 
   / /\ \| | | |/ _` | |/ _ \ / /\ \ | '_ \ / _` | | | | / __|/ _ \ '__|
  / ____ \ |_| | (_| | | (_) / ____ \| | | | (_| | | |_| \__ \  __/ |   
 /_/    \_\__,_|\__,_|_|\___/_/    \_\_| |_|\__,_|_|\__, |___/\___|_|   
                                                     __/ |              
                                                    |___/               
"""
print(RADIO_ART)
print("Author: sp0ttybug\n")
print("Please ensure that the target audio file is in the same directory as the tool")
def compare_audios(target_file, directory):
    target_audio, sr = librosa.load(target_file)
    target_mfcc = librosa.feature.mfcc(y=target_audio, sr=sr)

    scores = {}

    for audio_file in tqdm(os.listdir(directory)):  # wrap os.listdir(directory) with tqdm for progress bar
        if not audio_file.lower().endswith(('.wav', '.mp3', '.ogg', '.flac', '.au', '.aiff')):  # skip non-audio files
            continue

        try:
            audio_path = os.path.join(directory, audio_file)
            audio, _ = librosa.load(audio_path)
            mfcc = librosa.feature.mfcc(y=audio, sr=sr)

            # If the MFCCs have different lengths, pad the shorter one with zeros
            if target_mfcc.shape[1] > mfcc.shape[1]:
                mfcc = np.pad(mfcc, ((0, 0), (0, target_mfcc.shape[1] - mfcc.shape[1])), mode='constant')
            elif mfcc.shape[1] > target_mfcc.shape[1]:
                target_mfcc = np.pad(target_mfcc, ((0, 0), (0, mfcc.shape[1] - target_mfcc.shape[1])), mode='constant')

            # Use Dynamic Time Warping to compare MFCCs
            distance, path = fastdtw(target_mfcc.T, mfcc.T, dist=euclidean)
            scores[audio_file] = distance
        except Exception as e:
            print(f"Could not process file {audio_file}: {e}")

    # Normalize scores to a range between 0 and 100
    max_score = max(scores.values())
    min_score = min(scores.values())
    for audio_file in scores:
        scores[audio_file] = 100 - ((scores[audio_file] - min_score) / (max_score - min_score) * 100)

    # Sort the scores in descending order and get the top 5 matches
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:5]

    return sorted_scores

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python signal_analyser.py <target_file> <directory>")
        sys.exit(1)

    target_file = sys.argv[1]
    directory = sys.argv[2]

    top_matches = compare_audios(target_file, directory)

    for audio_file, match_percentage in top_matches:
        print(f'Audio file: {audio_file}, Match percentage: {match_percentage}%')
