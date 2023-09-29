# SDR-Audio-Analyzer
SDR Audio Analyzer is an innovative tool for comparing audio signals. It allows you to take any sample from sigidwiki.com and compare it with your own recordings to get approximate results. This tool is currently under development and promises to be a game-changer in the field of Software Defined Radio (SDR) analysis.

# How It Works
The tool works by extracting MFCCs (Mel-frequency cepstral coefficients) from each audio file using the librosa library. It then uses Dynamic Time Warping (implemented by the fastdtw library) to compute a distance between the MFCCs of the target audio file and each of the other audio files in the directory.
The distances are normalized to a range between 0 and 100, with 100 being a perfect match. The tool then outputs the top 5 matches based on these distances.
You can manually add as many samples as you wish in the directory for comparison. The more samples you have, the more comprehensive your results will be

# Usage
```
python signal_analyser.py target_file.mp3 directory
```
Please replace target_file.mp3 and directory with your actual file and directory paths. The script will output the top 5 audio files from the directory that are most similar to the target file.

# Installation
```
1. git clone https://github.com/sp0ttybug/sdr_audio_analyzer.git.
2. cd sdr_audio_analyzer.
3. pip install -r requirements.txt
```
