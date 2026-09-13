# -*- coding: utf-8 -*-
import pyaudio
import wave
import keyboard as kb
import librosa
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# Add more imports if required
# --- workaround for missing TypeIs in typing_extensions 4.8.0 ---
import typing_extensions
if not hasattr(typing_extensions, 'TypeIs'):
    from typing_extensions import TypeGuard
    typing_extensions.TypeIs = TypeGuard
# ------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

## Define your Architecture
# NOTE: this MUST exactly match the architecture used when you trained and
# saved speech_model.t7 — the layer names/shapes have to line up for
# load_state_dict() to work.
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels=900, out_channels=400, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(400)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(1)
        self.dropout = nn.Dropout(p=0.25)

        self.conv2 = nn.Conv1d(in_channels=400, out_channels=200, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(200)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool1d(1)

        self.conv3 = nn.Conv1d(in_channels=200, out_channels=100, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(100)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool1d(1)

        self.fc1 = nn.Linear(100, 6)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.maxpool1(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)
        out = self.dropout(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.maxpool3(out)
        out = self.dropout(out)

        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.logsoftmax(out)
        return out

# get_features() function is used to return the audio features
def get_features(filepath, sr=8000, n_mfcc=30, n_mels=128, frames=15):
    y, sr = librosa.load(filepath, sr=sr)
    D = np.abs(librosa.stft(y)) ** 2
    S = librosa.feature.melspectrogram(S=D)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    log_S = librosa.power_to_db(S, ref=np.max)
    features = librosa.feature.mfcc(S=log_S, n_mfcc=n_mfcc)
    if features.shape[1] < frames:
        features = np.hstack((features, np.zeros((n_mfcc, frames - features.shape[1]))))
    elif features.shape[1] > frames:
        features = features[:, :frames]

    # Find 1st order delta_mfcc
    delta1_mfcc = librosa.feature.delta(features, order=1)

    # Find 2nd order delta_mfcc
    delta2_mfcc = librosa.feature.delta(features, order=2)

    features = np.hstack((delta1_mfcc.flatten(), delta2_mfcc.flatten()))
    features = features.flatten()[:, np.newaxis]
    features = Variable(torch.from_numpy(features)).float()
    return features

def wait_for_key():
    while True:
        try:
            if kb.is_pressed('s'):
                return
            else:
                pass
        except:
            continue

# Function to record the voice sample, total recording time is 1 sec
# Username is the identifier for the person recording the voice
# j is the label for the sample For Example : if you recording the sample for "one" label is 1, for "two" it is 2 etc.
# v is the unique identifier for each sample recorded by a person
# Example username is r1 , j is 1 (label), v is 10 (10th sample recorded by that person) audio file will be saved with the name 1_r1_10.wav
# returns the filepath after recording
def record_voice(Username, j, v, dir):
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 8000
    CHUNK = 1024
    RECORD_SECONDS = 1
    WAVE_OUTPUT_FILENAME = "file.wav"
    audio = pyaudio.PyAudio()

    # start Recording
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("recording...")
    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("finished recording")

    # stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate()
    WAVE_OUTPUT_FILENAME = str(j) + "_" + Username + "_" + str(v) + ".wav"
    waveFile = wave.open(dir + WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()
    return dir + WAVE_OUTPUT_FILENAME


##Given audio file path, this plays that wav file
def play_audio(path):
    CHUNK = 1024
    wf = wave.open(path, 'rb')
    # instantiate PyAudio (1)
    p = pyaudio.PyAudio()
    # open stream (2)
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)
    # read data
    data = wf.readframes(CHUNK)
    # play stream (3)
    while len(data) > 0:
        stream.write(data)
        data = wf.readframes(CHUNK)
    # stop stream (4)
    stream.stop_stream()
    stream.close()
    # close PyAudio (5)
    p.terminate()