import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow_hub as hub
import sounddevice as sd
import numpy as np
import csv, urllib.request

print("Loading YAMNet...")
model = hub.load('https://tfhub.dev/google/yamnet/1')

labels = []
with urllib.request.urlopen('https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv') as f:
    reader = csv.DictReader(f.read().decode('utf-8').splitlines())
    for row in reader:
        labels.append(row['display_name'])

print("Mic ready - حرف بزن یا صدای بوق از گوشیت پخش کن (خروج با Ctrl+C)")

def callback(indata, frames, time, status):
    wav = indata[:, 0].astype(np.float32)
    scores, _, _ = model(wav)
    top = np.max(scores.numpy(), axis=0)
    idx = np.argmax(top)
    name = labels[idx]
    print(f"{name}: {top[idx]:.2f}")

# ۱ ثانیه، ۱۶ کیلوهرتز
with sd.InputStream(channels=1, samplerate=16000, blocksize=16000, callback=callback):
    while True:
        sd.sleep(1000)