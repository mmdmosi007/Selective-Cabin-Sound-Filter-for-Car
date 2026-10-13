import os
import tensorflow_hub as hub
import librosa
import numpy as np
import csv, urllib.request


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


print("Loading YAMNet...")
model = hub.load('https://tfhub.dev/google/yamnet/1')



labels = []
with urllib.request.urlopen('https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv') as f:
    reader = csv.DictReader(f.read().decode('utf-8').splitlines())
    for row in reader:
        labels.append(row['display_name'])



def predict_file(path):
    wav, sr = librosa.load(path, sr=16000, mono=True)
    # حذف سکوت اول و آخر
    wav, _ = librosa.effects.trim(wav, top_db=25)
    if len(wav) < 16000:
        wav = np.pad(wav, (0, 16000 - len(wav)))
    scores, _, _ = model(wav)
    # مهم: با max حساب می‌کنیم نه mean
    top = np.max(scores.numpy(), axis=0)
    idx = np.argsort(top)[-5:][::-1]
    print(f"\n--- {path} ---")
    for i in idx:
        print(f"{labels[i]}: {top[i]:.2f}")
    top5 = " ".join([labels[i].lower() for i in idx])
    if 'horn' in top5 or 'siren' in top5 or 'toot' in top5 or 'vehicle' in top5:
        print(">>> RESULT: HORN/SIREN - عبور بده")
    elif 'speech' in top5 or 'conversation' in top5 or 'narration' in top5:
        print(">>> RESULT: SPEECH - عبور بده")
    else:
        print(">>> RESULT: NOISE - حذف کن")


predict_file('car_horn.wav')
predict_file('car_horn2.wav')
predict_file('human_speech.wav')