# Selective-Cabin-Sound-Filter-for-Car

Selective Active Noise Cancellation + AI Sound Transparency for vehicles.

## What it does
- Listens to car cabin mic in realtime
- Detects horn/siren/speech with YAMNet
- Phase 1 results on PC: 94% horn, 100% speech
- Filters: PASS horn/speech, BLOCK noise

## Stack
Python, YAMNet, librosa, sounddevice

## Run
pip install -r requirements.txt
py live_mic.py

## Roadmap
- v1: PC realtime detection (done)
- v2: User select buttons [human / horn / nature]
- v3: Raspberry Pi + ESP32-S3 box

Target roles: Volvo Cars / NXP Edge AI - Sweden / Netherlands
