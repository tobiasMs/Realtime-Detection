# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 16:54:22 2025

@author: tobias.sulistiyo
"""

import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import time
import platform
import os

# === Label & Transform ===
LABELS = ["non_drowsy", "drowsy"]
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

val_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean,std)
])

# === Load Model ===
def load_model(weights="runs/mobilenetv2/best.pt", device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    m = models.mobilenet_v2(weights=None)
    m.classifier[1] = nn.Linear(m.last_channel, 2)
    ckpt = torch.load(weights, map_location=device)
    m.load_state_dict(ckpt["model"])
    m.eval().to(device)
    return m, device

# === Preprocess Face ===
def preprocess_face(frame, box):
    x,y,w,h = box
    crop = frame[y:y+h, x:x+w]
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    rgb_resized = cv2.resize(rgb, (224,224))
    tensor = val_tf(rgb_resized).unsqueeze(0)
    return tensor

# === Buzzer function ===
def buzzer():
    system = platform.system()
    if system == "Windows":
        import winsound
        winsound.Beep(1000, 1000)  # freq=1000Hz, dur=1s
    else:
        # Linux/Mac
        os.system('play -nq -t alsa synth 1 sine 1000')

# === Main realtime loop ===
def main(weights="runs/mobilenetv2/best.pt", cam=0):
    model, device = load_model(weights)
    cap = cv2.VideoCapture(cam)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    drowsy_start = None
    buzzer_triggered = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2,
                                              minNeighbors=5, minSize=(80,80))

        label, conf = "non_drowsy", 0.0
        if len(faces) > 0:
            faces = sorted(faces, key=lambda b: b[2]*b[3], reverse=True)
            (x,y,w,h) = faces[0]

            inp = preprocess_face(frame, (x,y,w,h)).to(device)
            with torch.inference_mode():
                logits = model(inp)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            pred_idx = int(np.argmax(probs))
            label = LABELS[pred_idx]
            conf  = probs[pred_idx]

            color = (0,255,0) if label=="non_drowsy" else (0,0,255)
            cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # === Drowsy detection logic ===
        if label == "drowsy" and conf > 0.7:
            if drowsy_start is None:
                drowsy_start = time.time()
            elapsed = time.time() - drowsy_start

            # sudah ngantuk ≥ 5 detik
            if elapsed >= 5 and not buzzer_triggered:
                print("[INFO] Kantuk terdeteksi, buzzer akan bunyi 3 detik lagi...")
                time.sleep(3)  # tunggu 3 detik
                buzzer()
                buzzer_triggered = True
        else:
            drowsy_start = None
            buzzer_triggered = False

        cv2.imshow("Drowsy Detection + Buzzer", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
