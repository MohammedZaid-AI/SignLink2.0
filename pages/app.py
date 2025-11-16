import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque

# =====================
# CONFIG
# =====================
MODEL_PATH = "model.tflite"   # rename your file to this
LABELS_PATH = "labels.txt"    # rename "labels" to labels.txt
INPUT_SIZE = (224, 224)
HISTORY = 8

# =====================
# LOAD LABELS
# =====================
with open(LABELS_PATH, "r") as f:
    class_names = [c.strip() for c in f.readlines()]

# =====================
# LOAD TFLITE MODEL
# =====================
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# =====================
# MEDIAPIPE
# =====================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

def preprocess(img):
    img = cv2.resize(img, INPUT_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0   # TM normalization
    img = np.expand_dims(img, axis=0)
    return img

def predict(img):
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0]

def crop_hand(frame, lm):
    h, w, _ = frame.shape
    x = [p.x * w for p in lm.landmark]
    y = [p.y * h for p in lm.landmark]
    x1 = int(min(x)) - 40
    x2 = int(max(x)) + 40
    y1 = int(min(y)) - 40
    y2 = int(max(y)) + 40
    return frame[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]

# =====================
# STREAMLIT UI
# =====================
st.title("🤟 Sign Language Detector (Teachable Machine TFLite)")
threshold = st.sidebar.slider("Confidence %", 10, 100, 50) / 100

frame_placeholder = st.empty()
text_placeholder = st.markdown("## Prediction: Waiting...")

history = deque(maxlen=HISTORY)
cap = cv2.VideoCapture(0)
stop = st.button("Stop")

while cap.isOpened() and not stop:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    final_label = "No Hand"

    if results.multi_hand_landmarks:
        crop = crop_hand(frame, results.multi_hand_landmarks[0])
        if crop.size > 0:
            img = preprocess(crop)
            pred = predict(img)
            idx = np.argmax(pred)
            conf = pred[idx]

            if conf > threshold:
                history.append(class_names[idx])  # no need to strip prefix
            else:
                history.append("Uncertain")

        if history:
            final_label = max(set(history), key=history.count)

    frame_placeholder.image(frame, channels="BGR")
    text_placeholder.markdown(f"### Prediction: **{final_label}**")

cap.release()
cv2.destroyAllWindows()
