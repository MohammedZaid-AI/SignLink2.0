import av
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
import mediapipe as mp

from pathlib import Path
from collections import deque
from streamlit_webrtc import webrtc_streamer, WebRtcMode


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "model.tflite"
LABELS_PATH = BASE_DIR / "labels.txt"

INPUT_SIZE = (224, 224)
HISTORY_SIZE = 8


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SignLink - Sign to Speech",
    page_icon="👋",
    layout="wide",
)


# ============================================================
# LOAD LABELS
# ============================================================

with open(LABELS_PATH, "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f.readlines() if line.strip()]


# ============================================================
# LOAD TFLITE MODEL
# ============================================================

@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(
        model_path=str(MODEL_PATH)
    )
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    return interpreter, input_details, output_details


interpreter, input_details, output_details = load_model()


# ============================================================
# MEDIAPIPE
# ============================================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess(img):
    img = cv2.resize(img, INPUT_SIZE)

    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    img = img.astype(np.float32) / 255.0

    return np.expand_dims(img, axis=0)


# ============================================================
# HAND CROP
# ============================================================

def crop_hand(frame, hand_landmarks):

    h, w, _ = frame.shape

    xs = [
        int(point.x * w)
        for point in hand_landmarks.landmark
    ]

    ys = [
        int(point.y * h)
        for point in hand_landmarks.landmark
    ]

    padding = 40

    x1 = max(0, min(xs) - padding)
    y1 = max(0, min(ys) - padding)

    x2 = min(w, max(xs) + padding)
    y2 = min(h, max(ys) + padding)

    crop = frame[y1:y2, x1:x2]

    return crop, (x1, y1, x2, y2)


# ============================================================
# VIDEO PROCESSOR
# ============================================================

class SignLinkProcessor:

    def __init__(self):

        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.history = deque(
            maxlen=HISTORY_SIZE
        )

        self.label = "No Hand"
        self.confidence = 0.0

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        # Mirror camera
        img = cv2.flip(img, 1)

        rgb = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        results = self.hands.process(rgb)

        self.label = "No Hand"
        self.confidence = 0.0

        if results.multi_hand_landmarks:

            hand = results.multi_hand_landmarks[0]

            crop, box = crop_hand(
                img,
                hand
            )

            if crop.size > 0:

                processed = preprocess(crop)

                # TensorFlow Lite prediction
                interpreter.set_tensor(
                    input_details[0]["index"],
                    processed
                )

                interpreter.invoke()

                prediction = interpreter.get_tensor(
                    output_details[0]["index"]
                )[0]

                index = int(
                    np.argmax(prediction)
                )

                confidence = float(
                    prediction[index]
                )

                predicted_label = class_names[index]

                # Store prediction
                if confidence >= 0.50:

                    self.history.append(
                        predicted_label
                    )

                else:

                    self.history.append(
                        "Uncertain"
                    )

                # Majority voting
                if self.history:

                    self.label = max(
                        set(self.history),
                        key=self.history.count
                    )

                self.confidence = confidence

            # Draw MediaPipe landmarks
            mp_draw.draw_landmarks(
                img,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

            # Draw bounding box
            x1, y1, x2, y2 = box

            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                (255, 255, 255),
                2
            )

        # Prediction text
        cv2.putText(
            img,
            f"{self.label} ({self.confidence * 100:.1f}%)",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )


# ============================================================
# UI
# ============================================================

st.title("SignLink 👋")

st.subheader(
    "Sign-to-Speech"
)

st.write(
    "Show one of the supported signs to the camera."
)

st.info(
    "Supported signs: Hello • I Love You • Yes • Flower"
)


# ============================================================
# CAMERA
# ============================================================

ctx = webrtc_streamer(
    key="signlink-camera",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=SignLinkProcessor,
    rtc_configuration={
        "iceServers": [
            {
                "urls": ["stun:stun.l.google.com:19302"]
            }
        ]
    },
    media_stream_constraints={
        "video": True,
        "audio": False,
    },
    async_processing=True,
)


# ============================================================
# INFORMATION
# ============================================================

st.divider()

st.subheader("How to use")

st.markdown(
    """
    1. Click **START** above.
    2. Allow camera access when your browser asks.
    3. Place your hand clearly inside the camera view.
    4. Hold the sign steady for a moment.
    5. The predicted sign will appear on the video.
    """
)

st.caption(
    "SignLink uses MediaPipe for hand detection and "
    "a TensorFlow Lite model for sign classification."
)