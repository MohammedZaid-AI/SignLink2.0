import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import json
import mediapipe as mp
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from collections import deque

# --- 1. Configuration and Model Loading ---
MODEL_PATH = 'best_model.h5'
LABELS_PATH = 'labels.json'
MODEL_INPUT_SIZE = (128, 128)
PREDICTION_HISTORY_LEN = 10 # Frames to average over for stable prediction

# Load the TFLite model and labels (cached for performance)
@st.cache_resource
def load_my_model():
    """Loads the TFLite model."""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

@st.cache_data
def load_my_labels():
    """Loads the class labels."""
    try:
        with open(LABELS_PATH, 'r') as f:
            labels = json.load(f)
        print(f"Loaded {len(labels)} labels.")
        return labels
    except Exception as e:
        st.error(f"Error loading labels: {e}")
        st.stop()

model = load_my_model()
labels = load_my_labels()

# --- 2. Mediapipe Hand Detection Setup ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,                # We only want to detect one hand
    min_detection_confidence=0.7,   # Higher confidence
    min_tracking_confidence=0.5
)

# --- 3. Helper Functions ---
def get_hand_bounding_box(hand_landmarks, image_shape, padding_pixels):
    """Calculates the bounding box with fixed pixel padding."""
    h, w, _ = image_shape
    x_coords = [lm.x * w for lm in hand_landmarks.landmark]
    y_coords = [lm.y * h for lm in hand_landmarks.landmark]
    
    x_min = max(0, int(min(x_coords)) - padding_pixels)
    x_max = min(w, int(max(x_coords)) + padding_pixels)
    y_min = max(0, int(min(y_coords)) - padding_pixels)
    y_max = min(h, int(max(y_coords)) + padding_pixels)
    
    return [x_min, y_min, x_max, y_max]

def preprocess_for_model(img_crop, input_size):
    """
    Prepares the cropped image for the MobileNetV2 model:
    1. Resizes (stretching)
    2. Expands dimensions
    3. Applies MobileNet preprocessing
    """
    img_resized = cv2.resize(img_crop, input_size)
    img_array = np.asarray(img_resized)
    img_expanded = np.expand_dims(img_array, axis=0).astype(float)
    return preprocess_input(img_expanded)

# --- 4. Streamlit App UI ---
st.title("Sign Language Detector (MobileNetV2) 🚀")
st.write("Hold a sign up to your webcam and see the model in action.")

# --- Tunable Parameters ---
st.sidebar.header("Tune Model")
confidence_threshold = st.sidebar.slider("Confidence Threshold (%)", 0, 100, 70) / 100.0
padding_pixels = st.sidebar.slider("Bounding Box Padding (pixels)", 0, 100, 30)

show_debug_view = st.sidebar.checkbox("Show Debug View (what the model sees)")

frame_placeholder = st.empty()
result_placeholder = st.empty()
stop_button = st.button("Stop Webcam")

# --- 5. Main Webcam Loop ---
cap = cv2.VideoCapture(0)
# This deque will store the last 10 predictions for smoothing
prediction_history = deque(maxlen=PREDICTION_HISTORY_LEN)
debug_placeholder = st.empty() if show_debug_view else None

while cap.isOpened() and not stop_button:
    ret, frame = cap.read()
    if not ret:
        st.write("Webcam feed ended.")
        break
        
    frame = cv2.flip(frame, 1) # Flip horizontally for a "mirror" view
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process with Mediapipe
    results = hands.process(frame_rgb)
    
    annotated_frame = frame.copy()
    final_prediction_text = "No hand detected"
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 1. Get Bounding Box
            bbox = get_hand_bounding_box(hand_landmarks, frame.shape, padding_pixels)
            cv2.rectangle(annotated_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            
            # 2. Crop the hand from the original frame
            cropped_hand = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            
            if cropped_hand.size > 0:
                # 3. Preprocess the cropped image
                img_for_model = preprocess_for_model(cropped_hand, MODEL_INPUT_SIZE)
                
                # Show debug view if checked
                if show_debug_view:
                    debug_img = cv2.resize(cropped_hand, MODEL_INPUT_SIZE)
                    debug_placeholder.image(debug_img, channels="BGR", 
                                            caption=f"Model Input ({MODEL_INPUT_SIZE[0]}x{MODEL_INPUT_SIZE[1]})")

                # 4. Predict
                prediction = model.predict(img_for_model)
                pred_index = np.argmax(prediction)
                confidence = np.max(prediction)
                
                # 5. Check confidence
                if confidence >= confidence_threshold:
                    pred_label = labels[pred_index]
                    prediction_history.append(pred_label)
                else:
                    prediction_history.append("Uncertain")
            
            # --- 6. Get the most stable prediction from history ---
            if prediction_history:
                # Find the most common prediction in the deque
                final_prediction_text = max(set(prediction_history), key=prediction_history.count)
                
            # Display the prediction on the frame
            cv2.putText(annotated_frame, final_prediction_text, (bbox[0], bbox[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        # If no hand is detected, clear the history
        prediction_history.clear()

    # Display the annotated webcam frame
    frame_placeholder.image(annotated_frame, channels="BGR")
    
    # Display the final, stable prediction in a large font
    result_placeholder.markdown(f"**Prediction:** ## {final_prediction_text}")

# --- 6. Cleanup ---
cap.release()
cv2.destroyAllWindows()
hands.close()
st.write("Webcam stopped.")