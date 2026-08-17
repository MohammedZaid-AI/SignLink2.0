import streamlit as st
import os
import time
import speech_recognition as sr

# --- Configuration ---
IMAGE_DIR = "sign_images" 
TIME_PER_LETTER = 0.7 

st.title("Text-to-Fingerspelling App 🗣️➡️🖐️")

# --- Text-to-Sign (Fingerspelling) Function (No changes needed) ---
def display_fingerspelling(text, image_placeholder):
    """
    Displays the sign language images for each letter in the text.
    """
    # Convert text to uppercase
    text = text.upper()
    
    st.info(f"Fingerspelling: {text}")
    
    for char in text:
        if 'A' <= char <= 'Z':
            image_path = os.path.join(IMAGE_DIR, f"{char}.jpg")
        elif char == ' ':
            image_path = os.path.join(IMAGE_DIR, "space.jpg")
        else:
            # Skip characters we don't have images for (like '?' or '!')
            continue 

        if os.path.exists(image_path):
            image_placeholder.image(image_path, width=300)
            # Pause to create the animation effect
            time.sleep(TIME_PER_LETTER)
        else:
            st.warning(f"Warning: Image file not found for '{char}'")
            # Show a blank space even if the file is missing
            image_placeholder.empty() 
            time.sleep(TIME_PER_LETTER)
        
    time.sleep(1)
    image_placeholder.empty()
    st.success("Fingerspelling complete!")

# --- MODIFIED: Speech-to-Text Function ---
def recognize_speech_from_mic(limit_seconds):
    """Listens for speech via microphone for a set time limit."""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        # --- CHANGED: Tell user the time limit ---
        st.info(f"Listening for {limit_seconds} seconds... 🎤")
        
        r.adjust_for_ambient_noise(source, duration=0.5)
        
        # --- CHANGED: Added timeout and phrase_time_limit ---
        try:
            audio = r.listen(source, timeout=limit_seconds, phrase_time_limit=limit_seconds)
        except sr.WaitTimeoutError:
            st.warning("Listening timed out. Please try speaking again.")
            return None
    
    try:
        text = r.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        st.warning("Sorry, I could not understand the audio.")
        return None
    except sr.RequestError as e:
        st.error(f"Could not request results from Google Speech Recognition; {e}")
        return None

# --- Main App Layout (MODIFIED) ---

st.header("Translate English to Sign")
st.write("Type a word or sentence, or use your voice.")

# 1. This is the "screen" where the avatar's hands will appear.
avatar_placeholder = st.empty()
col1, col2 = st.columns(2)

with col1:
    st.subheader("Type your message:")
    user_text = st.text_input("Type here:")
    
    if st.button("Show Signs (from Text)"):
        if user_text:
            display_fingerspelling(user_text, avatar_placeholder)
        else:
            st.warning("Please type a message first.")

with col2:
    st.subheader("Speak your message:")
    
    # --- NEW: Added a slider for the time limit ---
    time_limit = st.slider("Set recording time limit (seconds)", 1, 10, 5) # Default 5 seconds
    
    if st.button("Speak Now 🎤"):
        # --- CHANGED: Pass the time limit to the function ---
        speech_text = recognize_speech_from_mic(limit_seconds=time_limit)
        
        if speech_text:
            st.success(f"You said: {speech_text}")
            time.sleep(0.5) # Short pause to let user read the text
            # Call the *same* animation function
            display_fingerspelling(speech_text, avatar_placeholder)