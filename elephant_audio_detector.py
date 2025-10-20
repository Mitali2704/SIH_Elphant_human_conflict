# elephant_audio_detector.py

import os
import numpy as np
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import sounddevice as sd
import streamlit as st

# ----------------------------
# STEP 1: Extract MFCC Features
# ----------------------------
def extract_mfcc_features(folder_path):
    features = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".wav"):
            file_path = os.path.join(folder_path, filename)
            y, sr = librosa.load(file_path, sr=44100)
            if len(y) == 0:
                continue  # skip empty files
            y = y / np.max(np.abs(y))  # normalize audio
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            mfcc_mean = np.mean(mfcc.T, axis=0)
            features.append(mfcc_mean)
    return features

# ----------------------------
# STEP 2: Streamlit App
# ----------------------------
def main():
    st.title("🐘 Real-Time Elephant Sound Detection (Audio-Based)")
    st.markdown("Detect elephant vocalizations using a local ML model trained on audio.")

    # 👉 Update these paths to your WAV folders
    elephant_wav = r"C:\Users\Mitali brahmankar\Downloads\elephant_sound_project\data\elephant_sound_wav"
    non_elephant_wav = r"C:\Users\Mitali brahmankar\Downloads\elephant_sound_project\data\non_elephant_sound_wav"

    st.info("🔍 Loading and processing audio data...")

    # Extract features
    elephant_features = extract_mfcc_features(elephant_wav)
    non_elephant_features = extract_mfcc_features(non_elephant_wav)

    if not elephant_features or not non_elephant_features:
        st.error("❌ Error: One or both folders are empty or have no valid .wav files.")
        return

    # Combine data
    X = np.array(elephant_features + non_elephant_features)
    y = np.array([1] * len(elephant_features) + [0] * len(non_elephant_features))

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train classifier
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)

    st.success(f"🎯 Model Trained! Accuracy: **{accuracy * 100:.2f}%**")

    st.markdown("Click the button below to record 7 seconds of audio and classify it.")

    duration = 7  # seconds
    fs = 44100    # sample rate

    if st.button("🎙️ Record and Detect"):
        st.info("🎧 Recording... Please make a sound or play elephant audio.")
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
        st.success("✅ Recording complete!")

        # Let user listen to the recorded audio
        st.audio(recording, sample_rate=fs)

        # Process recorded audio
        audio = recording.flatten()
        if np.max(np.abs(audio)) == 0:
            st.error("❌ No sound detected. Please try again.")
            return
        audio = audio / np.max(np.abs(audio))  # normalize

        mfcc = librosa.feature.mfcc(y=audio, sr=fs, n_mfcc=20)
        mfcc_mean = np.mean(mfcc.T, axis=0).reshape(1, -1)

        # Predict
        probabilities = clf.predict_proba(mfcc_mean)[0]
        elephant_prob = probabilities[1]
        non_elephant_prob = probabilities[0]

        st.markdown(f"🔬 Prediction Confidence:")
        st.write(f"- 🐘 Elephant: **{elephant_prob * 100:.2f}%**")
        st.write(f"- 🚫 Non-Elephant: **{non_elephant_prob * 100:.2f}%**")

        if elephant_prob > 0.6:  # Threshold can be adjusted
            st.error(f"🚨 Elephant Sound Detected! (Confidence: {elephant_prob * 100:.2f}%)")
        else:
            st.success(f"✅ No Elephant Sound Detected. (Confidence: {non_elephant_prob * 100:.2f}%)")

    st.markdown("---")
    st.caption("🚀 Created by Mitali for Human-Elephant Conflict Detection in Chhattisgarh")

# ----------------------------
# Run the app
# ----------------------------
if __name__ == "__main__":
    main()
