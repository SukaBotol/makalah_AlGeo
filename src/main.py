import librosa
import numpy as np
import joblib
from pathlib import Path
from scipy.signal import find_peaks

# Import functions from prepare_dataset.py
# (or you can copy-paste them here)
def detect_footsteps(audio, sr, min_distance_ms=200, threshold_percentile=60):
    frame_length = 2048
    hop_length = 512
    S = librosa.stft(audio, n_fft=frame_length, hop_length=hop_length)
    magnitude = np.abs(S)
    rms_energy = np.sqrt(np.mean(magnitude**2, axis=0))
    rms_energy = (rms_energy - rms_energy.min()) / (rms_energy.max() - rms_energy.min() + 1e-8)
    threshold = np.percentile(rms_energy, threshold_percentile)
    min_distance_frames = int(min_distance_ms * sr / (1000 * hop_length))
    peaks, _ = find_peaks(rms_energy, height=threshold, distance=min_distance_frames)
    peak_samples = peaks * hop_length
    return peak_samples

def extract_mfcc_features(audio, sr, n_mfcc=20):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    features = np.concatenate([mfcc_mean, mfcc_std])
    return features


# ============================================================
# STEP 1: Load trained models
# ============================================================
print("Loading trained models...")
scaler = joblib.load("scaler.joblib")
pca = joblib.load("pca.joblib")
agent_classifier = joblib.load("agent_model.joblib")
material_classifier = joblib.load("material_model.joblib")
print("Models loaded!")


# ============================================================
# STEP 2: Process new audio file
# ============================================================
def predict_footstep_labels(audio_filepath, sr=48000):
    """
    Input: path to WAV file
    Output: (agent, agent_confidence, material, material_confidence)
    """
    
    print(f"\nProcessing: {audio_filepath}")
    
    # Load audio
    y, sr_loaded = librosa.load(audio_filepath, sr=sr, mono=True)
    
    # Detect footsteps
    peaks = detect_footsteps(y, sr)
    print(f"Detected {len(peaks)} footsteps")
    
    if len(peaks) == 0:
        print("No footsteps detected!")
        return None
    
    # Extract features for each footstep
    window_samples = int(0.35 * sr)
    offset_samples = int(0.10 * sr)
    
    all_features = []
    for peak in peaks:
        start = max(0, peak - offset_samples)
        end = min(len(y), start + window_samples)
        if end - start < window_samples // 2:
            continue
        footstep_audio = y[start:end]
        features = extract_mfcc_features(footstep_audio, sr, n_mfcc=20)
        all_features.append(features)
    
    all_features = np.array(all_features)  # Shape: (num_footsteps, 40)
    
    # ========================================================
    # STEP 3: Transform through pipeline (normalize → PCA)
    # ========================================================
    X_scaled = scaler.transform(all_features)  # Shape: (num_footsteps, 40)
    X_pca = pca.transform(X_scaled)            # Shape: (num_footsteps, 20)
    
    # ========================================================
    # STEP 4: Predict for each footstep
    # ========================================================
    agent_predictions = agent_classifier.predict(X_pca)
    material_predictions = material_classifier.predict(X_pca)
    
    # Get prediction probabilities (for confidence)
    agent_decision = agent_classifier.decision_function(X_pca)
    material_decision = material_classifier.decision_function(X_pca)
    
    # ========================================================
    # STEP 5: Majority voting
    # ========================================================
    """
    Final prediction = most common label across all footsteps
    Confidence = proportion of votes for winner
    """
    
    # Agent voting
    agent_unique, agent_counts = np.unique(agent_predictions, return_counts=True)
    agent_winner_idx = np.argmax(agent_counts)
    agent_final = agent_unique[agent_winner_idx]
    agent_confidence = agent_counts[agent_winner_idx] / len(agent_predictions)
    
    # Material voting
    material_unique, material_counts = np.unique(material_predictions, return_counts=True)
    material_winner_idx = np.argmax(material_counts)
    material_final = material_unique[material_winner_idx]
    material_confidence = material_counts[material_winner_idx] / len(material_predictions)
    
    return agent_final, agent_confidence, material_final, material_confidence


# ============================================================
# STEP 6: Run prediction on a test file
# ============================================================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_audio_file>")
        print("Example: python predict.py dataset_raw/metal/Jett_01.wav")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    result = predict_footstep_labels(audio_file)
    
    if result is not None:
        agent, agent_conf, material, material_conf = result
        
        print("\n" + "="*50)
        print("PREDICTION RESULTS")
        print("="*50)
        print(f"Agent:   {agent} (confidence: {agent_conf:.1%})")
        print(f"Material: {material} (confidence: {material_conf:.1%})")
        print("="*50)