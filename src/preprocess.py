import librosa
import numpy as np
import pandas as pd
from pathlib import Path

def audio_to_feet(audio, sr, minMs=200, thresholdPercent=60):
    frameLength = 2048
    hopLength = 512
    
    S = librosa.stft(audio, n_fft=frameLength, hop_length=hopLength)
    magnitude = np.abs(S)
    rms = np.sqrt(np.mean(magnitude**2, axis=0))
    rms = (rms - rms.min()) / (rms.max() - rms.min() + 1e-8)
    
    threshold = np.percentile(rms, thresholdPercent)
    
    from scipy.signal import find_peaks
    minDistanceFrames = int(minMs * sr / (1000 * hopLength))
    peaks, _ = find_peaks(rms, height=threshold, distance=minDistanceFrames)
    
    peakSample = peaks * hopLength
    return peakSample

def extract_mfcc(audio, sr, n_mfcc):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfccMean = np.mean(mfcc, axis=1)
    mfccStd = np.std(mfcc, axis=1)
    features = np.concatenate([mfccMean, mfccStd])
    return features

def process_audio_file(path, sr=48000, window_duration=0.35, hop_duration=0.05):
    y, srLoad = librosa.load(path, sr=sr, mono=True)
    if srLoad != sr:
        print(f"Warning: resampled {path} to {sr}Hz")
    
    peaks = audio_to_feet(y,sr)
    if len(peaks) == 0:
        return []
    
    windowSamples = int(window_duration * sr)
    offsetSamples = int(0.10 * sr)
    
    allFeatures = []
    for peak in peaks:
        start = max(0, peak-offsetSamples)
        end = min(len(y), start+windowSamples)
        if end - start < windowSamples//2:
            continue
        feetAudio = y[start:end]
        features = extract_mfcc(feetAudio, sr, n_mfcc=20)
        allFeatures.append(features)
        
    return allFeatures

def save_dataset(inputPath="dataset_raw", outputPath="features.csv"):
    rawPath = Path(inputPath)
    allRows = []
    for materialFolder in rawPath.iterdir():
        if not materialFolder.is_dir():
            continue
        material = materialFolder.name
        
        for audioFile in materialFolder.glob("*.wav"):
            agent = audioFile.stem.rsplit('_',1)[0]
            print(f"processing {material} | {agent}")
            featuresList = process_audio_file(str(audioFile))
            
            for features in featuresList:
                row = list(features) + [agent,material]
                allRows.append(row)
    featuresCols = [f"feature_{i}" for i in range(40)]
    df = pd.DataFrame(allRows, columns=featuresCols + ["agent", "material"])
    df.to_csv(outputPath,index=False)
    print(f"\nDataset saved: {outputPath}")
    print(f"Total footsteps: {len(df)}")
    print(f"Agents: {df['agent'].unique()}")
    print(f"Materials: {df['material'].unique()}")
    
    return df

if __name__ == "__main__":
    df = save_dataset(inputPath="dataset_raw", outputPath="features.csv")
    