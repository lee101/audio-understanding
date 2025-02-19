import numpy as np  # type: ignore
import librosa  # type: ignore
from librosa import feature  # type: ignore

# Example 1: Using librosa features with reduced sensitivity

def extract_features(y, sr):
    mfcc = feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=1024)  # Reduced from 40 to 13 coefficients
    chroma = feature.chroma_stft(y=y, sr=sr, hop_length=1024, n_fft=2048)  # Larger window
    contrast = feature.spectral_contrast(y=y, sr=sr, hop_length=1024)  # Larger hop length
    return np.vstack([mfcc, chroma, contrast])

# Example 2: Onset detection with larger window and threshold

def detect_onsets(y, sr):
    # Increased hop length and added smoothing
    onsets = librosa.onset.onset_detect(y=y, sr=sr, hop_length=1024, 
                                      pre_max=5, post_max=5, delta=0.2)
    onset_times = librosa.frames_to_time(onsets, sr=sr, hop_length=1024)
    return onset_times

# Example 3: Beat tracking with tighter tempo constraints

def detect_beats(y, sr):
    # Constrained tempo range and larger hop length
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=1024,
                                         start_bpm=60, tightness=100)
    beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=1024)
    return tempo, beat_times

# Example 4: Tempogram with larger analysis window

def extract_tempogram(y, sr):
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=1024)
    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=1024)
    return tempogram

# Example 5: Spectral features with larger windows

def compute_spectral_features(y, sr):
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=1024, n_fft=2048)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=1024, n_fft=2048)
    zero_cross = librosa.feature.zero_crossing_rate(y, frame_length=2048, hop_length=1024)
    return centroid, rolloff, zero_cross

# Example usage remains same
if __name__ == "__main__":
    filename = 'a-call-to-the-soul-relax-calm.mp3'
    y, sr = librosa.load(filename)
    features = extract_features(y, sr)
    onsets = detect_onsets(y, sr)
    tempo, beats = detect_beats(y, sr)
    tempogram = extract_tempogram(y, sr)
    centroid, rolloff, zero_cross = compute_spectral_features(y, sr)
    
    print("Features shape:", features.shape)
    print("Onset times:", onsets)
    print("Tempo:", tempo)
    print("Beat times:", beats)
    print("Tempogram shape:", tempogram.shape)
    print("Spectral Centroid shape:", centroid.shape)
    print("Spectral Rolloff shape:", rolloff.shape)
    print("Zero Crossing Rate shape:", zero_cross.shape)