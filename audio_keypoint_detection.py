import numpy as np  # type: ignore
import librosa  # type: ignore
from librosa import feature  # type: ignore
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os
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

    # Visualize features and waveform in video

    
    # Create visualization frames
    n_frames = len(y)
    frame_rate = 30
    duration = len(y) / sr
    n_viz_frames = int(duration * frame_rate)
    
    if not os.path.exists('frames'):
        os.makedirs('frames')
    
    # Generate frames
    for i in range(n_viz_frames):
        time_point = i / frame_rate
        sample_point = int(time_point * sr)
        
        plt.figure(figsize=(12, 8))
        
        # Plot waveform
        plt.subplot(4, 1, 1)
        window = 10000  # Show 10000 samples around current point
        start = max(0, sample_point - window//2)
        end = min(len(y), sample_point + window//2)
        plt.plot(y[start:end])
        plt.axvline(x=window//2, color='r')
        plt.title('Waveform')
        
        # Plot onsets
        plt.subplot(4, 1, 2)
        plt.vlines(onsets * sr/1024, 0, 1, color='r', alpha=0.5)  # Convert seconds to frames
        plt.axvline(x=sample_point/1024, color='b')  # Current position
        plt.title('Onsets')
        
        # Plot tempogram
        plt.subplot(4, 1, 3)
        plt.imshow(tempogram, aspect='auto', origin='lower')
        curr_frame = int(sample_point/1024)
        plt.axvline(x=curr_frame, color='w')
        plt.title('Tempogram')
        
        # Plot spectral features
        plt.subplot(4, 1, 4)
        plt.plot(centroid[0], label='Centroid')
        plt.plot(rolloff[0], label='Rolloff')
        plt.plot(zero_cross[0] * 1000, label='Zero Crossing Rate x1000')
        plt.axvline(x=curr_frame, color='r')
        plt.legend()
        plt.title('Spectral Features')
        
        plt.tight_layout()
        plt.savefig(f'frames/frame_{i:05d}.png')
        plt.close()
        
    # Combine frames into video using ffmpeg
    cmd = [
        'ffmpeg', '-y',
        '-framerate', str(frame_rate),
        '-i', 'frames/frame_%05d.png',
        '-i', filename,  # Add original audio
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-c:a', 'aac',
        'visualization.mp4'
    ]
    subprocess.run(cmd)
    
    # Cleanup frames
    for file in os.listdir('frames'):
        os.remove(os.path.join('frames', file))
    os.rmdir('frames')