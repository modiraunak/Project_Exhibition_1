import librosa
import matplotlib.pyplot as plt

# Load audio file
y, sr = librosa.load("sample_speech.wav")

# Extract pitch (fundamental frequency)
pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

pitch_values = []
for t in range(pitches.shape[1]):
    index = magnitudes[:, t].argmax()
    pitch = pitches[index, t]
    if pitch > 0:  # filter out silence
        pitch_values.append(pitch)

# Plot pitch variation
plt.plot(pitch_values)
plt.title("Pitch Variation Over Time")
plt.xlabel("Frame")
plt.ylabel("Frequency (Hz)")
plt.show()