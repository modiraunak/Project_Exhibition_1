from pydub import AudioSegment

# Load MP3
audio = AudioSegment.from_mp3("nervous_speech.mp3")

# Export as WAV
audio.export("nervous_speech.wav", format="wav")

print("Converted nervous_speech.wav successfully!")