from gtts import gTTS
import os

# Nervous sample text
text = """
Goooad Morning Everyoneee
i aaam s-eecond Ye--ar Stu---dent 
at V-i-it Bhop--al Univer----sity
"""

# Generate nervous voice audio
tts = gTTS(text=text, lang='en', slow=True)

# Save as wav file
tts.save("nervous_speech.wav")

print("Nervous speech saved as 'nervous_speech.wav'")