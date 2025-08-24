from gtts import gTTS
import os

# Nervous sample text
text = """
Uh… g-good morning everyone… I-I am here to… um… 
talk about, uh… confidence. 
So… so the main th-thing is… umm… we often feel… uh… 
nervous when speaking in front of people.
"""

# Generate nervous voice audio
tts = gTTS(text=text, lang='en', slow=False)

# Save as wav file
tts.save("nervous_speech.mp3")

print("Nervous speech saved as 'nervous_speech.mp3'")