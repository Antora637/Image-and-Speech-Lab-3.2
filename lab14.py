import speech_recognition as sr

# Create recognizer object
recognizer = sr.Recognizer()

# Load speech file
with sr.AudioFile("voice.wav") as source:
    audio = recognizer.record(source)

# Recognize speech
try:
    result = recognizer.recognize_google(audio)

    print("Recognized Speech:")
    print(result)

except sr.UnknownValueError:
    print("Sorry, speech could not be understood.")

except sr.RequestError:
    print("Could not connect to the recognition service.")