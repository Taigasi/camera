import sounddevice as sd
import numpy as np
import whisper
import wave

SAMPLE_RATE = 16000
DURACAO = 5  # segundos

print("Gravando 5 segundos... fale algo em voz alta:")
audio = sd.rec(int(DURACAO * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="int16")
sd.wait()
print("Gravação concluída.")

with wave.open("teste.wav", "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(audio.tobytes())

model = whisper.load_model("small")
result = model.transcribe("teste.wav", language="pt", task="transcribe", temperature=0.0)
print("TRANSCRIÇÃO:", result["text"])
