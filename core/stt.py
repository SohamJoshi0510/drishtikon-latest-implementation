import os
import time
import numpy as np
import sounddevice as sd
from google.cloud import speech
from google.oauth2 import service_account

from core.utils import absolute_path, load_credential_path
from core.logger import log

# ================================================================
#   MICROPHONE SETTINGS
# ================================================================
SAMPLE_RATE = 16000   # Google recommended
CHANNELS = 1          # Mono


# ================================================================
#   GOOGLE CREDENTIALS
#   Each module uses its own cred/ folder.
#   For STT:  core/cred/stt-key.json
# ================================================================
CRED_PATH = load_credential_path("core", "stt-key.json")

try:
    STT_CREDENTIALS = service_account.Credentials.from_service_account_file(
        CRED_PATH
    )
    speech_client = speech.SpeechClient(credentials=STT_CREDENTIALS)
except Exception as e:
    print(f"[STT] ERROR loading credentials from {CRED_PATH}: {e}")
    speech_client = None


# ================================================================
#   AUDIO RECORDING
# ================================================================
def record_audio(duration=4):
    """
    Records audio for `duration` seconds using sounddevice.
    Returns raw bytes suitable for Google STT.
    """

    print(f"[STT] Recording {duration}s...")

    try:
        audio = sd.rec(
            int(duration * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16"
        )
        sd.wait()

    except Exception as e:
        log("STT", "-", f"Microphone recording error: {e}")
        return None

    print("[STT] Recording complete.")
    return audio.tobytes()


# ================================================================
#   GOOGLE SPEECH-TO-TEXT
# ================================================================
def speech_to_text(audio_bytes):
    """
    Sends recorded audio to Google Speech-to-Text and returns transcript.
    """

    if not speech_client:
        print("[STT] Client not initialized.")
        return None

    audio = speech.RecognitionAudio(content=audio_bytes)

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=SAMPLE_RATE,
        language_code="en-US"
    )

    try:
        response = speech_client.recognize(config=config, audio=audio)
    except Exception as e:
        log("STT", "-", f"Google STT error: {e}")
        return None

    if not response.results:
        return None

    return response.results[0].alternatives[0].transcript


# ================================================================
#   PUBLIC LISTEN() FUNCTION
# ================================================================
def listen(duration=4):
    """
    High-level wrapper:
    - Records audio
    - Sends to Google STT
    - Logs time taken
    - Returns recognized speech
    """

    t0 = time.time()

    audio_bytes = record_audio(duration)
    if not audio_bytes:
        return None

    text = speech_to_text(audio_bytes)

    t1 = time.time()
    log("STT", "-", f"Heard '{text}'" if text else "No speech detected", round(t1 - t0, 2))

    if text:
        print(f"[STT] Heard:", text)
    else:
        print("[STT] No speech detected.")

    return text
