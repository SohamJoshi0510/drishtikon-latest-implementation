import os
import time
import datetime
import winsound
from google.cloud import texttospeech
from google.oauth2 import service_account

from core.utils import absolute_path, ensure_dir, load_credential_path
from core.logger import log

# ================================================================
#   DIRECTORIES
# ================================================================
AUDIO_DIR = absolute_path("results", "audio_outputs")
ensure_dir(AUDIO_DIR)

# ================================================================
#   GOOGLE CREDENTIALS
#   Each module uses its own cred folder → this module expects
#   /core/cred/tts-key.json  (You can rename as needed)
# ================================================================
CRED_PATH = load_credential_path("core", "tts-key.json")

try:
    TTS_CREDENTIALS = service_account.Credentials.from_service_account_file(
        CRED_PATH
    )
    tts_client = texttospeech.TextToSpeechClient(credentials=TTS_CREDENTIALS)
except Exception as e:
    print(f"[TTS] ERROR loading credentials from {CRED_PATH}: {e}")
    tts_client = None


# ================================================================
#   SPEAK FUNCTION
# ================================================================
def speak(text: str):
    """
    Convert text → speech using Google Cloud TTS.
    Windows-safe playback using winsound.
    Logs timing and stores WAV file.
    """

    if not tts_client:
        print("[TTS] Client not initialized.")
        return

    t0 = time.time()

    synthesis_input = texttospeech.SynthesisInput(text=text)

    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16
    )

    try:
        response = tts_client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )
    except Exception as e:
        log("TTS", "-", f"TTS ERROR: {e}")
        return

    # Save file
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    audio_path = absolute_path("results", "audio_outputs", f"tts_{ts}.wav")

    try:
        with open(audio_path, "wb") as f:
            f.write(response.audio_content)
    except Exception as e:
        log("TTS", "-", f"File write error: {e}")
        return

    # Play sound (Windows only)
    try:
        winsound.PlaySound(audio_path, winsound.SND_FILENAME)
    except Exception as e:
        print(f"[TTS] Playback error: {e}")

    t1 = time.time()
    log("TTS", audio_path, f"Played {len(text)} chars", round(t1 - t0, 2))
