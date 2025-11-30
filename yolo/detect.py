import cv2
import time
import threading
import subprocess
import os
from ultralytics import YOLO

from google.cloud import texttospeech
from google.oauth2 import service_account
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

############################################################
#                      CONFIG
############################################################

# --- GEMINI CONFIG ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

ENV_UPDATE_INTERVAL = 60  # seconds
last_gemini_time = 0
gemini_busy = False

# --- GOOGLE TTS ---
TTS_CREDENTIALS = service_account.Credentials.from_service_account_file(
    r"cred\imperial-glyph-448202-p6-d36e2b69bd92.json"
)
tts_client = texttospeech.TextToSpeechClient(credentials=TTS_CREDENTIALS)


############################################################
#                      TEXT TO SPEECH
############################################################

def speak(message):
    """Direct, synchronous TTS with WAV playback."""
    print("Speaking:", message)

    synthesis_input = texttospeech.SynthesisInput(text=message)

    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16
    )

    response = tts_client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
    )

    # Unique temporary output file
    temp_audio = f"temp_{int(time.time() * 1000)}.wav"
    with open(temp_audio, "wb") as f:
        f.write(response.audio_content)

    subprocess.run([
        "powershell", "-NoProfile", "-Command",
        f"(New-Object Media.SoundPlayer '{temp_audio}').PlaySync()"
    ])

    # Cleanup temp file
    try:
        os.remove(temp_audio)
    except:
        pass


############################################################
#                  GEMINI SCENE ANALYSIS
############################################################

def gemini_scene(frame):
    """Snapshot scene description using Gemini Vision."""
    global gemini_busy
    gemini_busy = True

    start_total = time.time()
    print("\n------- GEMINI CALL STARTED -------")
    print("Timestamp:", start_total)

    try:
        if not GEMINI_API_KEY or not GEMINI_MODEL:
            print("❌ Gemini missing API key or model name.")
            return

        # ---- Encode frame ----
        encode_start = time.time()
        print("Encoding frame...")
        success, encoded = cv2.imencode(".jpg", frame)
        encode_end = time.time()

        print(f"Encode success: {success}, time: {encode_end - encode_start:.4f}s")
        if not success:
            print("❌ Frame encoding failed.")
            return

        image_bytes = encoded.tobytes()

        # ---- Create Gemini model ----
        model = genai.GenerativeModel(GEMINI_MODEL)

        prompt = """
        Describe this scene in a short, clear, concise paragraph.
        Mention objects, people, actions, layout, and anything important(40 WORDS ONLY).
        """

        # ---- Gemini Call ----
        call_start = time.time()
        print("Sending request to Gemini...")
        res = model.generate_content(
            [
                {"mime_type": "image/jpeg", "data": image_bytes},
                prompt,
            ]
        )
        call_end = time.time()
        print(f"Gemini call completed in {call_end - call_start:.4f}s")

        # ---- Extract text ----
        text = getattr(res, "text", None)
        print("Gemini output:", text)

        if not text:
            print("⚠ Gemini returned NO text.")
            return

        # ---- Speak result ----
        speak("Gemini summary: " + text)

    except Exception as e:
        print("❌ Gemini ERROR:", e)

    finally:
        gemini_busy = False
        end_total = time.time()
        print(f"------- GEMINI DONE (Total {end_total - start_total:.4f}s) -------\n")


############################################################
#                  YOLO HELPERS
############################################################

model = YOLO("yolov8n.pt")

def describe_yolo(boxes, names):
    if not boxes:
        return "I do not see any objects."

    counts = {}
    for cls_idx in boxes:
        label = names[int(cls_idx)]
        counts[label] = counts.get(label, 0) + 1

    parts = []
    for label, count in counts.items():
        if count == 1:
            parts.append(f"a {label}")
        else:
            parts.append(f"{count} {label}s")

    return "I can see " + ", ".join(parts) + "."


############################################################
#                       MAIN LOOP
############################################################

def main():
    global last_gemini_time

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera not found.")
        return

    print("Running... Press 'Y' for YOLO burst, 'G' for Gemini, 'Q' to quit.")

    yolo_burst_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        key = cv2.waitKey(1) & 0xFF

        # ---- Quit ----
        if key == ord('q'):
            print("Exiting...")
            break

        # ---- YOLO Burst Trigger ----
        if key == ord('y'):
            print("YOLO burst triggered (1 frame).")
            yolo_burst_frames = 1

        # ---- Gemini Trigger ----
        if key == ord('g'):
            now = time.time()
            cooldown = now - last_gemini_time
            print(f"'G' pressed. Cooldown = {cooldown:.1f}s")

            if not gemini_busy and cooldown >= ENV_UPDATE_INTERVAL:
                print("Starting Gemini call...")
                last_gemini_time = now

                threading.Thread(
                    target=gemini_scene,
                    args=(frame.copy(),),
                    daemon=True
                ).start()
            else:
                print("Gemini busy or cooldown not finished.")

        # ---- YOLO Processing ----
        if yolo_burst_frames > 0:
            results = model.predict(frame, verbose=False)
            annotated = results[0].plot()

            box_classes = [box.cls[0] for box in results[0].boxes]
            desc = describe_yolo(box_classes, results[0].names)

            print("YOLO:", desc)
            speak("YOLO update: " + desc)

            yolo_burst_frames -= 1
            cv2.imshow("Vision", annotated)
            os.makedirs("annotated_images", exist_ok=True)
            cv2.imwrite("annotated_images/annotated_{int(time.time())}.png", annotated)
        else:
            cv2.imshow("Vision", frame)

    cap.release()
    cv2.destroyAllWindows()


############################################################

if __name__ == "__main__":
    main()
