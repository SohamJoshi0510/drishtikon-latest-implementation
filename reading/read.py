import tkinter as tk
from tkinter import filedialog
from google.cloud import texttospeech, speech, vision
import google.generativeai as genai
from google.oauth2 import service_account
import cv2
import os
import io
import sys
import time
import datetime
from PIL import Image
from dotenv import load_dotenv

# Always lock CWD to script directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

############################################################
#                 LOAD ENVIRONMENT
############################################################
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL")
genai.configure(api_key=GEMINI_API_KEY)

############################################################
#                 GOOGLE CREDENTIALS
############################################################
CRED_DIR = os.path.join(BASE_DIR, "cred")

STT_TTS_CREDENTIALS = service_account.Credentials.from_service_account_file(
    os.path.join(CRED_DIR, "imperial-glyph-448202-p6-d36e2b69bd92.json")
)

VISION_CREDENTIALS = service_account.Credentials.from_service_account_file(
    os.path.join(CRED_DIR, "vision-key.json")
)

############################################################
#                 DIRECTORIES
############################################################
RES = os.path.join(BASE_DIR, "results")
AUDIO_DIR = os.path.join(RES, "audio_outputs")
os.makedirs(AUDIO_DIR, exist_ok=True)

############################################################
#                 CLIENTS
############################################################
tts_client = texttospeech.TextToSpeechClient(credentials=STT_TTS_CREDENTIALS)
stt_client = speech.SpeechClient(credentials=STT_TTS_CREDENTIALS)
vision_client = vision.ImageAnnotatorClient(credentials=VISION_CREDENTIALS)


############################################################
#                 TEXT TO SPEECH (NO SUBPROCESS)
############################################################
def speak(text):
    """
    Convert text to speech and play it using pure Python.
    No subprocess, no external calls.
    """
    synthesis_input = texttospeech.SynthesisInput(text=text)

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

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    audio_path = os.path.join(AUDIO_DIR, f"tts_{ts}.wav")

    with open(audio_path, "wb") as out:
        out.write(response.audio_content)

    # Playback using Media.SoundPlayer through ctypes (no subprocess)
    try:
        import winsound
        winsound.PlaySound(audio_path, winsound.SND_FILENAME)
    except Exception:
        pass


############################################################
#                 SPEECH INPUT (DISCONNECTED)
############################################################
def listen():
    """Placeholder STT function — currently returns None (intentionally)."""
    return None


############################################################
#               GOOGLE VISION OCR
############################################################
def extract_text_with_google_vision(image_bgr):
    """Runs Google Cloud Vision OCR on a BGR OpenCV image."""
    success, encoded_image = cv2.imencode(".jpg", image_bgr)
    if not success:
        return ""

    image = vision.Image(content=encoded_image.tobytes())
    response = vision_client.text_detection(image=image)

    if response.error.message:
        print("Google Vision OCR Error:", response.error.message)
        return ""

    annotations = response.text_annotations
    return annotations[0].description if annotations else ""


############################################################
#              GEMINI MULTIMODAL CORRECTION
############################################################

def optimize_image(image_path):
    img = Image.open(image_path)
    img.thumbnail((1800, 1800))  # Resize
   
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return buf.getvalue()

def refine_text_with_gemini_and_image(prompt_text, image_path):
    if not GEMINI_API_KEY:
        return prompt_text
    
    optimized_bytes = optimize_image(image_path)
    model = genai.GenerativeModel("gemini-2.5-flash")

    final_text = []
    mime_type = "image/jpeg"

    gemini_start = time.time()

    # --- The correct streaming method for your SDK ---
    response = model.generate_content(
        [
            {"mime_type": mime_type, "data": optimized_bytes},
             prompt_text 
        ],
        stream=True  # This is the actual streaming flag
    )

    for chunk in response:   # <-- this yields chunks
        if hasattr(chunk, "text") and chunk.text:
            final_text.append(chunk.text)

    gemini_end = time.time()
    time_taken = gemini_end - gemini_start
    print("Gemini time (optimized):", time_taken)

    return "".join(final_text), time_taken
############################################################

def get_mime_type(path):
    ext = path.lower().split(".")[-1]
    mapping = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "gif": "image/gif",
        "webp": "image/webp",
        "bmp": "image/bmp",
    }
    return mapping.get(ext, "image/jpeg")


############################################################
#               FILE CHOOSER (GUI)
############################################################
def choose_file():
    root = tk.Tk()
    root.withdraw()
    filepath = filedialog.askopenfilename(
        initialdir="images",
        title="Select an image file",
        filetypes=(
            ("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp"),
            ("All files", "*.*"),
        ),
    )
    root.destroy()
    return filepath


############################################################
#                IMAGE CAPTURE
############################################################
def capture_image_on_space():
    """
    Opens webcam feed and captures a single frame on SPACE.
    """
    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        speak("Camera not found.")
        return None, None

    img_counter = 0
    temp_image_path = None

    while True:
        ret, frame = cam.read()
        if not ret:
            speak("Camera disconnected.")
            break

        cv2.imshow("Press SPACE to capture, Q to quit", frame)
        k = cv2.waitKey(1)

        if k == 32:  # SPACE
            temp_image_path = f"captured_{img_counter}.png"
            cv2.imwrite(temp_image_path, frame)
            break

        elif k == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

    return temp_image_path, frame

# ##########################################################
# LOGGER
# ##########################################################
def log(service="", image_path="", message="", time_taken=None):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        os.makedirs(RES, exist_ok=True)
        with open(os.path.join(RES, "app.log"), "a") as f:
            f.write(f"[{ts}]\t{service}: {image_path}: {len(message)}\t(Time taken: {time_taken}s)\n" if time_taken is not None else f"[{ts}] {service}: {image_path}: {message}\n")
    except Exception as e:
        print(f"Error creating log directory: {e}", file=sys.stderr)

############################################################
#                      MAIN LOGIC
############################################################
def main():

    # Step 1: Try picking a file
    speak("Select an image file, or press cancel to use the camera.")
    img_path = choose_file()

    # Step 2: If user CANCELS → go to VIDEO MODE
    if not img_path:
        print("[INFO] No file selected. Switching to video mode...")

        speak("No image selected. Switching to video mode.")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            speak("Camera could not be opened.")
            return

        speak("Press space to capture, or escape to exit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            cv2.imshow("Live Camera - Press SPACE to capture", frame)
            key = cv2.waitKey(1)

            if key == 32:  # Spacebar → capture
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                img_path = os.path.join("results", f"capture_{ts}.jpg")
                cv2.imwrite(img_path, frame)
                print(f"[INFO] Captured: {img_path}")
                break

            elif key == 27:  # ESC to exit
                cap.release()
                cv2.destroyAllWindows()
                speak("Exiting video mode.")
                return

        cap.release()
        cv2.destroyAllWindows()

    # Step 3: Process the chosen or captured image
    refinement_prompt = f"""
                    If the image contains:
                    A lot of text => SUMMARIZE contextual elements in 20 WORDS, then include the complete, UNSUMMARIZED text content.
                    Medical text => Issue alarms and include UNSUMMARIZED text.
                    Little text => SUMMARIZE contextual elements along with text in 25 WORDS.
                    No text => Say "NO TEXT FOUND." and SUMMARIZE the visual in 15 WORDS.
                    DO NOT include asterisks, quotes, or any formatting.
                    """

    refined, time_taken = refine_text_with_gemini_and_image(refinement_prompt, img_path)
    log("Gemini Multimodal", img_path,refined, time_taken)


    print("\nRefined OCR result:\n", refined)
    speak(refined)

############################################################
if __name__ == "__main__":
    main()
